#!/usr/bin/env python3
"""Reproducible two-stage training for ESMDynamic.

Stage 1 uses ``--dataset-type rcsb`` and trains a one-condition dynamic head.
Stage 2 uses ``--dataset-type mdcath`` and can initialize its five-condition
dynamic head from the stage-1 checkpoint with ``--init-checkpoint``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset

from esm.esmfold.v1.misc import batch_encode_sequences
from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmdynamic.training.data_reader import (
    ESMDynamicDataset,
    collate_samples,
    read_identifiers,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # Training remains usable without the optional UI package.
    SummaryWriter = None


LOSS_NAMES = {
    "dynamic_logits",
    "dynamic_confidence",
    "frequency_pred",
    "frequency_residual_pred",
    "kinetic_logits",
    "kinetic_confidence",
}


def distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return rank() == 0


def main_print(*values: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*values, **kwargs)


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic weighted draws partitioned across DDP ranks."""

    def __init__(self, weights: torch.Tensor, total_samples: int, seed: int) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        self.total_samples = total_samples
        self.seed = seed
        self.epoch = 0
        self.samples_per_rank = math.ceil(total_samples / world_size())
        self.padded_total = self.samples_per_rank * world_size()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.padded_total, replacement=True, generator=generator
        )
        return iter(indices[rank():self.padded_total:world_size()].tolist())

    def __len__(self) -> int:
        return self.samples_per_rank


def parse_list(value: str) -> list[str]:
    return shlex.split(value.replace(",", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dataset-type", choices=("rcsb", "mdcath"), required=True)
    parser.add_argument("--train-identifiers-file", "--train_identifiers_file", required=True)
    parser.add_argument("--val-identifiers-file", "--val_identifiers_file", required=True)
    parser.add_argument("--data-dir", "--data_dir", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--train-weight-file", default=None)
    parser.add_argument("--val-weight-file", default=None)
    parser.add_argument("--kin-class-weights", "--kin_class_weights", default=None)
    parser.add_argument("--loss-heads", "--loss_heads", type=parse_list, required=True)
    parser.add_argument("--init-checkpoint", "--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--batch-accum", "--batch_accum", type=int, default=16)
    parser.add_argument("--crop-length", type=int, default=256)
    parser.add_argument("--train-samples-per-epoch", "--train_samples_per_epoch", type=int, default=0)
    parser.add_argument("--val-samples-per-epoch", "--val_samples_per_epoch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=128)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--metric-condition", type=int, default=0)
    parser.add_argument("--aux-head-start-epoch", type=int, default=11,
                        help="1-based epoch that enables confidence/residual losses")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--weighted-random-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def load_tensor(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, Any, Any]:
    train_ids = read_identifiers(args.train_identifiers_file)
    val_ids = read_identifiers(args.val_identifiers_file)
    train_weights = load_tensor(args.train_weight_file)
    val_weights = load_tensor(args.val_weight_file)
    train_set = ESMDynamicDataset(
        args.data_dir, train_ids, args.crop_length, args.dataset_type, True, train_weights
    )
    val_set = ESMDynamicDataset(
        args.data_dir, val_ids, args.crop_length, args.dataset_type, False, val_weights
    )
    train_count = args.train_samples_per_epoch or len(train_set)
    # generator=None intentionally uses the checkpointed global torch RNG.
    if distributed():
        train_weights_for_sampler = train_set.weights
        if train_weights_for_sampler is None:
            train_weights_for_sampler = torch.ones(len(train_set), dtype=torch.double)
        train_sampler = DistributedWeightedSampler(
            train_weights_for_sampler, train_count, args.seed
        )
    else:
        train_sampler = train_set.weighted_random_sampler(train_count)

    # Validation is fixed across epochs. Sampling weights must never randomize it.
    val_data: Any = val_set
    val_sampler = None
    if args.weighted_random_validation:
        val_set.random_crop = True
        val_count = args.val_samples_per_epoch or len(val_set)
        if distributed():
            val_weights_for_sampler = val_set.weights
            if val_weights_for_sampler is None:
                val_weights_for_sampler = torch.ones(len(val_set), dtype=torch.double)
            val_sampler = DistributedWeightedSampler(
                val_weights_for_sampler, val_count, args.seed + 1_000_000
            )
        else:
            val_sampler = val_set.weighted_random_sampler(val_count)
    elif args.val_samples_per_epoch and args.val_samples_per_epoch < len(val_set):
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(val_set), generator=generator)[: args.val_samples_per_epoch]
        val_data = Subset(val_set, order.tolist())
    if distributed() and val_sampler is None:
        val_sampler = DistributedSampler(
            val_data, num_replicas=world_size(), rank=rank(), shuffle=False, drop_last=False
        )

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_samples,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(train_set, sampler=train_sampler, **common)
    val_loader = DataLoader(val_data, sampler=val_sampler, shuffle=False, **common)
    main_print(
        f"Dataset: train={len(train_set)}, val={len(val_set)}, "
        f"global_samples/epoch={train_count}, world_size={world_size()}"
    )
    return train_loader, val_loader, train_sampler, val_sampler


def selected_prefixes(loss_heads: list[str]) -> list[str]:
    unknown = set(loss_heads) - LOSS_NAMES
    if unknown:
        raise ValueError(f"Unknown loss heads: {sorted(unknown)}")
    prefixes = sorted({name.split("_", 1)[0] for name in loss_heads})
    return prefixes


def initialize_model(args: argparse.Namespace, prefixes: list[str]) -> ESMDynamic:
    if args.dataset_type == "rcsb":
        if prefixes != ["dynamic"]:
            raise ValueError("RCSB stage supports only the dynamic head")
        definitions = [{
            "name": "dynamic",
            "task_type": "classification",
            "n_conditions": 1,
            "use_confidence_head": "dynamic_confidence" in args.loss_heads,
            "use_residual_head": False,
        }]
        model = ESMDynamic(head_definitions=definitions)
    else:
        model = ESMDynamic(heads_to_load=prefixes)
    model.set_chunk_size(args.chunk_size)
    model.esmfold.requires_grad_(False)
    return model


def _extract_head_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {
        key: value for key, value in payload.items()
        if isinstance(value, torch.Tensor) and key.startswith("heads.")
    }


def load_initial_weights(model: ESMDynamic, path: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = _extract_head_state(payload)
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    replicated: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        if value.shape == target[key].shape:
            compatible[key] = value
        elif (
            key.startswith("heads.dynamic.")
            and value.ndim >= 1
            and value.shape[0] == 1
            and target[key].shape[0] == 5
            and value.shape[1:] == target[key].shape[1:]
        ):
            compatible[key] = value.repeat(5, *([1] * (value.ndim - 1)))
            replicated.append(key)
        else:
            skipped.append(key)
    if not compatible:
        raise RuntimeError(f"No compatible head weights found in {path}")
    model.load_state_dict(compatible, strict=False)
    main_print(
        f"Initialized {len(compatible)} tensors from {path}; "
        f"replicated 1->5 conditions for {len(replicated)} tensors; skipped {len(skipped)}"
    )


def unwrap_model(model: torch.nn.Module) -> ESMDynamic:
    return model.module if isinstance(model, DistributedDataParallel) else model


def forward_heads(
    model: torch.nn.Module, sequences: list[str], num_recycles: int,
    active_loss_heads: list[str],
) -> dict[str, torch.Tensor]:
    """Run frozen ESMFold plus trainable heads, omitting PDB/native-contact work."""
    base_model = unwrap_model(model)
    aatype, mask, residx, _, _ = batch_encode_sequences(sequences)
    aatype, mask, residx = (
        x.to(base_model.device, non_blocking=True) for x in (aatype, mask, residx)
    )
    return_keys = set(active_loss_heads)
    if "dynamic_confidence" in return_keys:
        return_keys.add("dynamic_logits")
    if "kinetic_confidence" in return_keys:
        return_keys.add("kinetic_logits")
    if "frequency_residual_pred" in return_keys:
        return_keys.add("frequency_pred")
    return model(
        aa=aatype, mask=mask, residx=residx, num_recycles=num_recycles,
        compute_native_contacts=False, return_keys=return_keys,
    )


def valid_pair_mask(lengths: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    valid = positions[None, :] < lengths.to(device)[:, None]
    return valid[:, :, None] & valid[:, None, :]


def build_loss(
    output: dict[str, torch.Tensor], batch: dict, args: argparse.Namespace,
    kinetic_weights: torch.Tensor | None,
) -> torch.Tensor:
    lengths = batch["lengths"].to(args.device)
    terms: list[torch.Tensor] = []
    for name in getattr(args, "active_loss_heads", args.loss_heads):
        if name == "dynamic_logits":
            logits = output[name]
            target = batch["dynamic"].to(args.device)
            mask = valid_pair_mask(lengths, logits.shape[-1], logits.device)[:, None]
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            probability = torch.sigmoid(logits)
            pt = probability * target + (1 - probability) * (1 - target)
            alpha_t = args.alpha * target + (1 - args.alpha) * (1 - target)
            terms.append((alpha_t * (1 - pt).pow(args.gamma) * raw)[mask.expand_as(raw)].mean())
        elif name == "dynamic_confidence":
            logits = output["dynamic_logits"].detach()
            truth = batch["dynamic"].to(args.device)
            correct = ((torch.sigmoid(logits) > 0.5) == (truth > 0.5)).float()
            pair_mask = valid_pair_mask(lengths, logits.shape[-1], logits.device)[:, None]
            count = pair_mask.sum(dim=-1).clamp_min(1)
            target = (correct * pair_mask).sum(dim=-1) / count
            residue_mask = pair_mask.any(dim=-1).expand_as(output[name])
            terms.append(F.mse_loss(output[name][residue_mask], target[residue_mask]))
        elif name == "frequency_pred":
            pred = output[name]
            target = batch["frequency"].to(args.device)
            mask = valid_pair_mask(lengths, pred.shape[-1], pred.device)[:, None].expand_as(pred)
            terms.append(F.mse_loss(pred[mask], target[mask]))
        elif name == "frequency_residual_pred":
            pred = output[name]
            truth = batch["frequency"].to(args.device)
            residual_target = (truth - output["frequency_pred"].detach()).abs()
            mask = valid_pair_mask(lengths, pred.shape[-1], pred.device)[:, None].expand_as(pred)
            terms.append(F.mse_loss(pred[mask], residual_target[mask]))
        elif name == "kinetic_logits":
            logits = output[name]
            target = batch["kinetic"].to(args.device).long()
            mask2d = valid_pair_mask(lengths, logits.shape[-2], logits.device)
            losses = []
            for rate in range(logits.shape[2]):
                selected = mask2d[:, None].expand(-1, logits.shape[1], -1, -1)
                weight = None if kinetic_weights is None else kinetic_weights[rate].to(
                    device=logits.device, dtype=logits.dtype
                )
                losses.append(F.cross_entropy(logits[:, :, rate][selected], target[:, :, rate][selected], weight=weight))
            terms.append(torch.stack(losses).mean())
        elif name == "kinetic_confidence":
            logits = output["kinetic_logits"].detach()
            truth = batch["kinetic"].to(args.device).long()
            correct = (logits.argmax(dim=-1) == truth).float().mean(dim=2)
            pair_mask = valid_pair_mask(lengths, logits.shape[-2], logits.device)[:, None]
            target = (correct * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(1)
            residue_mask = pair_mask.any(dim=-1).expand_as(output[name])
            terms.append(F.mse_loss(output[name][residue_mask], target[residue_mask]))
    if not terms:
        raise RuntimeError("No losses were constructed")
    return torch.stack(terms).mean()


def binary_auroc(truth: np.ndarray, scores: np.ndarray) -> float:
    truth = truth.astype(bool, copy=False)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def protein_metrics(
    truth: np.ndarray, probability: np.ndarray,
    frequency_true: np.ndarray | None = None,
    frequency_pred: np.ndarray | None = None,
) -> dict[str, float]:
    indices = np.triu_indices(truth.shape[-1], k=1)
    y = truth[indices] > 0.5
    score = probability[indices]
    pred = score > 0.5
    tp = int(np.count_nonzero(pred & y))
    tn = int(np.count_nonzero(~pred & ~y))
    fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result = {
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auroc": binary_auroc(y, score),
        "rmse": float("nan"),
    }
    if frequency_true is not None and frequency_pred is not None:
        difference = frequency_pred[indices] - frequency_true[indices]
        result["rmse"] = float(np.sqrt(np.mean(np.square(difference))))
    return result


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name in ("balanced_accuracy", "precision", "recall", "f1", "auroc", "rmse"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[name] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "sem": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan"),
        }
    return summary


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_epoch(
    model: torch.nn.Module, loader: DataLoader, args: argparse.Namespace,
    kinetic_weights: torch.Tensor | None, optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, dict[str, float | int]], float]:
    training = optimizer is not None
    model.train(training)
    base_model = unwrap_model(model)
    if training:
        base_model.esmfold.eval()  # frozen trunk must not update dropout/batch statistics
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = 0
    metric_rows: list[dict[str, float]] = []
    amp_enabled = args.device.startswith("cuda") and not args.no_amp
    synchronize(args.device)
    started = time.perf_counter()

    grad_context = nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader):
            group_start = (batch_index // args.effective_batch_accum) * args.effective_batch_accum
            group_size = min(args.effective_batch_accum, len(loader) - group_start)
            should_step = batch_index + 1 == group_start + group_size
            sync_context = nullcontext()
            if training and isinstance(model, DistributedDataParallel) and not should_step:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    output = forward_heads(
                        model, batch["sequences"], args.num_recycles, args.active_loss_heads
                    )
                    loss = build_loss(output, batch, args, kinetic_weights)
                if training:
                    (loss / group_size).backward()
            if training:
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            batches += 1

            if not training and "dynamic_logits" in output:
                condition = args.metric_condition
                if condition >= output["dynamic_logits"].shape[1]:
                    raise IndexError(f"metric-condition {condition} is unavailable")
                probabilities = torch.sigmoid(output["dynamic_logits"][:, condition]).float().cpu()
                dynamic = batch["dynamic"][:, condition].float()
                predicted_frequency = output.get("frequency_pred")
                for i, length in enumerate(batch["lengths"].tolist()):
                    freq_true = freq_pred = None
                    if batch["frequency"] is not None and predicted_frequency is not None:
                        freq_true = batch["frequency"][i, condition, :length, :length].numpy()
                        freq_pred = predicted_frequency[i, condition, :length, :length].float().cpu().numpy()
                    metric_rows.append(protein_metrics(
                        dynamic[i, :length, :length].numpy(),
                        probabilities[i, :length, :length].numpy(), freq_true, freq_pred,
                    ))
            del output, loss

    synchronize(args.device)
    elapsed = time.perf_counter() - started
    if distributed():
        totals = torch.tensor([total_loss, batches, elapsed], device=args.device, dtype=torch.float64)
        dist.all_reduce(totals[:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(totals[2:], op=dist.ReduceOp.MAX)
        total_loss, batches, elapsed = totals.tolist()
        if not training:
            gathered: list[list[dict[str, float]] | None] = [None] * world_size()
            dist.all_gather_object(gathered, metric_rows)
            metric_rows = [row for rows in gathered if rows is not None for row in rows]
    return total_loss / max(batches, 1), summarize_metrics(metric_rows), elapsed


def head_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model = unwrap_model(model)
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items()
        if key.startswith("heads.")
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and state["cuda"]:
        saved = state["cuda"]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved)
        else:
            torch.cuda.set_rng_state(saved[min(rank(), len(saved) - 1)])


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, best_val_loss: float, args: argparse.Namespace,
    epochs_without_improvement: int,
) -> None:
    payload = {
        "format_version": 2,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state_dict": head_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "rng_state": rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    fields = list(row)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.loss_heads = list(args.loss_heads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed():
        if not args.device.startswith("cuda"):
            raise ValueError("DDP currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.device = f"cuda:{local_rank}"
        if args.batch_accum % world_size() != 0:
            raise ValueError(
                f"batch-accum={args.batch_accum} must be divisible by world_size={world_size()} "
                "to preserve the configured global effective batch size"
            )
        args.effective_batch_accum = args.batch_accum // world_size()
    else:
        args.effective_batch_accum = args.batch_accum
    if args.dataset_type == "rcsb" and "dynamic_logits" not in args.loss_heads:
        raise ValueError("RCSB pretraining requires dynamic_logits")
    if args.dataset_type == "mdcath" and "dynamic_logits" not in args.loss_heads:
        main_print("Warning: the six Table-1 metrics require dynamic_logits; only available metrics will be reported")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.outpath)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.deterministic)
    total_started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, train_sampler, val_sampler = make_loaders(args)
    prefixes = selected_prefixes(args.loss_heads)
    load_started = time.perf_counter()
    model = initialize_model(args, prefixes)
    if args.init_checkpoint and not args.resume:
        load_initial_weights(model, args.init_checkpoint)
    model.to(args.device)
    synchronize(args.device)
    model_load_seconds = time.perf_counter() - load_started
    optimizer = torch.optim.Adam(
        [parameter for head in model.heads.values() for parameter in head.parameters()],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    kinetic_weights = load_tensor(args.kin_class_weights)
    start_epoch, best_val_loss, epochs_without_improvement = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        # A world-size change cannot retain an identical RNG trajectory; give
        # each rank a deterministic independent stream after restoring state.
        if distributed():
            torch.manual_seed(args.seed + start_epoch * 100_003 + rank())
            torch.cuda.manual_seed(args.seed + start_epoch * 100_003 + rank())
        main_print(f"Resumed {args.resume} at epoch {start_epoch + 1}")

    if distributed():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        main_print(
            f"DDP enabled on {world_size()} GPUs; per-GPU batch={args.batch_size}, "
            f"accumulation={args.effective_batch_accum}, "
            f"global effective batch={args.batch_size * args.effective_batch_accum * world_size()}"
        )

    writer = SummaryWriter(output_dir / "tensorboard") if SummaryWriter and is_main_process() else None
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "model_load_seconds": model_load_seconds,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if is_main_process():
        metadata_name = "run_metadata.json" if not args.resume else f"run_metadata_resume_epoch_{start_epoch + 1}.json"
        (output_dir / metadata_name).write_text(json.dumps(metadata, indent=2, allow_nan=True))

    history_path = output_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)
        auxiliary = {"dynamic_confidence", "kinetic_confidence", "frequency_residual_pred"}
        if epoch + 1 < args.aux_head_start_epoch:
            args.active_loss_heads = [name for name in args.loss_heads if name not in auxiliary]
        else:
            args.active_loss_heads = list(args.loss_heads)
        main_print(f"Epoch {epoch + 1} active losses: {', '.join(args.active_loss_heads)}")
        train_loss, _, train_seconds = run_epoch(
            model, train_loader, args, kinetic_weights, optimizer
        )
        val_loss, metrics, val_seconds = run_epoch(
            model, val_loader, args, kinetic_weights, None
        )
        row: dict[str, Any] = {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "train_seconds": train_seconds, "val_seconds": val_seconds,
            "epoch_seconds": train_seconds + val_seconds,
        }
        for name, values in metrics.items():
            for statistic, value in values.items():
                row[f"val_{name}_{statistic}"] = value
        if args.device.startswith("cuda"):
            peak = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=args.device)
            if distributed():
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            row["peak_gpu_memory_mib"] = float(peak)
        if is_main_process():
            append_history(history_path, row)
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/validation", val_loss, epoch + 1)
            writer.add_scalar("time/train_seconds", train_seconds, epoch + 1)
            writer.add_scalar("time/validation_seconds", val_seconds, epoch + 1)
            for name, values in metrics.items():
                if math.isfinite(float(values["mean"])):
                    writer.add_scalar(f"metrics/{name}", values["mean"], epoch + 1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if is_main_process():
            save_checkpoint(
                output_dir / "checkpoint_last.pt", model, optimizer, epoch,
                best_val_loss, args, epochs_without_improvement,
            )
            if improved:
                save_checkpoint(
                    output_dir / "checkpoint_best.pt", model, optimizer, epoch,
                    best_val_loss, args, epochs_without_improvement,
                )
                torch.save(head_state_dict(model), output_dir / "heads_best.pt")
        if distributed():
            dist.barrier()
        means = " ".join(
            f"{name}={values['mean']:.4f}" for name, values in metrics.items()
            if math.isfinite(float(values["mean"]))
        )
        main_print(
            f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6g} "
            f"val_loss={val_loss:.6g} train={train_seconds:.1f}s val={val_seconds:.1f}s {means}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            main_print(
                f"Early stopping: validation loss did not improve for "
                f"{epochs_without_improvement} epochs"
            )
            break

    total_seconds = time.perf_counter() - total_started
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_load_seconds": model_load_seconds,
        "total_process_seconds": total_seconds,
        "best_val_loss": best_val_loss,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else None
        ),
    }
    if is_main_process():
        (output_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2))
        torch.save(head_state_dict(model), output_dir / "heads_last.pt")
    if writer:
        writer.close()
    main_print(f"Training complete in {total_seconds:.1f}s; best validation loss={best_val_loss:.6g}")
    if distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
