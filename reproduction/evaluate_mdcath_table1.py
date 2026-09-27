#!/usr/bin/env python3
"""Reproduce the ESMDynamic column of Table 1 on mdCATH at 320 K."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.stats
import torch

from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmfold.v1.misc import batch_encode_sequences


PAPER_VALUES = {
    "balanced_accuracy": (0.796, 0.007),
    "precision": (0.511, 0.012),
    "recall": (0.767, 0.010),
    "f1": (0.569, 0.008),
    "auroc": (0.889, 0.006),
    "rmse": (0.076, 0.002),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--num-recycles", type=int, default=3)
    p.add_argument("--max-initial-gpu-used-mib", type=int, default=512)
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def gpu_snapshot(physical_gpu: int) -> dict[str, int | str]:
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={physical_gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [x.strip() for x in out.split(",")]
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_total_mib": int(values[2]),
        "memory_used_mib": int(values[3]),
        "memory_free_mib": int(values[4]),
        "utilization_percent": int(values[5]),
    }


def read_ids(path: Path) -> list[str]:
    ids = []
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip() and row[0].strip().lower() not in {"id", "name", "identifier"}:
                ids.append(row[0].strip())
    return ids


def load_sequence(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "".join(line.strip() for line in lines if line and not line.startswith(">"))


def auroc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.uint8, copy=False)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy.stats.rankdata(scores, method="average")
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y_true: np.ndarray, probability: np.ndarray, occupancy_true: np.ndarray,
            occupancy_pred: np.ndarray, upper_only: bool) -> dict[str, float]:
    if upper_only:
        idx = np.triu_indices(y_true.shape[0], k=1)
        y = y_true[idx]
        prob = probability[idx]
        freq_true = occupancy_true[idx]
        freq_pred = occupancy_pred[idx]
    else:
        y = y_true.reshape(-1)
        prob = probability.reshape(-1)
        freq_true = occupancy_true.reshape(-1)
        freq_pred = occupancy_pred.reshape(-1)

    pred = prob > 0.5
    truth = y > 0.5
    tp = int(np.count_nonzero(pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc_binary(truth, prob),
        "rmse": float(np.sqrt(np.mean(np.square(freq_pred - freq_true)))),
    }


def predict_320k(model: ESMDynamic, sequence: str, num_recycles: int) -> tuple[np.ndarray, np.ndarray]:
    aa, mask, residx, _, _ = batch_encode_sequences([sequence])
    aa, mask, residx = (x.cuda(non_blocking=True) for x in (aa, mask, residx))
    with torch.inference_mode():
        base = model.esmfold(aa, mask, residx, None, num_recycles)
        base["mask"] = mask

        dynamic_state = dict(base)
        model.heads["dynamic"](dynamic_state, num_recycles=num_recycles)
        dynamic_prob = dynamic_state["dynamic_prob"][0, 0].float().cpu().numpy()
        del dynamic_state

        frequency_state = dict(base)
        model.heads["frequency"](frequency_state, num_recycles=num_recycles)
        frequency_pred = frequency_state["frequency_pred"][0, 0].float().cpu().numpy()
        del frequency_state, base, aa, mask, residx

    return dynamic_prob, frequency_pred


def summarize(rows: list[dict], suffix: str) -> dict[str, dict[str, float]]:
    result = {}
    for metric_name in PAPER_VALUES:
        values = np.asarray([float(r[f"{metric_name}_{suffix}"]) for r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[metric_name] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "sem": float(values.std(ddof=1) / math.sqrt(values.size)),
            "paper_mean": PAPER_VALUES[metric_name][0],
            "paper_sem": PAPER_VALUES[metric_name][1],
            "mean_difference": float(values.mean() - PAPER_VALUES[metric_name][0]),
        }
    return result


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise SystemExit(
            f"Set CUDA_VISIBLE_DEVICES={args.physical_gpu}; got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    initial_gpu = gpu_snapshot(args.physical_gpu)
    if not args.allow_busy_gpu and initial_gpu["memory_used_mib"] > args.max_initial_gpu_used_mib:
        raise SystemExit(
            f"GPU {args.physical_gpu} is not exclusive: {initial_gpu['memory_used_mib']} MiB already used "
            f"(limit {args.max_initial_gpu_used_mib} MiB)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_protein_csv = args.output_dir / "per_protein_metrics.csv"
    summary_json = args.output_dir / "summary.json"
    metadata_json = args.output_dir / "run_metadata.json"
    ids = read_ids(args.test_csv)
    if len(ids) != 270 or len(set(ids)) != 270:
        raise ValueError(f"Expected 270 unique test IDs, found {len(ids)} rows and {len(set(ids))} unique IDs")
    missing = [protein_id for protein_id in ids if not (args.data_dir / protein_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test samples: {missing[:10]}")

    completed: dict[str, dict] = {}
    if args.resume and per_protein_csv.exists():
        with per_protein_csv.open(newline="") as fh:
            completed = {row["protein_id"]: row for row in csv.DictReader(fh)}
    elif per_protein_csv.exists():
        raise FileExistsError(f"Output exists: {per_protein_csv}; use --resume or a new output directory")

    start_iso = datetime.now(timezone.utc).isoformat()
    total_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model = ESMDynamic(heads_to_load=["dynamic", "frequency"])
    # This project checkpoint is generated locally and contains NumPy metadata
    # in addition to the state dict; explicitly use the legacy loader for it.
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    # Training checkpoints wrap the actual parameters together with optimizer
    # and progress metadata; evaluation only needs the model state dict.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    load_result = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in load_result.missing_keys if not k.startswith("esmfold.") and k != "dummy_buffer"]
    bad_unexpected = [k for k in load_result.unexpected_keys if not k.startswith("heads.kinetic.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Weight mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.set_chunk_size(args.chunk_size)
    model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    model_load_peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    overall_peak_mib = model_load_peak_mib

    fieldnames = ["protein_id", "length", "inference_seconds", "peak_gpu_memory_mib"]
    for suffix in ("full", "upper"):
        fieldnames.extend(f"{name}_{suffix}" for name in PAPER_VALUES)

    mode = "a" if completed else "w"
    with per_protein_csv.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()
            fh.flush()

        inference_start = time.perf_counter()
        for index, protein_id in enumerate(ids, 1):
            if protein_id in completed:
                continue
            sample_dir = args.data_dir / protein_id
            sequence = load_sequence(sample_dir / "consensus.fasta")
            y_true = torch.load(sample_dir / "dynamic_contacts.pt", map_location="cpu", weights_only=True)[0].numpy()
            occupancy_true = torch.load(sample_dir / "frequency.pt", map_location="cpu", weights_only=True)[0].numpy()
            if y_true.shape != (len(sequence), len(sequence)) or occupancy_true.shape != y_true.shape:
                raise ValueError(f"Shape mismatch for {protein_id}")

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            sample_start = time.perf_counter()
            probability, occupancy_pred = predict_320k(model, sequence, args.num_recycles)
            torch.cuda.synchronize()
            sample_seconds = time.perf_counter() - sample_start
            sample_peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            overall_peak_mib = max(overall_peak_mib, sample_peak_mib)
            if probability.shape != y_true.shape or occupancy_pred.shape != y_true.shape:
                raise ValueError(f"Prediction shape mismatch for {protein_id}")

            row = {
                "protein_id": protein_id,
                "length": len(sequence),
                "inference_seconds": sample_seconds,
                "peak_gpu_memory_mib": sample_peak_mib,
            }
            for suffix, upper_only in (("full", False), ("upper", True)):
                values = metrics(y_true, probability, occupancy_true, occupancy_pred, upper_only)
                row.update({f"{name}_{suffix}": value for name, value in values.items()})
            writer.writerow(row)
            fh.flush()
            print(
                f"[{index:03d}/{len(ids)}] {protein_id} L={len(sequence)} "
                f"time={sample_seconds:.2f}s peak={row['peak_gpu_memory_mib']:.0f}MiB",
                flush=True,
            )
        torch.cuda.synchronize()
        inference_seconds_this_process = time.perf_counter() - inference_start

    with per_protein_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(ids):
        raise RuntimeError(f"Evaluation incomplete: {len(rows)}/{len(ids)} rows")

    summaries = {"full_matrix": summarize(rows, "full"), "upper_triangle_no_diagonal": summarize(rows, "upper")}
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n")
    torch.cuda.synchronize()
    total_seconds_this_process = time.perf_counter() - total_start
    per_sample_times = np.asarray([float(r["inference_seconds"]) for r in rows])
    metadata = {
        "start_utc": start_iso,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": initial_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "weights": str(args.weights.resolve()),
        "weights_size_bytes": args.weights.stat().st_size,
        "data_dir": str(args.data_dir.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "n_test_proteins": len(ids),
        "temperature_kelvin": 320,
        "temperature_index": 0,
        "classification_threshold": 0.5,
        "batch_size": 1,
        "chunk_size": args.chunk_size,
        "num_recycles": args.num_recycles,
        "model_load_seconds": model_load_seconds,
        "model_load_peak_gpu_memory_mib": model_load_peak_mib,
        "inference_seconds_sum": float(per_sample_times.sum()),
        "inference_seconds_mean_per_protein": float(per_sample_times.mean()),
        "inference_seconds_this_process": inference_seconds_this_process,
        "total_wall_seconds_this_process": total_seconds_this_process,
        "peak_gpu_memory_allocated_mib": overall_peak_mib,
        "resumed": bool(completed),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "timing": metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
