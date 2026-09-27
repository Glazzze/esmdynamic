#!/usr/bin/env python3
"""CPU-only validation of mdCATH data and stage-1 checkpoint transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmdynamic.training.data_reader import ESMDynamicDataset, read_identifiers
from esm.esmdynamic.training.train import load_initial_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--kin-class-weights", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--crop-length", type=int, default=64)
    return parser.parse_args()


def validate_samples(args: argparse.Namespace) -> None:
    expected = {"dynamic": 5, "frequency": 5, "kinetic": 5}
    for name, split in (("train", args.train_split), ("val", args.val_split)):
        identifiers = read_identifiers(split)[: args.samples_per_split]
        dataset = ESMDynamicDataset(
            args.data_dir,
            identifiers,
            crop_length=args.crop_length,
            dataset_type="mdcath",
            random_crop=False,
        )
        for index, identifier in enumerate(identifiers):
            sample = dataset[index]
            for target, conditions in expected.items():
                value = sample[target]
                if value is None or value.shape[0] != conditions:
                    raise ValueError(
                        f"{name}/{identifier}: invalid {target} shape "
                        f"{None if value is None else tuple(value.shape)}"
                    )
            if sample["kinetic"].shape[1] != 2:
                raise ValueError(f"{name}/{identifier}: kinetics must contain on/off rates")
            print(
                f"{name}/{identifier}: length={sample['length']} "
                f"dynamic={tuple(sample['dynamic'].shape)} "
                f"frequency={tuple(sample['frequency'].shape)} "
                f"kinetic={tuple(sample['kinetic'].shape)}"
            )


def validate_checkpoint(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", -1)) + 1
    print(f"checkpoint={args.checkpoint}; completed_epoch={epoch}")

    model = ESMDynamic(
        load_esmfold=False,
        heads_to_load=["dynamic", "frequency", "kinetic"],
    )
    load_initial_weights(model, str(args.checkpoint))
    dynamic = model.heads["dynamic"]
    weight = dynamic.prediction_linear.weight.detach()
    bias = dynamic.prediction_linear.bias.detach()
    if weight.shape[0] != 5 or not all(torch.equal(weight[0], weight[i]) for i in range(1, 5)):
        raise ValueError("dynamic prediction weights were not replicated to five conditions")
    if not all(torch.equal(bias[0], bias[i]) for i in range(1, 5)):
        raise ValueError("dynamic prediction biases were not replicated to five conditions")

    batch, length = 1, 4
    base = {
        "lddt_head": torch.zeros(1, batch, length, 37, 50),
        "lm_logits": torch.zeros(batch, length, 23),
        "s_s": torch.zeros(batch, length, 1024),
        "ptm_logits": torch.zeros(batch, length, length, 64),
        "distogram_logits": torch.zeros(batch, length, length, 64),
        "s_z": torch.zeros(batch, length, length, 128),
        "residue_index": torch.arange(length).view(1, length),
        "mask": torch.ones(batch, length),
    }
    with torch.no_grad():
        for name, head in model.heads.items():
            state = {key: value.clone() for key, value in base.items()}
            head(state, num_recycles=0)
            shapes = {
                key: tuple(value.shape)
                for key, value in state.items()
                if key.startswith(name + "_") and isinstance(value, torch.Tensor)
            }
            print(f"{name} outputs: {shapes}")


def main() -> None:
    args = parse_args()
    kinetic_weights = torch.load(
        args.kin_class_weights, map_location="cpu", weights_only=True
    )
    if tuple(kinetic_weights.shape) != (2, 6):
        raise ValueError(
            f"kinetic class weights must have shape (2, 6), got {tuple(kinetic_weights.shape)}"
        )
    validate_samples(args)
    validate_checkpoint(args)
    print("Stage-2 CPU validation passed")


if __name__ == "__main__":
    main()
