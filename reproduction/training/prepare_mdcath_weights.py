#!/usr/bin/env python3
"""Create local length-proportional sampling weights for mdCATH splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from esm.esmdynamic.training.data_reader import read_fasta, read_identifiers


REQUIRED_FILES = (
    "consensus.fasta",
    "dynamic_contacts.pt",
    "frequency.pt",
    "kinetics.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_save(tensor: torch.Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def prepare_split(name: str, split_path: Path, data_dir: Path, output_dir: Path) -> None:
    identifiers = read_identifiers(split_path)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} split contains duplicate identifiers")

    lengths: list[int] = []
    missing: list[str] = []
    for index, identifier in enumerate(identifiers, 1):
        sample_dir = data_dir / identifier
        absent = [filename for filename in REQUIRED_FILES if not (sample_dir / filename).is_file()]
        if absent:
            missing.append(f"{identifier}: {', '.join(absent)}")
            continue
        lengths.append(len(read_fasta(sample_dir / "consensus.fasta")))
        if index % 500 == 0:
            print(f"{name}: checked {index}/{len(identifiers)}", flush=True)

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{name} has {len(missing)} incomplete samples; first entries:\n{preview}"
        )
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{name} contains an empty FASTA sequence")

    weights = torch.tensor(lengths, dtype=torch.double)
    destination = output_dir / f"{name}_weights.pt"
    atomic_save(weights, destination)
    print(
        f"{name}: wrote {len(weights)} weights to {destination}; "
        f"length range={int(weights.min())}-{int(weights.max())}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_split("train", args.train_split, args.data_dir, args.output_dir)
    prepare_split("val", args.val_split, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
