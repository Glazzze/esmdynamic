"""Datasets used by the two-stage ESMDynamic training procedure."""

from __future__ import annotations

import csv
import mmap
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def read_identifiers(path: str | Path) -> list[str]:
    """Read the first column of a CSV/text split, tolerating a header."""
    identifiers: list[str] = []
    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[0].strip()
            if value and value.lower() not in {"id", "name", "identifier"}:
                identifiers.append(value)
    if not identifiers:
        raise ValueError(f"No identifiers found in {path}")
    return identifiers


def read_fasta(path: str | Path) -> str:
    """Read a FASTA sequence, including wrapped/multiline FASTA files."""
    with Path(path).open() as handle:
        sequence = "".join(
            line.strip() for line in handle if line.strip() and not line.startswith(">")
        )
    if not sequence:
        raise ValueError(f"Empty FASTA sequence: {path}")
    return sequence


def restore_strict_upper_triangle(values: torch.Tensor, length: int) -> torch.Tensor:
    """Restore a symmetric LxL matrix from L*(L-1)/2 strict-upper values."""
    values = values.reshape(-1)
    expected = length * (length - 1) // 2
    if values.numel() != expected:
        raise ValueError(
            f"Expected {expected} strict-upper values for length {length}, "
            f"found {values.numel()}"
        )
    matrix = torch.zeros((length, length), dtype=values.dtype)
    indices = torch.triu_indices(length, length, offset=1)
    matrix[indices[0], indices[1]] = values
    matrix[indices[1], indices[0]] = values
    return matrix


def crop_strict_upper_triangle(
    values: torch.Tensor, length: int, start: int, end: int
) -> torch.Tensor:
    """Restore only a contiguous crop from strict-upper row-major values."""
    values = values.reshape(-1)
    expected = length * (length - 1) // 2
    if values.numel() != expected:
        raise ValueError(
            f"Expected {expected} strict-upper values for length {length}, "
            f"found {values.numel()}"
        )
    crop_length = end - start
    matrix = torch.zeros((crop_length, crop_length), dtype=values.dtype)
    for source_i in range(start, end):
        destination_i = source_i - start
        source_offset = source_i * (2 * length - source_i - 1) // 2
        first_source_j = source_offset + max(start, source_i + 1) - source_i - 1
        destination_j = max(start, source_i + 1) - start
        count = end - max(start, source_i + 1)
        if count > 0:
            row = values[first_source_j:first_source_j + count]
            matrix[destination_i, destination_j:destination_j + count] = row
            matrix[destination_j:destination_j + count, destination_i] = row
    return matrix


def _torch_load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


class ESMDynamicDataset(Dataset):
    """Task-aware reader for either RCSB pretraining or mdCATH fine-tuning."""

    def __init__(
        self,
        data_dir: str | Path,
        identifiers: Iterable[str],
        crop_length: int = 256,
        dataset_type: str = "mdcath",
        random_crop: bool = True,
        weights: torch.Tensor | None = None,
    ) -> None:
        if dataset_type not in {"rcsb", "mdcath"}:
            raise ValueError("dataset_type must be 'rcsb' or 'mdcath'")
        self.data_dir = Path(data_dir)
        self.identifiers = list(identifiers)
        self.crop_length = crop_length
        self.dataset_type = dataset_type
        self.random_crop = random_crop
        self.weights = weights
        if weights is not None and len(weights) != len(self.identifiers):
            raise ValueError(
                f"Sampling weights ({len(weights)}) do not match identifiers "
                f"({len(self.identifiers)})"
            )

    def __len__(self) -> int:
        return len(self.identifiers)

    def _load_rcsb_dynamic(
        self, sample_dir: Path, length: int, start: int, end: int
    ) -> torch.Tensor:
        tensor_path = sample_dir / "dynamic_contacts.pt"
        if tensor_path.exists():
            values = _torch_load(tensor_path)
        else:
            csv_path = sample_dir / "dynamic_contacts.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing RCSB labels in {sample_dir}")
            expected = length * (length - 1) // 2
            # Read only the requested crop directly from the newline-delimited
            # upper triangle. The integrity audit has already checked the full
            # file count for filtered splits.
            with csv_path.open("rb") as handle:
                mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                crop_length = end - start
                matrix = torch.zeros((crop_length, crop_length), dtype=torch.float32)
                for source_i in range(start, end):
                    first_j = max(start, source_i + 1)
                    count = end - first_j
                    if count <= 0:
                        continue
                    row_offset = source_i * (2 * length - source_i - 1) // 2
                    label_offset = 2 * (row_offset + first_j - source_i - 1)
                    raw_row = mapped[label_offset:label_offset + 2 * count:2]
                    row = torch.frombuffer(bytearray(raw_row), dtype=torch.uint8).float() - ord("0")
                    i = source_i - start
                    j = first_j - start
                    matrix[i, j:j + count] = row
                    matrix[j:j + count, i] = row
                mapped.close()
            return matrix.unsqueeze(0)

        values = values.float()
        if values.ndim == 2 and tuple(values.shape) == (length, length):
            matrix = values[start:end, start:end]
        elif values.ndim == 3 and tuple(values.shape) == (1, length, length):
            return values[:, start:end, start:end]
        else:
            try:
                matrix = crop_strict_upper_triangle(values, length, start, end)
            except ValueError as error:
                raise ValueError(f"Invalid RCSB sample {sample_dir.name!r} in {sample_dir}: {error}") from error
        return matrix.unsqueeze(0)

    def __getitem__(self, index: int) -> dict:
        identifier = self.identifiers[index]
        sample_dir = self.data_dir / identifier
        sequence = read_fasta(sample_dir / "consensus.fasta")
        length = len(sequence)

        if self.crop_length <= 0 or self.crop_length >= length:
            start, end = 0, length
        elif self.random_crop:
            # The final valid crop starts at length-crop_length, hence +1.
            start = int(torch.randint(0, length - self.crop_length + 1, ()).item())
            end = start + self.crop_length
        else:
            start, end = 0, self.crop_length

        if self.dataset_type == "rcsb":
            dynamic = self._load_rcsb_dynamic(sample_dir, length, start, end)
            kinetics = frequency = None
        else:
            dynamic = _torch_load(sample_dir / "dynamic_contacts.pt").float()
            kinetics = _torch_load(sample_dir / "kinetics.pt").long()
            frequency = _torch_load(sample_dir / "frequency.pt").float()

        result = {
            "identifier": identifier,
            "sequence": sequence[start:end],
            "length": end - start,
            "dynamic": dynamic if self.dataset_type == "rcsb" else dynamic[:, start:end, start:end],
            "kinetic": None,
            "frequency": None,
        }
        if kinetics is not None:
            result["kinetic"] = kinetics[:, :, start:end, start:end]
        if frequency is not None:
            result["frequency"] = frequency[:, start:end, start:end]
        return result

    def weighted_random_sampler(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> WeightedRandomSampler:
        weights = self.weights
        if weights is None:
            # The paper samples proteins with probability proportional to
            # sequence length when a split-specific weight file is absent.
            lengths = [len(read_fasta(self.data_dir / identifier / "consensus.fasta"))
                       for identifier in self.identifiers]
            weights = torch.as_tensor(lengths, dtype=torch.double)
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )


def collate_samples(samples: list[dict]) -> dict:
    """Pad optional task targets and retain IDs/sequences for per-protein metrics."""
    max_length = max(sample["length"] for sample in samples)

    def pad_pairwise(key: str) -> torch.Tensor | None:
        available = [sample[key] for sample in samples]
        if all(value is None for value in available):
            return None
        if any(value is None for value in available):
            raise ValueError(f"Only part of a batch has target {key!r}")
        first = available[0]
        assert first is not None
        prefix = first.shape[:-2]
        output = torch.zeros(
            (len(samples), *prefix, max_length, max_length), dtype=first.dtype
        )
        for i, value in enumerate(available):
            assert value is not None
            length = value.shape[-1]
            if value.shape[:-2] != prefix:
                raise ValueError(f"Inconsistent shape for {key}: {value.shape} vs {first.shape}")
            output[i, ..., :length, :length] = value
        return output

    return {
        "identifiers": [sample["identifier"] for sample in samples],
        "sequences": [sample["sequence"] for sample in samples],
        "lengths": torch.tensor([sample["length"] for sample in samples], dtype=torch.long),
        "dynamic": pad_pairwise("dynamic"),
        "kinetic": pad_pairwise("kinetic"),
        "frequency": pad_pairwise("frequency"),
    }


# Backwards-compatible name used by external scripts.
DynContactDataset = ESMDynamicDataset
