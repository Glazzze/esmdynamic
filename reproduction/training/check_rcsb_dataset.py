#!/usr/bin/env python3
"""Exhaustively validate the released RCSB train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Result:
    split: str
    index: int
    identifier: str
    sequence_length: int | None
    expected_labels: int | None
    actual_labels: int | None
    status: str
    detail: str


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    identifiers: list[str] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[0].strip()
            if value and value.lower() not in {"id", "name", "identifier"}:
                identifiers.append(value)
    return identifiers


def parse_fasta(path: Path) -> tuple[str, int]:
    records: list[list[str]] = []
    current: list[str] | None = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = []
                records.append(current)
            elif current is None:
                raise ValueError("sequence appears before FASTA header")
            else:
                current.append(line)
    if not records or not records[0]:
        raise ValueError("empty FASTA")
    sequence = "".join(records[0]).upper()
    invalid = sorted(set(sequence) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if invalid:
        raise ValueError(f"invalid sequence characters: {invalid}")
    return sequence, len(records)


def validate_two_byte_labels(path: Path, expected: int) -> tuple[int, str | None]:
    """Validate the release format: one ASCII 0/1 plus newline per label."""
    size = path.stat().st_size
    if size not in {2 * expected, max(0, 2 * expected - 1)}:
        # Fully parse anomalous-size files so the report has an exact count and
        # distinguishes a length mismatch from malformed content.
        count = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip() not in {b"0", b"1"}:
                    return count, f"invalid label at line {line_number}: {line[:40]!r}"
                count += 1
        return count, None

    count = 0
    carry = b""
    with path.open("rb", buffering=1024 * 1024) as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            data = carry + chunk
            usable = len(data) - len(data) % 2
            view = np.frombuffer(data[:usable], dtype=np.uint8).reshape(-1, 2)
            valid_values = (view[:, 0] == ord("0")) | (view[:, 0] == ord("1"))
            valid_newlines = view[:, 1] == ord("\n")
            # A missing final newline leaves one byte as carry and is handled below.
            if not bool(np.all(valid_values & valid_newlines)):
                bad = int(np.flatnonzero(~(valid_values & valid_newlines))[0])
                return count + bad, "labels are not one ASCII 0/1 value per line"
            count += view.shape[0]
            carry = data[usable:]
    if carry:
        if carry not in {b"0", b"1"}:
            return count, f"invalid final label byte: {carry!r}"
        count += 1
    return count, None


def check_sample(data_dir: Path, split: str, index: int, identifier: str) -> Result:
    sample_dir = data_dir / identifier
    fasta = sample_dir / "consensus.fasta"
    labels = sample_dir / "dynamic_contacts.csv"
    base = dict(split=split, index=index, identifier=identifier)
    if not sample_dir.is_dir():
        return Result(**base, sequence_length=None, expected_labels=None,
                      actual_labels=None, status="missing_directory", detail=str(sample_dir))
    if not fasta.is_file():
        return Result(**base, sequence_length=None, expected_labels=None,
                      actual_labels=None, status="missing_fasta", detail=str(fasta))
    if not labels.is_file():
        return Result(**base, sequence_length=None, expected_labels=None,
                      actual_labels=None, status="missing_labels", detail=str(labels))
    try:
        sequence, records = parse_fasta(fasta)
    except Exception as error:
        return Result(**base, sequence_length=None, expected_labels=None,
                      actual_labels=None, status="invalid_fasta", detail=str(error))
    length = len(sequence)
    expected = length * (length - 1) // 2
    try:
        actual, content_error = validate_two_byte_labels(labels, expected)
    except Exception as error:
        return Result(**base, sequence_length=length, expected_labels=expected,
                      actual_labels=None, status="label_read_error", detail=str(error))
    if content_error:
        return Result(**base, sequence_length=length, expected_labels=expected,
                      actual_labels=actual, status="invalid_label_content", detail=content_error)
    if records != 1:
        return Result(**base, sequence_length=length, expected_labels=expected,
                      actual_labels=actual, status="multiple_fasta_records",
                      detail=f"found {records} FASTA records")
    if actual != expected:
        return Result(**base, sequence_length=length, expected_labels=expected,
                      actual_labels=actual, status="label_count_mismatch",
                      detail=f"expected {expected}, found {actual}")
    return Result(**base, sequence_length=length, expected_labels=expected,
                  actual_labels=actual, status="ok", detail="")


def weight_report(split_dir: Path, split: str, expected: int) -> dict:
    path = split_dir / f"{split}_weights.pt"
    report = {"path": str(path), "exists": path.is_file(), "expected": expected}
    if not path.is_file():
        return report
    weights = torch.load(path, map_location="cpu", weights_only=True)
    array = torch.as_tensor(weights)
    report.update({
        "count": int(array.numel()),
        "shape": list(array.shape),
        "finite": bool(torch.isfinite(array).all()),
        "nonnegative": bool((array >= 0).all()),
        "positive_sum": bool(array.sum() > 0),
        "count_matches": int(array.numel()) == expected,
    })
    return report


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {name: read_ids(args.split_dir / f"{name}.csv")
                 for name in ("train", "val", "test")}
    tasks = [(name, index, identifier)
             for name, identifiers in split_ids.items()
             for index, identifier in enumerate(identifiers, 1)]
    started = time.perf_counter()
    issues: list[Result] = []
    status_counts: dict[str, int] = {}
    completed = 0
    print(f"Checking {len(tasks)} split entries with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_sample, args.data_dir, split, index, identifier):
            (split, index, identifier)
            for split, index, identifier in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            if result.status != "ok":
                issues.append(result)
                print(
                    f"ISSUE {result.split}:{result.index} {result.identifier} "
                    f"{result.status}: {result.detail}", flush=True
                )
            if completed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                remaining = (len(tasks) - completed) / rate if rate else math.inf
                print(
                    f"Progress {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%), "
                    f"{rate:.1f} samples/s, ETA {remaining/60:.1f} min", flush=True
                )

    issues.sort(key=lambda item: (item.split, item.index))
    issue_csv = args.output_dir / "issues.csv"
    with issue_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(issues[0]).keys()) if issues else
                                list(Result.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(item) for item in issues)

    locations: dict[str, list[str]] = {}
    for split, identifiers in split_ids.items():
        for identifier in identifiers:
            locations.setdefault(identifier, []).append(split)
    duplicates_within = {
        split: len(ids) - len(set(ids)) for split, ids in split_ids.items()
    }
    overlaps = {identifier: names for identifier, names in locations.items() if len(names) > 1}
    weights = {name: weight_report(args.split_dir, name, len(ids))
               for name, ids in split_ids.items()}
    elapsed = time.perf_counter() - started
    summary = {
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir),
        "total_entries": len(tasks),
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "status_counts": status_counts,
        "issue_count": len(issues),
        "duplicates_within_split": duplicates_within,
        "cross_split_overlap_count": len(overlaps),
        "cross_split_overlaps": overlaps,
        "weight_files": weights,
        "elapsed_seconds": elapsed,
        "issues_csv": str(issue_csv),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
