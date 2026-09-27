#!/usr/bin/env python3
"""Remove integrity-check failures from RCSB splits and aligned weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def read_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--issues-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bad_by_split: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    with args.issues_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            bad_by_split[row["split"]].add(row["identifier"])

    report = {"source_dir": str(args.source_dir), "issues_csv": str(args.issues_csv), "splits": {}}
    for split in ("train", "val", "test"):
        ids = read_ids(args.source_dir / f"{split}.csv")
        weights = torch.load(
            args.source_dir / f"{split}_weights.pt", map_location="cpu", weights_only=True
        )
        weights = torch.as_tensor(weights)
        if weights.numel() != len(ids):
            raise ValueError(f"{split}: {weights.numel()} weights for {len(ids)} IDs")

        bad = bad_by_split[split]
        source_positions = {identifier: index for index, identifier in enumerate(ids)}
        unknown = sorted(bad - source_positions.keys())
        if unknown:
            raise ValueError(f"{split}: issue IDs absent from source split: {unknown[:10]}")
        keep = torch.tensor([identifier not in bad for identifier in ids], dtype=torch.bool)
        filtered_ids = [identifier for identifier in ids if identifier not in bad]
        filtered_weights = weights[keep]

        with (args.output_dir / f"{split}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows([identifier] for identifier in filtered_ids)
        torch.save(filtered_weights, args.output_dir / f"{split}_weights.pt")
        with (args.output_dir / f"{split}_excluded.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_index_1based", "identifier"])
            writer.writerows(
                [source_positions[identifier] + 1, identifier] for identifier in sorted(bad)
            )

        report["splits"][split] = {
            "source_count": len(ids),
            "excluded_count": len(bad),
            "filtered_count": len(filtered_ids),
            "source_weight_count": int(weights.numel()),
            "filtered_weight_count": int(filtered_weights.numel()),
            "weight_dtype": str(filtered_weights.dtype),
            "weight_sum": float(filtered_weights.sum()),
        }

    report["total_excluded"] = sum(len(values) for values in bad_by_split.values())
    (args.output_dir / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
