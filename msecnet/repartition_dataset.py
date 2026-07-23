#!/usr/bin/env python3
"""Create a balanced, group-disjoint split for an existing prepared dataset."""
import argparse
import json
from pathlib import Path

import numpy as np

from split_utils import build_group_split


def main():
    parser = argparse.ArgumentParser(description="Repartition a prepared MSECNet dataset by generalization group")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON; defaults to split_by_generalization_group.json in DATASET_DIR")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    labels_path = dataset_dir / "labels_manual3d.npz"
    if not manifest_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("DATASET_DIR must contain manifest.jsonl and labels_manual3d.npz")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    manifest_files = {str(row["file"]) for row in rows}
    label_files = {str(name) for name in np.load(labels_path)["files"]}
    if manifest_files != label_files:
        missing_labels = len(manifest_files - label_files)
        missing_manifest = len(label_files - manifest_files)
        raise ValueError(f"manifest/labels mismatch: labels_missing={missing_labels} manifest_missing={missing_manifest}")
    split, _, metadata = build_group_split(rows, args.seed, args.val_fraction, args.test_fraction)
    output = args.out or (dataset_dir / "split_by_generalization_group.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = {**split, **metadata}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("split " + " ".join(f"{name}={metadata['counts'][name]}" for name in ("train", "val", "test")))
    print(f"generalization groups={len(metadata['groups'])}")


if __name__ == "__main__":
    main()
