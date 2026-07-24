"""Leakage-resistant split helpers for the manual pseudo-OBB dataset."""
from collections import defaultdict
from pathlib import Path
import re

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def _canonical_model(value):
    """Normalize superficial spelling differences while preserving model identity."""
    return re.sub(r"[\s_\-]+", "", value).casefold()


def generalization_group(row):
    """Return one non-overlapping group for a sample and a human-readable label.

    The source index has several placeholder ``car_model`` values which span
    multiple real cars. Those real model names are recovered from filenames
    when present; otherwise a capture-session group prevents near-duplicate
    frames from crossing train/validation/test boundaries.
    """
    model = str(row["car_model"])
    dataset = str(row.get("dataset", ""))
    source = Path(str(row.get("source_cloud", row["file"]))).stem

    # testdept buckets encode the actual model between the bucket label and
    # ``_正常数据``. Join it to an identical directly-labelled model if present.
    encoded_model = re.search(r"车型\d+至车型\d+_(.+?)_正常数据", source)
    if encoded_model:
        label = encoded_model.group(1)
        return f"model:{_canonical_model(label)}", f"model:{label}"

    if model == "_未分类_易车网":
        match = re.search(r"易车网_data\d+_(.+)_\d+$", source)
        if match:
            label = match.group(1)
            return f"easycar_variant:{label}", f"easycar_variant:{label}"

    if model == "prod_open_inside":
        match = re.match(r"(\d{4}_\d{2}_\d{2})_", source)
        if match:
            label = match.group(1)
            return f"prod_day:{label}", f"prod_day:{label}"

    if model == "_未分类_color":
        match = re.search(r"__(\d{8})_color", source)
        if match:
            label = match.group(1)
            return f"color_day:{label}", f"color_day:{label}"

    # A labelled car model is the safest grouping key. For opaque residual
    # datasets, retain the source dataset in the key instead of implying that
    # all placeholder names represent the same physical car.
    if model.startswith("_未分类_"):
        return f"source_file:{dataset}:{source}", f"source_file:{dataset}:{source}"
    return f"model:{_canonical_model(model)}", f"model:{model}"


def build_group_split(rows, seed, val_fraction=0.1, test_fraction=0.1):
    """Split whole generalization groups into balanced train/val/test lists."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")

    groups = defaultdict(list)
    labels = {}
    for row in rows:
        key, label = generalization_group(row)
        groups[key].append(row)
        labels[key] = label

    total = len(rows)
    targets = {
        "val": round(total * val_fraction),
        "test": round(total * test_fraction),
    }
    targets["train"] = total - targets["val"] - targets["test"]
    if min(targets.values()) <= 0 or len(groups) < len(SPLIT_NAMES):
        raise ValueError("not enough samples or groups for a three-way split")

    rng = np.random.default_rng(seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    group_rank = {key: index for index, key in enumerate(group_keys)}
    split_order = list(SPLIT_NAMES)
    rng.shuffle(split_order)
    split_rank = {name: index for index, name in enumerate(split_order)}

    # Largest groups first keeps the sample totals close to their targets. The
    # squared, target-normalized error makes the 80/10/10 proportions compete
    # fairly rather than greedily exhausting validation with tiny groups.
    ordered_keys = sorted(group_keys, key=lambda key: (-len(groups[key]), group_rank[key]))
    counts = {name: 0 for name in SPLIT_NAMES}
    assignment = {}
    for key in ordered_keys:
        size = len(groups[key])

        def score(split_name):
            projected = dict(counts)
            projected[split_name] += size
            return sum(
                ((projected[name] - targets[name]) / targets[name]) ** 2
                for name in SPLIT_NAMES
            )

        chosen = min(SPLIT_NAMES, key=lambda name: (score(name), split_rank[name]))
        assignment[key] = chosen
        counts[chosen] += size

    split = {name: [] for name in SPLIT_NAMES}
    file_split = {}
    for key, members in groups.items():
        split_name = assignment[key]
        for row in members:
            file_name = str(row["file"])
            if file_name in file_split:
                raise ValueError(f"duplicate file name: {file_name}")
            split[split_name].append(file_name)
            file_split[file_name] = split_name

    metadata = {
        "schema": "generalization_group_v1",
        "seed": int(seed),
        "targets": targets,
        "counts": counts,
        "groups": {
            key: {"label": labels[key], "split": assignment[key], "samples": len(members)}
            for key, members in sorted(groups.items())
        },
    }
    return split, file_split, metadata

