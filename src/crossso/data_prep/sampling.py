from __future__ import annotations

import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_classes(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split("|") if part.strip()]


def load_patch_rows(path: str | Path) -> tuple[list[dict[str, Any]], set[int]]:
    rows: list[dict[str, Any]] = []
    classes: set[int] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"patch_id", "classes"}.issubset(
                reader.fieldnames):
            raise ValueError(f"{path} must contain patch_id and classes")
        for source in reader:
            row: dict[str, Any] = dict(source)
            row["_classes"] = parse_classes(source.get("classes"))
            rows.append(row)
            classes.update(row["_classes"])
    return rows, classes


def sample_labels(
    rows: Iterable[dict[str, Any]],
    *,
    max_samples: int,
    seed: int = 42,
) -> set[str]:

    rows = list(rows)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not rows:
        return set()
    rng = random.Random(seed)
    all_classes = {value for row in rows for value in row["_classes"]}
    selected: set[str] = set()
    counts: defaultdict[int, int] = defaultdict(int)
    balanced_classes = all_classes
    remaining = list(rows)
    rng.shuffle(remaining)
    if not balanced_classes:
        selected.update(
            str(row["patch_id"])
            for row in remaining[:max_samples - len(selected)])
        return selected
    target = float(max_samples) / len(balanced_classes)
    for row in remaining:
        if len(selected) >= max_samples:
            break
        score = sum(
            max(0.0, target - counts[value]) for value in row["_classes"])
        if score <= 0:
            continue
        selected.add(str(row["patch_id"]))
        for value in row["_classes"]:
            counts[value] += 1
    for row in remaining:
        if len(selected) >= max_samples:
            break
        selected.add(str(row["patch_id"]))
    return selected


def _filter_csv(source: Path, target: Path, selected: set[str]) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with source.open(newline="", encoding="utf-8") as src, temporary.open(
            "w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames or "patch_id" not in reader.fieldnames:
            raise ValueError(f"{source} must contain patch_id")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            if str(row["patch_id"]) in selected:
                writer.writerow(row)
                count += 1
    os.replace(temporary, target)
    return count


def sample_index_pair(
    sentinel_csv: str | Path,
    naip_csv: str | Path,
    *,
    output_suffix: str = "_sampled",
    max_samples: int,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    sentinel = Path(sentinel_csv)
    naip = Path(naip_csv)
    if not output_suffix or not re.fullmatch(r"[A-Za-z0-9_-]+", output_suffix):
        raise ValueError("output_suffix must be non-empty and filename-safe")
    rows, _ = load_patch_rows(sentinel)
    selected = sample_labels(rows, max_samples=max_samples, seed=seed)
    sent_out = sentinel.with_name(
        f"{sentinel.stem}{output_suffix}{sentinel.suffix}")
    naip_out = naip.with_name(f"{naip.stem}{output_suffix}{naip.suffix}")
    existing_outputs = [
        str(path) for path in (sent_out, naip_out) if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Refusing to replace sampled indices without --overwrite: "
            f"{existing_outputs}")
    sentinel_count = _filter_csv(sentinel, sent_out, selected)
    tile_count = _filter_csv(naip, naip_out, selected)
    return {
        "selected_patches": sentinel_count,
        "selected_tile_rows": tile_count,
        "sentinel_csv": str(sent_out),
        "naip_csv": str(naip_out),
    }


__all__ = [
    "sample_labels",
    "load_patch_rows",
    "parse_classes",
    "sample_index_pair",
]
