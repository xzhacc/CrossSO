from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from crossso.data_prep.profile import DataProfile


def parse_scene_name(value: str) -> tuple[int, float, float, list[int]]:
    parts = Path(value).stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected scene filename: {value}")
    try:
        return int(parts[0]), float(parts[1]), float(
            parts[2]), [int(item) for item in parts[3:]]
    except ValueError as error:
        raise ValueError(f"Unexpected scene filename: {value}") from error


def collect_scenes(
    data_root: str | Path,
    regions: Iterable[str],
    *,
    max_labels: int,
    class_count: int,
    tile_count: int = 100,
    absolute_paths: bool = False,
) -> list[dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for region in regions:
        lr_dir = root / region / "LR"
        if not lr_dir.is_dir():
            raise FileNotFoundError(
                f"Missing LR directory for required region {region}: {lr_dir}")
        for lr_path in sorted(lr_dir.glob("*.jpg")):
            patch_id, lat, lon, labels = parse_scene_name(lr_path.name)
            invalid = sorted(set(labels) - set(range(class_count)))
            if invalid:
                raise ValueError(
                    f"Out-of-range labels in {lr_path}: {invalid}")
            if len(labels) > max_labels:
                raise ValueError(
                    f"Scene has {len(labels)} labels but max_labels={max_labels}: {lr_path}"
                )
            hr_dir = root / region / "HR" / lr_path.stem
            missing_tile = next(
                (hr_dir / f"{index}.jpg" for index in range(tile_count)
                 if not (hr_dir / f"{index}.jpg").is_file()),
                None,
            )
            if missing_tile is not None:
                raise FileNotFoundError(
                    f"Incomplete LR/HR pair for {lr_path.name}: {missing_tile}"
                )
            path = lr_path if absolute_paths else lr_path.relative_to(root)
            row: dict[str, Any] = {
                "id": patch_id,
                "lat": lat,
                "lon": lon,
                "labels": labels,
                "region": region,
                "filepath": str(path),
            }
            for index in range(max_labels):
                row[f"label_{index}"] = labels[index] if index < len(
                    labels) else ""
            rows.append(row)
    return rows


def _write_split(path: Path, rows: Iterable[Mapping[str, Any]],
                 max_labels: int) -> int:
    rows = list(rows)
    fields = [
        "id", "lat", "lon", *(f"label_{index}" for index in range(max_labels)),
        "filepath"
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            writer.writerow({key: source[key] for key in fields})
    os.replace(temporary, path)
    return len(rows)


def _distribution(rows: Iterable[Mapping[str, Any]],
                  class_count: int) -> list[dict[str, Any]]:
    rows = list(rows)
    counts = Counter(label for row in rows for label in row["labels"])
    return [{
        "label_id": label,
        "count": counts[label],
        "ratio": counts[label] / len(rows) if rows else 0.0,
    } for label in sorted(range(class_count),
                          key=lambda value: (-counts[value], value))]


def construct_splits(
    profile: DataProfile,
    work_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    absolute_paths: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:

    work = Path(work_dir).expanduser().resolve()
    data_root = work / "data"
    output = Path(
        output_dir).expanduser().resolve() if output_dir else data_root
    output.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        *(output / f"{split}_loc.csv" for split in ("train", "valid", "test")),
        output / "label_distribution.csv",
    ]
    existing_outputs = [
        str(path) for path in protected_outputs if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing split artifacts without --overwrite: "
            f"{existing_outputs}")
    config = profile.raw["split"]
    class_count = len(profile.class_names)
    tile_count = int(profile.raw["imagery"]["grid_size"])**2
    max_labels = int(config.get("max_labels", 30))
    split_rows = {
        split:
        collect_scenes(
            data_root,
            config[f"{split}_regions"],
            max_labels=max_labels,
            class_count=class_count,
            tile_count=tile_count,
            absolute_paths=absolute_paths,
        )
        for split in ("train", "valid", "test")
    }
    scene_splits: dict[str, str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            key = f"{row['region']}/{Path(row['filepath']).name}"
            previous = scene_splits.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"Scene occurs in both {previous} and {split}: {key}")
    counts = {
        split:
        _write_split(output / f"{split}_loc.csv", split_rows[split],
                     max_labels)
        for split in ("train", "valid", "test")
    }
    distribution = _distribution(split_rows["train"] + split_rows["valid"],
                                 class_count)
    distribution_path = output / "label_distribution.csv"
    distribution_temporary = output / ".label_distribution.csv.tmp"
    with distribution_temporary.open("w", newline="",
                                     encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["label_id", "count", "ratio"])
        writer.writeheader()
        writer.writerows(distribution)
    os.replace(distribution_temporary, distribution_path)
    return {
        "profile": profile.name,
        "counts": counts,
        "label_distribution": str(distribution_path),
    }


__all__ = [
    "collect_scenes",
    "construct_splits",
    "parse_scene_name",
]
