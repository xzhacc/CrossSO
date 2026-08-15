from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _scene_paths(raw_root: Path,
                 filepath: str) -> tuple[str, Path, list[Path]]:
    source = Path(str(filepath))
    if source.parent.name != "LR" or len(source.parents) < 2:
        raise ValueError(f"Expected REGION/LR/image path, got: {filepath}")
    region = source.parent.parent.name
    lr_path = raw_root / region / "LR" / source.name
    hr_dir = raw_root / region / "HR" / source.stem
    return region, lr_path, [hr_dir / f"{index}.jpg" for index in range(100)]


def expand(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    loc_manifest_dir: str | Path | None = None,
    verify_paths: bool = True,
) -> dict[str, int]:

    root = Path(raw_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifests = (Path(loc_manifest_dir).expanduser().resolve()
                 if loc_manifest_dir is not None else root)
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        name: manifests / f"{name}_loc.csv"
        for name in ("train", "valid", "test")
    }
    missing_sources = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Canonical split manifests are required; prepare-data never re-splits scenes. "
            f"Missing: {missing_sources}")

    required = {
        "id", "lat", "lon", "filepath", *(f"label_{i}" for i in range(30))
    }
    counts: dict[str, int] = {}
    scene_split: dict[str, str] = {}

    for split, source_path in source_paths.items():
        frame = pd.read_csv(source_path)
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{source_path} is missing columns: {sorted(missing_columns)}")

        lr_paths: list[Path] = []
        hr_paths_by_scene: list[list[Path]] = []
        for filepath in frame["filepath"].astype(str):
            region, lr_path, hr_paths = _scene_paths(root, filepath)
            scene_key = f"{region}/LR/{lr_path.name}"
            previous_split = scene_split.get(scene_key)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"Scene occurs in both {previous_split} and {split}: {scene_key}"
                )
            scene_split.setdefault(scene_key, split)
            if verify_paths:
                if not lr_path.is_file():
                    raise FileNotFoundError(lr_path)
                missing_hr = next(
                    (path for path in hr_paths if not path.is_file()), None)
                if missing_hr is not None:
                    raise FileNotFoundError(missing_hr)
            lr_paths.append(lr_path)
            hr_paths_by_scene.append(hr_paths)

        columns: dict[str, Any] = {
            "id": frame["id"],
            "lat": frame["lat"],
            "lon": frame["lon"],
            "lr_path": [str(path) for path in lr_paths],
        }
        for index in range(100):
            columns[f"hr_{index}"] = [
                str(paths[index]) for paths in hr_paths_by_scene
            ]
        for index in range(30):
            columns[f"label_{index}"] = frame[f"label_{index}"]

        grouped = pd.DataFrame(columns)
        output_path = output / f"{split}_loc_group.csv"
        grouped.to_csv(output_path, index=False)
        counts[split] = len(grouped)
    return counts


PREPARE_STAGES = (
    "expand",
    "download-osm",
    "download-footprints",
    "build-index",
    "sample-index",
    "download-images",
    "split",
)


def run(
    stage: str = "expand",
    *,
    work_dir: str | Path | None = None,
    regions: list[str] | None = None,
    raw_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    loc_manifest_dir: str | Path | None = None,
    verify_paths: bool = True,
    ee_project: str | None = None,
    workers: int = 1,
    overwrite: bool = False,
    repair: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    url_template: str | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    csv_suffix: str = "_sampled",
    output_region_suffix: str = "",
    absolute_paths: bool = False,
) -> dict[str, Any]:

    if stage not in PREPARE_STAGES:
        raise ValueError(
            f"Unknown prepare-data stage {stage!r}; choose from {PREPARE_STAGES}"
        )
    if stage == "expand":
        if raw_root is None or output_dir is None:
            raise ValueError(
                "prepare-data expand requires --raw-root and --output-dir")
        return expand(
            raw_root,
            output_dir,
            loc_manifest_dir=loc_manifest_dir,
            verify_paths=verify_paths,
        )

    from crossso.data_prep.profile import load_profile, select_regions

    selected_profile = load_profile()
    if work_dir is None:
        raise ValueError(f"prepare-data {stage} requires --work-dir")
    selected_regions = select_regions(selected_profile, regions)

    if stage == "download-osm":
        from crossso.data_prep.indexing import download_osm_extracts

        return download_osm_extracts(
            selected_profile,
            work_dir,
            selected_regions,
            overwrite=overwrite,
            dry_run=dry_run,
            url_template=url_template,
        )
    if stage == "download-footprints":
        from crossso.data_prep.indexing import download_footprints

        return download_footprints(
            selected_profile,
            work_dir,
            selected_regions,
            ee_project=ee_project,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    if stage == "build-index":
        from crossso.data_prep.indexing import build_osm_indices

        if dry_run:
            root = Path(work_dir).expanduser().resolve()
            return {
                "profile":
                selected_profile.name,
                "status":
                "planned",
                "regions": [{
                    "region":
                    region,
                    "pbf":
                    str(root / "osm" /
                        f"{selected_profile.region(region)['geofabrik_slug']}-latest.osm.pbf"
                        ),
                    "output":
                    str(root / "data" / region),
                } for region in selected_regions],
            }
        return build_osm_indices(
            selected_profile,
            work_dir,
            selected_regions,
            overwrite=overwrite,
            limit=limit,
        )
    if stage == "sample-index":
        from crossso.data_prep.sampling import load_patch_rows, sample_index_pair

        root = Path(work_dir).expanduser().resolve()
        summaries = []
        for region in selected_regions:
            index_root = root / "data" / region
            sentinel = index_root / "sentinel_patches.csv"
            naip = index_root / "naip_tiles.csv"
            target = max_samples
            if target is None:
                configured = selected_profile.region(region).get(
                    "sample_target")
                if configured is None:
                    rows, _ = load_patch_rows(sentinel)
                    target = len(rows)
                else:
                    target = int(configured)
            if dry_run:
                summaries.append({
                    "region": region,
                    "status": "planned",
                    "max_samples": target,
                    "output_suffix": csv_suffix,
                })
                continue
            result = sample_index_pair(
                sentinel,
                naip,
                output_suffix=csv_suffix,
                max_samples=target,
                seed=int(seed if seed is not None else selected_profile.
                         raw["split"]["seed"]),
                overwrite=overwrite,
            )
            summaries.append({"region": region, **result})
        return {
            "profile": selected_profile.name,
            "regions": summaries,
        }
    if stage == "download-images":
        from crossso.data_prep.imagery import download_images

        return download_images(
            selected_profile,
            work_dir,
            selected_regions,
            ee_project=ee_project,
            csv_suffix=csv_suffix,
            output_region_suffix=output_region_suffix,
            workers=workers,
            limit=limit,
            overwrite=overwrite,
            repair=repair,
            dry_run=dry_run,
        )
    if stage == "split":
        from crossso.data_prep.splitting import construct_splits

        if regions:
            raise ValueError(
                "prepare-data split uses the configured region sets; omit --regions"
            )
        if dry_run:
            return {
                "profile":
                selected_profile.name,
                "status":
                "planned",
                "work_dir":
                str(Path(work_dir).expanduser().resolve()),
                "output_dir":
                str(Path(output_dir).expanduser().resolve())
                if output_dir else None,
                "mode":
                str(selected_profile.raw["split"]["mode"]),
            }
        return construct_splits(
            selected_profile,
            work_dir,
            output_dir=output_dir,
            absolute_paths=absolute_paths,
            overwrite=overwrite,
        )
    raise AssertionError(stage)


__all__ = ["PREPARE_STAGES", "expand", "run"]
