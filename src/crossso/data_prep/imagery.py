from __future__ import annotations

import csv
import os
import random
import re
import shutil
import time
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from crossso.data_prep.profile import DataProfile


def _initialize_ee(project: str | None):
    try:
        import ee
    except ImportError as error:
        raise RuntimeError(
            "Earth Engine support requires `pip install -e '.[data]'`"
        ) from error
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def _read_candidates(sentinel_path: Path,
                     naip_path: Path) -> list[dict[str, str]]:
    with sentinel_path.open(newline="", encoding="utf-8") as handle:
        sentinel = {row["patch_id"]: row for row in csv.DictReader(handle)}
    with naip_path.open(newline="", encoding="utf-8") as handle:
        naip_ids = {row["patch_id"] for row in csv.DictReader(handle)}
    common = sorted(set(sentinel).intersection(naip_ids), key=int)
    return [sentinel[patch_id] for patch_id in common]


def _patch_name(row: Mapping[str, str]) -> str:
    classes = str(row.get("classes", "")).replace("|", "_") or "no_class"
    return (f"{int(row['patch_id']):05d}_{float(row['center_lat']):.6f}_"
            f"{float(row['center_lon']):.6f}_{classes}")


def _image_valid(
    path: Path,
    threshold: float,
    *,
    pixel_threshold: int = 0,
    pixel_mode: str = "channels",
) -> bool:
    try:
        with Image.open(path) as image:
            value = np.asarray(image)
        if not value.size:
            return False
        dark = value <= pixel_threshold
        if pixel_mode == "all_channels" and value.ndim == 3:
            dark = np.all(dark, axis=-1)
        return float(np.mean(dark)) < threshold
    except Exception:
        return False


def _pair_status(
    lr_path: Path,
    hr_dir: Path,
    grid: int,
    threshold: float,
    *,
    pixel_threshold: int,
    pixel_mode: str,
) -> str:
    lr_exists = lr_path.is_file()
    tile_paths = [hr_dir / f"{index}.jpg" for index in range(grid * grid)]
    hr_exists = hr_dir.is_dir()
    if lr_exists and hr_exists and all(path.is_file() for path in tile_paths):
        validation = {
            "pixel_threshold": pixel_threshold,
            "pixel_mode": pixel_mode,
        }
        if _image_valid(lr_path, threshold, **validation) and all(
                _image_valid(path, threshold, **validation)
                for path in tile_paths):
            return "complete"
        return "invalid"
    if not lr_exists and not hr_exists:
        return "missing"
    return "incomplete"


def _assert_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Refusing to modify path outside the data region: {path}"
        ) from error


def _remove_pair(lr_path: Path, hr_dir: Path, root: Path) -> None:
    _assert_inside(lr_path, root)
    _assert_inside(hr_dir, root)
    lr_path.unlink(missing_ok=True)
    if hr_dir.is_dir():
        shutil.rmtree(hr_dir)


def _download(url: str, target: Path, *, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                    url, timeout=120) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            return
        except Exception:
            target.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep((2**attempt) + random.random())


def _masked_sentinel(ee, region, settings: Mapping[str, Any], start: str,
                     end: str):
    collection = (ee.ImageCollection(str(
        settings["sentinel_collection"])).filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                float(settings["sentinel_cloud_percentage"]))).filterDate(
                    start, end).filterBounds(region))

    def mask(image):
        scl = image.select("SCL")
        bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(
            scl.eq(9)).Or(scl.eq(10))
        return image.updateMask(bad.Not()).divide(10000)

    return collection.map(mask).median().select(
        list(settings["sentinel_bands"]))


def _naip_source(ee, region, settings: Mapping[str, Any]):
    collection = (ee.ImageCollection(str(
        settings["naip_collection"])).filterBounds(region).filterDate(
            str(settings["naip_start"]),
            str(settings["naip_end"])).sort("system:time_start", False))
    image = collection.first()
    metadata = ee.Dictionary({
        "asset_id": image.get("system:id"),
        "system_index": image.get("system:index"),
        "time_start": image.get("system:time_start"),
    }).getInfo()
    if not metadata or metadata.get("time_start") is None:
        raise RuntimeError(
            "No NAIP image intersects this patch in the configured time window"
        )
    return image, metadata


def _sentinel_window(settings: Mapping[str, Any],
                     naip_millis: int) -> tuple[str, str, str]:
    default_start = datetime.fromisoformat(str(
        settings["sentinel_start"])).replace(tzinfo=timezone.utc)

    default_end = datetime.fromisoformat(str(
        settings["sentinel_end"])).replace(tzinfo=timezone.utc)
    naip_date = datetime.fromtimestamp(float(naip_millis) / 1000.0,
                                       tz=timezone.utc)
    pad = timedelta(days=int(settings["sentinel_time_pad_days"]))
    start = max(default_start, naip_date - pad)
    end = min(default_end, naip_date + pad)
    if start >= end:
        return str(settings["sentinel_start"]), str(
            settings["sentinel_end"]), "default"
    return start.date().isoformat(), end.date().isoformat(), "naip_aligned"


def _download_one(
    ee,
    row: Mapping[str, str],
    *,
    region_root: Path,
    settings: Mapping[str, Any],
    overwrite: bool,
    repair: bool,
) -> dict[str, Any]:
    name = _patch_name(row)
    grid = int(settings["grid_size"])
    threshold = float(settings["black_ratio_threshold"])
    pixel_threshold = int(settings.get("black_pixel_threshold", 0))
    pixel_mode = str(settings.get("black_pixel_mode", "channels"))
    validation = {"pixel_threshold": pixel_threshold, "pixel_mode": pixel_mode}
    lr_path = region_root / "LR" / f"{name}.jpg"
    hr_dir = region_root / "HR" / name
    status = _pair_status(
        lr_path,
        hr_dir,
        grid,
        threshold,
        pixel_threshold=pixel_threshold,
        pixel_mode=pixel_mode,
    )
    if status == "complete" and not overwrite:
        return {
            "patch_id": int(row["patch_id"]),
            "name": name,
            "status": "existing"
        }
    if status != "missing":
        if not (repair or overwrite):
            raise RuntimeError(
                f"Existing pair is {status}: {name}. Re-run with --repair to replace only this pair."
            )
        _remove_pair(lr_path, hr_dir, region_root)
    staging_root = region_root / ".staging"
    staging = staging_root / f"{name}.{uuid.uuid4().hex}"
    stage_lr = staging / "lr.jpg"
    stage_mosaic = staging / "mosaic.jpg"
    stage_hr = staging / "HR"
    staging.mkdir(parents=True, exist_ok=False)
    bounds = [
        float(row["sent_min_lon"]),
        float(row["sent_min_lat"]),
        float(row["sent_max_lon"]),
        float(row["sent_max_lat"]),
    ]
    region = ee.Geometry.Rectangle(bounds)
    try:
        naip, naip_metadata = _naip_source(ee, region, settings)
        aligned = _sentinel_window(settings, int(naip_metadata["time_start"]))
        windows = [aligned]
        default_window = (
            str(settings["sentinel_start"]),
            str(settings["sentinel_end"]),
            "default_fallback",
        )
        if aligned[2] != "default":
            windows.append(default_window)
        sentinel_error: Exception | None = None
        for start, end, _ in windows:
            try:
                sentinel = _masked_sentinel(ee, region, settings, start, end)
                sentinel_url = sentinel.getThumbURL({
                    "format":
                    "jpg",
                    "crs":
                    "EPSG:4326",
                    "region":
                    region,
                    "dimensions":
                    (f"{int(settings['lr_pixels'])}x{int(settings['lr_pixels'])}"
                     ),
                    "min":
                    float(settings["sentinel_min"]),
                    "max":
                    float(settings["sentinel_max"]),
                    "gamma":
                    1.0,
                })
                _download(sentinel_url, stage_lr)
                if not _image_valid(stage_lr, threshold, **validation):
                    raise RuntimeError(
                        "Sentinel thumbnail failed black-pixel validation")
                break
            except Exception as error:
                sentinel_error = error
                stage_lr.unlink(missing_ok=True)
        else:
            raise RuntimeError("Every configured Sentinel time window failed"
                               ) from sentinel_error
        naip_options = {
            "bands":
            list(settings["naip_primary_bands"]),
            "format":
            "jpg",
            "crs":
            "EPSG:4326",
            "region":
            region,
            "dimensions": (f"{int(settings['hr_tile_pixels']) * grid}x"
                           f"{int(settings['hr_tile_pixels']) * grid}"),
            "min":
            0,
            "max":
            255,
        }
        try:
            naip_url = naip.getThumbURL(naip_options)
        except Exception:
            naip_options["bands"] = list(settings["naip_fallback_bands"])
            naip_url = naip.getThumbURL(naip_options)
        _download(naip_url, stage_mosaic)
        if not _image_valid(stage_mosaic, threshold, **validation):
            raise RuntimeError("NAIP mosaic failed the black-pixel validation")
        stage_hr.mkdir()
        with Image.open(stage_mosaic) as mosaic:
            width, height = mosaic.size
            tile_width, tile_height = width // grid, height // grid
            for row_index in range(grid):
                for column_index in range(grid):
                    index = row_index * grid + column_index
                    tile = mosaic.crop((
                        column_index * tile_width,
                        row_index * tile_height,
                        (column_index + 1) * tile_width,
                        (row_index + 1) * tile_height,
                    ))
                    tile.save(stage_hr / f"{index}.jpg", format="JPEG")
        invalid = next(
            (stage_hr / f"{index}.jpg" for index in range(grid * grid)
             if not _image_valid(stage_hr /
                                 f"{index}.jpg", threshold, **validation)),
            None,
        )
        if invalid is not None:
            raise RuntimeError(
                f"NAIP tile failed the black-pixel validation: {invalid.name}")
        lr_path.parent.mkdir(parents=True, exist_ok=True)
        hr_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_hr, hr_dir)
        os.replace(stage_lr, lr_path)
        return {
            "patch_id": int(row["patch_id"]),
            "name": name,
            "status": "downloaded"
        }
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)


def download_images(
    profile: DataProfile,
    work_dir: str | Path,
    regions: Iterable[str],
    *,
    ee_project: str | None = None,
    csv_suffix: str = "_sampled",
    output_region_suffix: str = "",
    workers: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
    repair: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:

    if workers <= 0:
        raise ValueError("workers must be positive")
    for label, value in (
        ("csv_suffix", csv_suffix),
        ("output_region_suffix", output_region_suffix),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]*", value):
            raise ValueError(
                f"{label} may contain only letters, numbers, '_' and '-'")
    root = Path(work_dir).expanduser().resolve()
    region_jobs: list[tuple[str, Path, list[dict[str, str]]]] = []
    for region in regions:
        index_root = root / "data" / region
        sentinel = index_root / f"sentinel_patches{csv_suffix}.csv"
        naip = index_root / f"naip_tiles{csv_suffix}.csv"
        if not sentinel.is_file() or not naip.is_file():
            raise FileNotFoundError(
                f"Missing paired index CSVs: {sentinel}, {naip}")
        candidates = _read_candidates(sentinel, naip)
        if limit is not None:
            candidates = candidates[:limit]
        output_root = root / "data" / f"{region}{output_region_suffix}"
        region_jobs.append((region, output_root, candidates))
    if dry_run:
        return {
            "profile":
            profile.name,
            "status":
            "planned",
            "regions": [{
                "region": region,
                "output": str(output),
                "patches": len(candidates)
            } for region, output, candidates in region_jobs],
        }
    ee = _initialize_ee(ee_project)
    settings = profile.raw["imagery"]
    summaries: list[dict[str, Any]] = []
    for region, output, candidates in region_jobs:
        output.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = defaultdict(int)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _download_one,
                    ee,
                    row,
                    region_root=output,
                    settings=settings,
                    overwrite=overwrite,
                    repair=repair,
                ) for row in candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                counts[str(result["status"])] += 1
        summaries.append({
            "region": region,
            "output": str(output),
            "requested": len(candidates),
            "status_counts": dict(counts),
        })
    return {"profile": profile.name, "regions": summaries}


__all__ = ["download_images"]
