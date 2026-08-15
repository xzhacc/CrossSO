from __future__ import annotations

import csv
import json
import math
import numbers
import os
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from crossso.data_prep.osm_labels import classify_osm_tags
from crossso.data_prep.profile import DataProfile


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) +
                         "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def download_osm_extracts(
    profile: DataProfile,
    work_dir: str | Path,
    regions: Iterable[str],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    url_template: str | None = None,
) -> dict[str, Any]:

    root = Path(work_dir).expanduser().resolve()
    osm_dir = root / "osm"
    template = url_template or str(profile.raw["osm"]["url_template"])
    records: list[dict[str, Any]] = []
    for region in regions:
        settings = profile.region(region)
        slug = str(settings["geofabrik_slug"])
        url = template.format(slug=slug, region=region)
        target = osm_dir / f"{slug}-latest.osm.pbf"
        record: dict[str, Any] = {
            "region": region,
            "url": url,
            "path": str(target)
        }
        if dry_run:
            record["status"] = "planned"
            records.append(record)
            continue
        osm_dir.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not overwrite:
            record.update(
                status="existing",
                bytes=target.stat().st_size,
            )
            records.append(record)
            continue
        with tempfile.NamedTemporaryFile(dir=osm_dir,
                                         prefix=f".{slug}.",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        if temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded an empty OSM extract from {url}")
        os.replace(temporary, target)
        record.update(
            status="downloaded",
            bytes=target.stat().st_size,
        )
        records.append(record)
    return {
        "profile": profile.name,
        "extracts": records,
    }


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


def download_footprints(
    profile: DataProfile,
    work_dir: str | Path,
    regions: Iterable[str],
    *,
    ee_project: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:

    root = Path(work_dir).expanduser().resolve()
    output = root / "footprints"
    imagery = profile.raw["imagery"]
    planned: list[dict[str, Any]] = []
    if dry_run:
        for region in regions:
            settings = profile.region(region)
            slug = str(settings["footprint_slug"])
            sentinel_enabled = bool(settings.get("sentinel_footprint", True))
            naip_enabled = bool(settings.get("naip_footprint", True))
            planned.append({
                "region":
                region,
                "sentinel": {
                    "path":
                    str(output / f"{slug}_sentinel2_footprint.geojson"),
                    "status": "planned" if sentinel_enabled else "disabled",
                },
                "naip": {
                    "path": str(output / f"{slug}_naip_footprint.geojson"),
                    "status": "planned" if naip_enabled else "disabled",
                },
                "status":
                "planned" if sentinel_enabled or naip_enabled else "disabled",
            })
        return {"profile": profile.name, "footprints": planned}
    ee = _initialize_ee(ee_project)
    output.mkdir(parents=True, exist_ok=True)
    states = ee.FeatureCollection("TIGER/2018/States")
    for region in regions:
        settings = profile.region(region)
        slug = str(settings["footprint_slug"])
        item: dict[str, Any] = {"region": region}
        sensors = (
            (
                "sentinel",
                bool(settings.get("sentinel_footprint", True)),
                str(
                    settings.get("sentinel_footprint_collection",
                                 imagery["sentinel_collection"])),
                str(imagery["sentinel_start"]),
                str(imagery["sentinel_end"]),
                str(
                    imagery.get("sentinel_footprint_geometry_mode",
                                "intersection")),
                float(imagery.get("sentinel_footprint_max_error_m", 0.001)),
                output / f"{slug}_sentinel2_footprint.geojson",
            ),
            (
                "naip",
                bool(settings.get("naip_footprint", True)),
                str(imagery["naip_collection"]),
                str(
                    settings.get(
                        "naip_footprint_start",
                        imagery.get("naip_footprint_start",
                                    imagery["naip_start"]),
                    )),
                str(
                    settings.get(
                        "naip_footprint_end",
                        imagery.get("naip_footprint_end", imagery["naip_end"]),
                    )),
                str(imagery.get("naip_footprint_geometry_mode",
                                "intersection")),
                float(imagery.get("naip_footprint_max_error_m", 0.001)),
                output / f"{slug}_naip_footprint.geojson",
            ),
        )
        if not any(spec[1] for spec in sensors):
            for sensor, _, _, _, _, _, _, target in sensors:
                item[sensor] = {
                    "path": str(target),
                    "status": "disabled",
                }
            planned.append(item)
            continue
        state_name = str(settings["display_name"])
        state = states.filter(ee.Filter.eq("NAME", state_name)).first()
        state_geometry = state.geometry()
        for sensor, enabled, collection, start, end, geometry_mode, max_error, target in sensors:
            if not enabled:
                item[sensor] = {"path": str(target), "status": "disabled"}
                continue
            if target.is_file() and not overwrite:
                item[sensor] = {"path": str(target), "status": "existing"}
                continue
            geometry = (ee.ImageCollection(collection).filterDate(
                start,
                end).filterBounds(state_geometry).geometry().intersection(
                    state_geometry, max_error))
            if geometry_mode == "bounds":
                geometry = geometry.bounds(max_error)
            elif geometry_mode != "intersection":
                raise ValueError(
                    f"Unsupported footprint geometry mode: {geometry_mode}")
            feature = {
                "type":
                "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": geometry.getInfo(),
                }],
            }
            _write_json(target, feature)
            item[sensor] = {"path": str(target), "status": "generated"}
        planned.append(item)
    return {"profile": profile.name, "footprints": planned}


def _load_footprint(path: Path):
    if not path.is_file():
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as error:
        raise RuntimeError(
            "OSM indexing requires `pip install -e '.[data]'`") from error
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("type") == "FeatureCollection":
        geometries = [
            shape(item["geometry"]) for item in raw.get("features", ())
            if item.get("geometry")
        ]
        return unary_union(geometries) if geometries else None
    if raw.get("type") == "Feature":
        return shape(raw["geometry"])
    return shape(raw)


def _collect_osm_geometries(
    profile: DataProfile,
    pbf_path: Path,
) -> tuple[list[Any], list[list[int]], list[tuple[float, float]]]:
    try:
        import osmium
        import shapely.wkb
    except ImportError as error:
        raise RuntimeError(
            "OSM indexing requires `pip install -e '.[data]'`") from error
    whitelist = set(
        str(value) for value in profile.raw["osm"].get("tag_keys", ()))
    geoms: list[Any] = []
    labels: list[list[int]] = []
    centers: list[tuple[float, float]] = []

    class Handler(osmium.SimpleHandler):

        def __init__(self):
            super().__init__()
            self.factory = osmium.geom.WKBFactory()

        def area(self, area):
            tags = {tag.k: tag.v for tag in area.tags}
            if not tags or not whitelist.intersection(tags):
                return
            class_ids = classify_osm_tags(tags)
            if not class_ids:
                return
            try:
                geometry = shapely.wkb.loads(
                    self.factory.create_multipolygon(area), hex=True)
            except Exception:
                return
            if geometry.is_empty or not geometry.is_valid:
                return
            geoms.append(geometry)
            labels.append(class_ids)
            centers.append(
                (float(geometry.centroid.y), float(geometry.centroid.x)))

    handler = Handler()
    handler.apply_file(str(pbf_path), locations=True)
    return geoms, labels, centers


def _thin_centers(
    centers: Iterable[tuple[float, float]],
    *,
    spacing: float,
    max_per_cell: int,
    rounding: str,
) -> list[tuple[float, float]]:
    if spacing <= 0:
        return list(centers)
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    kept: list[tuple[float, float]] = []
    bucket = math.floor if rounding == "floor" else int
    for lat, lon in centers:
        key = (bucket(lat / spacing), bucket(lon / spacing))
        if counts[key] >= max_per_cell:
            continue
        counts[key] += 1
        kept.append((lat, lon))
    return kept


def _query_indices(tree, rectangle, id_to_index: Mapping[int,
                                                         int]) -> list[int]:
    hits = tree.query(rectangle)
    if hits is None or len(hits) == 0:
        return []
    if isinstance(hits[0], numbers.Integral):
        return [int(value) for value in hits]
    return [
        id_to_index[id(geometry)] for geometry in hits
        if id(geometry) in id_to_index
    ]


def _patch_at(
    center: tuple[float, float],
    *,
    tree,
    geoms: list[Any],
    labels: list[list[int]],
    id_to_index: Mapping[int, int],
    osm: Mapping[str, Any],
    sentinel_footprint,
    naip_footprint,
) -> dict[str, Any] | None:
    from shapely.geometry import box
    from shapely.ops import unary_union

    lat, lon = center
    halfwidth = float(osm["halfwidth"])
    if str(osm.get("halfwidth_unit", "degrees")) == "meters":
        half_lat = halfwidth / 111_320.0
        half_lon = halfwidth / (111_320.0 * math.cos(math.radians(lat)) +
                                1e-12)
    else:
        half_lat = half_lon = halfwidth
    scale = float(osm["sentinel_scale"])
    min_lat, max_lat = lat - half_lat * scale, lat + half_lat * scale
    min_lon, max_lon = lon - half_lon * scale, lon + half_lon * scale
    rectangle = box(min_lon, min_lat, max_lon, max_lat)
    if sentinel_footprint is not None:
        predicate = str(osm.get("sentinel_footprint_predicate", "covers"))
        accepted = (sentinel_footprint.contains(rectangle) if predicate
                    == "contains" else sentinel_footprint.covers(rectangle))
        if not accepted:
            return None
    indices = _query_indices(tree, rectangle, id_to_index)
    class_area: defaultdict[int, float] = defaultdict(float)
    intersections: list[Any] = []
    clipped: list[tuple[Any, list[int]]] = []
    for index in indices:
        geometry = geoms[index]
        if not geometry.intersects(rectangle):
            continue
        intersection = geometry.intersection(rectangle)
        if intersection.is_empty or intersection.area <= 0:
            continue
        area = float(intersection.area)
        for class_id in labels[index]:
            class_area[class_id] += area
        intersections.append(intersection)
        clipped.append((intersection, labels[index]))
    if len(class_area) < int(osm["min_num_classes"]):
        return None
    coverage_min = float(osm.get("coverage_min", 0.0))
    if coverage_min > 0:
        coverage = float(unary_union(intersections).area) / float(
            rectangle.area)
        if coverage < coverage_min:
            return None
    total = sum(class_area.values())
    classes = sorted(class_area)
    relative = {key: class_area[key] / total for key in classes}
    absolute = {
        key: class_area[key] / float(rectangle.area)
        for key in classes
    }
    grid = int(osm["num_tiles"])
    height = (max_lat - min_lat) / grid
    width = (max_lon - min_lon) / grid
    tiles: list[tuple[Any, ...]] = []
    for row in range(grid):
        tile_max_lat = max_lat - row * height
        tile_min_lat = max_lat - (row + 1) * height
        for column in range(grid):
            tile_min_lon = min_lon + column * width
            tile_max_lon = min_lon + (column + 1) * width
            tile = box(tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)
            if naip_footprint is not None and not naip_footprint.intersects(
                    tile):
                continue
            areas: defaultdict[int, float] = defaultdict(float)
            for geometry, class_ids in clipped:
                if not geometry.intersects(tile):
                    continue
                area = float(geometry.intersection(tile).area)
                if area <= 0:
                    continue
                for class_id in class_ids:
                    areas[class_id] += area
            denominator = sum(areas.values())
            fractions = {
                key: areas[key] / denominator
                for key in areas
            } if denominator else {}
            dominant: int | str = max(fractions,
                                      key=fractions.get) if fractions else ""
            tiles.append((
                row,
                column,
                row * grid + column,
                tile_min_lat,
                tile_min_lon,
                tile_max_lat,
                tile_max_lon,
                dominant,
                fractions,
            ))
    if not tiles:
        return None
    return {
        "center": center,
        "bounds": (min_lat, min_lon, max_lat, max_lon),
        "classes": classes,
        "relative": relative,
        "absolute": absolute,
        "class_area": dict(class_area),
        "tiles": tiles,
    }


def _write_index(
    output: Path,
    patches: Iterable[dict[str, Any]],
) -> int:
    sentinel_final = output / "sentinel_patches.csv"
    naip_final = output / "naip_tiles.csv"
    sentinel_path = output / ".sentinel_patches.csv.tmp"
    naip_path = output / ".naip_tiles.csv.tmp"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with sentinel_path.open("w", newline="",
                            encoding="utf-8") as sent_handle, naip_path.open(
                                "w", newline="",
                                encoding="utf-8") as naip_handle:
        sent = csv.writer(sent_handle)
        naip = csv.writer(naip_handle)
        sent.writerow([
            "patch_id", "center_lat", "center_lon", "sent_min_lat",
            "sent_min_lon", "sent_max_lat", "sent_max_lon", "classes",
            "class_fracs", "class_fracs_abs"
        ])
        naip.writerow([
            "patch_id", "tile_row", "tile_col", "tile_index", "tile_min_lat",
            "tile_min_lon", "tile_max_lat", "tile_max_lon", "tile_class",
            "tile_class_fracs"
        ])
        for patch in patches:
            count += 1
            lat, lon = patch["center"]
            min_lat, min_lon, max_lat, max_lon = patch["bounds"]
            classes = patch["classes"]
            sent.writerow([
                count,
                lat,
                lon,
                min_lat,
                min_lon,
                max_lat,
                max_lon,
                "|".join(map(str, classes)),
                "|".join(f"{key}:{patch['relative'][key]:.4f}"
                         for key in classes),
                "|".join(f"{key}:{patch['absolute'][key]:.8f}"
                         for key in classes),
            ])
            for row, column, index, tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon, dominant, fractions in patch[
                    "tiles"]:
                naip.writerow([
                    count,
                    row,
                    column,
                    index,
                    tile_min_lat,
                    tile_min_lon,
                    tile_max_lat,
                    tile_max_lon,
                    dominant,
                    "|".join(f"{key}:{fractions[key]:.4f}"
                             for key in sorted(fractions)),
                ])
    os.replace(sentinel_path, sentinel_final)
    os.replace(naip_path, naip_final)
    return count


def build_osm_indices(
    profile: DataProfile,
    work_dir: str | Path,
    regions: Iterable[str],
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:

    try:
        from shapely.strtree import STRtree
    except ImportError as error:
        raise RuntimeError(
            "OSM indexing requires `pip install -e '.[data]'`") from error
    root = Path(work_dir).expanduser().resolve()
    results: list[dict[str, Any]] = []
    osm = profile.raw["osm"]
    for region in regions:
        settings = profile.region(region)
        pbf = root / "osm" / f"{settings['geofabrik_slug']}-latest.osm.pbf"
        if not pbf.is_file():
            raise FileNotFoundError(f"Missing OSM extract for {region}: {pbf}")
        output = root / "data" / region
        artifacts = (
            output / "sentinel_patches.csv",
            output / "naip_tiles.csv",
        )
        present = [path.is_file() for path in artifacts]
        if any(present) and not overwrite:
            if not all(present):
                raise RuntimeError(
                    f"Partial index exists for {region}; use --overwrite for {output}"
                )
            results.append({
                "region": region,
                "status": "existing",
                "path": str(artifacts[0])
            })
            continue
        geoms, labels, centers = _collect_osm_geometries(profile, pbf)
        if not geoms:
            raise RuntimeError(f"No matching OSM geometries found in {pbf}")
        tree = STRtree(geoms)
        id_to_index = {
            id(geometry): index
            for index, geometry in enumerate(geoms)
        }
        centers = _thin_centers(
            centers,
            spacing=float(osm["min_center_spacing_degrees"]),
            max_per_cell=int(osm["max_centers_per_cell"]),
            rounding=str(osm.get("grid_rounding", "floor")),
        )
        if limit is not None:
            centers = centers[:limit]
        footprint_slug = str(settings["footprint_slug"])
        sentinel_footprint = (_load_footprint(
            root / "footprints" /
            f"{footprint_slug}_sentinel2_footprint.geojson") if bool(
                settings.get("sentinel_footprint", True)) else None)
        naip_footprint = (_load_footprint(
            root / "footprints" /
            f"{footprint_slug}_naip_footprint.geojson") if bool(
                settings.get("naip_footprint", True)) else None)
        patches = (patch for center in centers if (patch := _patch_at(
            center,
            tree=tree,
            geoms=geoms,
            labels=labels,
            id_to_index=id_to_index,
            osm=osm,
            sentinel_footprint=sentinel_footprint,
            naip_footprint=naip_footprint,
        )) is not None)
        count = _write_index(output, patches)
        results.append({
            "region": region,
            "status": "built",
            "patches": count,
            "path": str(output)
        })
    return {"profile": profile.name, "regions": results}


__all__ = [
    "build_osm_indices",
    "download_footprints",
    "download_osm_extracts",
]
