from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DataProfile:
    raw: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["classes"])

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["regions"])

    def region(self, name: str) -> Mapping[str, Any]:
        try:
            return self.raw["regions"][name]
        except KeyError as error:
            raise KeyError(f"Unknown region {name!r} for profile {self.name!r}"
                           ) from error


def _validate(raw: Mapping[str, Any], source: str) -> None:
    required = {
        "schema_version", "name", "classes", "regions", "osm", "imagery",
        "split"
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(
            f"Data profile {source} is missing keys: {sorted(missing)}")
    if int(raw["schema_version"]) != 1:
        raise ValueError(
            f"Unsupported data profile schema in {source}: {raw['schema_version']}"
        )
    classes = raw["classes"]
    if not isinstance(classes, list) or not classes or len(classes) != len(
            set(classes)):
        raise ValueError(
            f"Data profile {source} must contain unique class names")
    regions = raw["regions"]
    if not isinstance(regions, dict) or not regions:
        raise ValueError(
            f"Data profile {source} must contain a region mapping")
    configured = set()
    for key in ("train_regions", "valid_regions", "test_regions"):
        configured.update(raw["split"].get(key, ()))
    unknown = configured - set(regions)

    if unknown:
        raise ValueError(
            f"Data profile {source} split references unknown regions: {sorted(unknown)}"
        )


def load_profile() -> DataProfile:
    source = "gl10m-66"
    payload = files("crossso.data_prep.profiles").joinpath(
        "gl10m-66.yaml").read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Data profile {source} must be a YAML mapping")
    _validate(raw, source)
    return DataProfile(raw=raw)


def select_regions(profile: DataProfile,
                   requested: list[str] | tuple[str, ...] | None) -> list[str]:
    if not requested:
        return list(profile.regions)
    values: list[str] = []
    for item in requested:
        values.extend(part.strip() for part in str(item).split(",")
                      if part.strip())
    unknown = sorted(set(values) - set(profile.regions))
    if unknown:
        raise ValueError(f"Unknown regions for {profile.name}: {unknown}")
    return values


__all__ = ["DataProfile", "load_profile", "select_regions"]
