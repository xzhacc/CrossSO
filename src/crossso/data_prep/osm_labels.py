from __future__ import annotations

from collections.abc import Callable, Mapping

Tags = Mapping[str, str]
Rule = Callable[[Tags], bool]


def _eq(tags: Tags, key: str, value: str) -> bool:
    return str(tags.get(key, "")) == value


def _one(tags: Tags, key: str, values: set[str]) -> bool:
    return str(tags.get(key, "")) in values


def _token(tags: Tags, key: str, value: str) -> bool:
    raw = str(tags.get(key, ""))
    for separator in (";", ",", "|", " "):
        if separator in raw:
            return value in {
                part.strip()
                for part in raw.split(separator) if part.strip()
            }
    return value == raw or value in raw


def _rules66() -> tuple[Rule, ...]:
    marina = lambda t: _eq(t, "leisure", "marina") or _eq(
        t, "harbour", "marina") or _eq(t, "waterway", "marina"
                                       ) or "mooring" in t
    highway = lambda t: _one(t, "highway", {
        "motorway", "trunk", "primary", "secondary", "tertiary"
    }) and not _eq(t, "bridge", "yes") and not _eq(
        t, "tunnel", "yes") and not _eq(t, "junction", "roundabout")
    return (
        lambda t: _eq(t, "leisure", "pitch") and _token(t, "sport", "tennis"),
        lambda t: _eq(t, "leisure", "skate_park") or
        (_eq(t, "leisure", "pitch") and _token(t, "sport", "skateboard")),
        lambda t: _eq(t, "leisure", "pitch") and _token(
            t, "sport", "american_football"),
        lambda t: _eq(t, "leisure", "swimming_pool"),
        lambda t: _eq(t, "leisure", "golf_course"),
        lambda t: _eq(t, "leisure", "pitch") and _token(
            t, "sport", "baseball"),
        lambda t: _eq(t, "leisure", "stadium"),
        lambda t: _eq(t, "junction", "roundabout"),
        lambda t: _eq(t, "aeroway", "aerodrome"),
        lambda t: not marina(t) and (_eq(t, "landuse", "port") or _eq(
            t, "industrial", "port") or _eq(t, "landuse", "harbour")),
        lambda t: _eq(t, "railway", "station") or _eq(t, "public_transport",
                                                      "station"),
        lambda t: _eq(t, "aeroway", "runway"),
        lambda t: _eq(t, "aeroway", "hangar") or _eq(t, "building", "hangar"),
        lambda t: _eq(t, "aeroway", "terminal") or _eq(t, "building",
                                                       "terminal"),
        lambda t: _eq(t, "aeroway", "helipad"),
        lambda t: _eq(t, "waterway", "dam") or _eq(t, "man_made", "dam"),
        lambda t: _eq(t, "power", "substation"),
        lambda t:
        (_eq(t, "man_made", "shipyard") or _eq(t, "industrial", "shipyard")
         ) and (_eq(t, "landuse", "industrial") or _eq(t, "man_made", "works")
                or "industrial" in t or "building" in t),
        lambda t:
        (_eq(t, "power", "plant") and _eq(t, "plant:source", "solar")) or
        (_eq(t, "power", "generator") and _eq(t, "generator:source", "solar")),
        lambda t:
        (_eq(t, "power", "plant") and _eq(t, "plant:source", "wind")) or
        (_eq(t, "power", "generator") and
         (_eq(t, "generator:source", "wind") or _eq(
             t, "generator:method", "wind_turbine"))) or _eq(
                 t, "man_made", "wind_turbine"),
        lambda t: _eq(t, "landuse", "quarry"),
        lambda t: _eq(t, "landuse", "landfill") or _eq(t, "amenity",
                                                       "waste_disposal"),
        lambda t: _one(t, "man_made", {"wastewater_plant", "water_works"}),
        marina,
        lambda t: _eq(t, "natural", "beach"),
        lambda t: _eq(t, "amenity", "parking") and _one(
            t, "parking", {"multi-storey", "underground"}),
        lambda t: _eq(t, "amenity", "parking") and _eq(t, "parking", "surface"
                                                       ),
        lambda t: _one(t, "leisure", {"pitch", "sports_centre", "stadium"}
                       ) and _token(t, "sport", "soccer"),
        lambda t: _eq(t, "landuse", "farmland"),
        lambda t: _eq(t, "landuse", "forest") or _eq(t, "natural", "wood"),
        lambda t: _eq(t, "water", "lake"),
        lambda t: _eq(t, "natural", "wetland"),
        lambda t: _eq(t, "leisure", "park"),
        lambda t: _eq(t, "waterway", "river") or _eq(
            t, "waterway", "riverbank") or _eq(t, "water", "river"),
        lambda t: _eq(t, "natural", "desert"),
        lambda t: _eq(t, "landuse", "meadow"),
        lambda t: _one(t, "natural", {"bare_rock", "scree", "shingle"}),
        lambda t: _eq(t, "natural", "scrub"),
        lambda t: _eq(t, "landuse", "industrial"),
        lambda t: _eq(t, "landuse", "aquaculture") or _eq(
            t, "man_made", "fish_farm") or "aquaculture" in t,
        lambda t: _eq(t, "landuse", "construction"),
        lambda t: _eq(t, "leisure", "nature_reserve") or _eq(
            t, "boundary", "protected_area"),
        lambda t: _eq(t, "amenity", "school"),
        lambda t: _eq(t, "amenity", "university") or _eq(
            t, "building", "university"),
        lambda t: _eq(t, "amenity", "hospital") or _eq(t, "healthcare",
                                                       "hospital"),
        lambda t: _eq(t, "shop", "supermarket"),
        lambda t: _eq(t, "shop", "mall") or _eq(t, "building", "mall"),
        lambda t: _eq(t, "building", "warehouse"),
        lambda t: _eq(t, "building", "office"),
        lambda t: _eq(t, "amenity", "place_of_worship") or _one(
            t, "building", {
                "church", "mosque", "cathedral", "synagogue", "temple",
                "chapel", "shrine"
            }),
        lambda t: _eq(t, "amenity", "fire_station"),
        lambda t: _eq(t, "amenity", "police"),
        lambda t: _eq(t, "amenity", "prison") or _eq(
            t, "landuse", "prison") or _eq(t, "building", "prison"),
        lambda t: _eq(t, "amenity", "fuel"),
        lambda t: _eq(t, "building", "barn"),
        lambda t: _eq(t, "shop", "car") or _eq(t, "amenity", "car_dealer"),
        lambda t: _one(t, "railway", {
            "rail", "light_rail", "subway", "tram", "narrow_gauge", "monorail"
        }),
        highway,
        lambda t: _eq(t, "bridge", "yes") and
        ("highway" in t or "railway" in t),
        lambda t: _eq(t, "bridge", "viaduct"),
        lambda t: _eq(t, "man_made", "tower"),
        lambda t: _eq(t, "barrier", "toll_booth") or _one(
            t, "highway", {"toll_gantry", "toll_booth"}),
        lambda t: _eq(t, "tunnel", "yes") and ("railway" in t or _one(
            t, "highway",
            {"motorway", "trunk", "primary", "secondary", "tertiary"})),
        lambda t: _eq(t, "water", "pond"),
        lambda t: _eq(t, "natural", "sand"),
        lambda t: _eq(t, "landuse", "cemetery") or _eq(t, "amenity",
                                                       "grave_yard"),
    )


_RULES = _rules66()


def classify_osm_tags(tags: Tags) -> list[int]:
    return [index for index, rule in enumerate(_RULES) if bool(rule(tags))]


__all__ = ["classify_osm_tags"]
