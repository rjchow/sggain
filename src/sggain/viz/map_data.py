from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from shapely.geometry import mapping

from sggain.routing.search import RouteResult
from sggain.viz.data import route_linestring


@dataclass(frozen=True)
class AreaBox:
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


# Local geolabelling keeps route names useful without depending on a geocoding API
# or bundling administrative boundary data.
AREA_BOXES: tuple[AreaBox, ...] = (
    AreaBox("Bukit Timah Nature Reserve", 1.345, 1.362, 103.768, 103.786),
    AreaBox("Dairy Farm Nature Park", 1.360, 1.392, 103.767, 103.786),
    AreaBox("Rifle Range Nature Park", 1.375, 1.400, 103.785, 103.810),
    AreaBox("MacRitchie", 1.340, 1.365, 103.800, 103.840),
    AreaBox("Central Catchment", 1.365, 1.405, 103.800, 103.835),
    AreaBox("Zhenghua", 1.392, 1.412, 103.760, 103.782),
    AreaBox("Bukit Batok", 1.345, 1.372, 103.745, 103.765),
    AreaBox("Bukit Panjang", 1.385, 1.405, 103.750, 103.770),
    AreaBox("Western Catchment", 1.380, 1.435, 103.635, 103.690),
    AreaBox("Jurong West", 1.330, 1.360, 103.645, 103.682),
    AreaBox("Jurong Lake", 1.330, 1.350, 103.720, 103.740),
    AreaBox("Kent Ridge", 1.275, 1.305, 103.775, 103.805),
    AreaBox("Southern Ridges", 1.260, 1.285, 103.805, 103.830),
    AreaBox("Mount Faber", 1.265, 1.275, 103.815, 103.825),
    AreaBox("Pearl's Hill", 1.280, 1.290, 103.835, 103.845),
    AreaBox("Sentosa", 1.210, 1.260, 103.810, 103.865),
    AreaBox("Fort Canning", 1.290, 1.300, 103.840, 103.852),
    AreaBox("Marina Bay", 1.275, 1.295, 103.850, 103.870),
    AreaBox("Bishan-Ang Mo Kio", 1.365, 1.385, 103.835, 103.850),
    AreaBox("Seletar", 1.395, 1.425, 103.855, 103.885),
    AreaBox("Lower Seletar", 1.405, 1.425, 103.805, 103.825),
    AreaBox("Punggol", 1.395, 1.425, 103.890, 103.925),
    AreaBox("Tampines", 1.380, 1.405, 103.918, 103.940),
    AreaBox("Pasir Ris", 1.405, 1.430, 103.925, 103.950),
    AreaBox("Changi Village", 1.405, 1.425, 103.970, 104.000),
    AreaBox("Changi", 1.360, 1.395, 103.950, 103.995),
    AreaBox("Pulau Ubin", 1.390, 1.430, 104.025, 104.085),
    AreaBox("Admiralty", 1.440, 1.460, 103.765, 103.790),
    AreaBox("Woodlands", 1.420, 1.455, 103.670, 103.705),
    AreaBox("Kranji", 1.420, 1.435, 103.670, 103.700),
)


def routes_to_feature_collection(
    routes: list[RouteResult],
    graph,
    route_samples: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    features = []
    for route in routes:
        geometry = mapping(route_linestring(route, graph, to_wgs84=True))
        properties = route_summary_record(route, (route_samples or {}).get(route.route_id, []))
        properties.pop("directed_edge_ids", None)
        properties.pop("undirected_edge_ids", None)
        properties.pop("node_ids", None)
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return {"type": "FeatureCollection", "features": features}


def route_summary_records(
    routes: list[RouteResult],
    route_samples: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    samples_by_route = route_samples or {}
    return [route_summary_record(route, samples_by_route.get(route.route_id, [])) for route in routes]


def route_summary_record(route: RouteResult, samples: list[dict[str, Any]]) -> dict[str, Any]:
    record = route.to_dict()
    record["display_name"] = route_display_name(route, samples)
    return record


def route_display_name(route: RouteResult, samples: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for sample in samples:
        name = sample.get("path_name")
        if name and name not in names:
            names.append(str(name))
        if len(names) >= 2:
            break
    area = route_area_name(samples)
    if names and area:
        label = f"{area}: {' + '.join(names)}"
    elif names:
        label = " + ".join(names)
    elif area:
        label = f"{area} {_route_kind(route)}"
    else:
        label = f"{route.mode.replace('_', ' ').title()} route"
    return f"{label} - {route.distance_m / 1000:.2f} km, {route.ascent_m:.0f} m gain"


def route_area_name(samples: list[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for sample in samples:
        lat = sample.get("lat")
        lon = sample.get("lon")
        if not isinstance(lat, int | float) or not isinstance(lon, int | float):
            continue
        for area in AREA_BOXES:
            if area.contains(float(lat), float(lon)):
                counts[area.name] += 1
                break
    if not counts:
        return None
    total = sum(counts.values())
    ranked = counts.most_common(2)
    primary, primary_count = ranked[0]
    if len(ranked) == 1:
        return primary
    secondary, secondary_count = ranked[1]
    if secondary_count / total >= 0.25 and secondary != primary:
        return f"{primary} to {secondary}"
    if primary_count / total < 0.5:
        return f"{primary} area"
    return primary


def _route_kind(route: RouteResult) -> str:
    return "loop" if route.mode == "loop" else "route"
