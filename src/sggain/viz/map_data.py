from __future__ import annotations

from typing import Any

from shapely.geometry import mapping

from sggain.routing.search import RouteResult
from sggain.viz.data import route_linestring


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
    label = " + ".join(names) if names else route.mode.replace("_", " ").title()
    return f"{label} - {route.distance_m / 1000:.2f} km, {route.ascent_m:.0f} m gain"
