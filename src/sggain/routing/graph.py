from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx

from sggain.gis.crs import SVY21
from sggain.gis.geometry import reverse_line


def build_multidigraph(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = str(edges.crs or nodes.crs or SVY21)

    for _, row in nodes.iterrows():
        graph.add_node(
            row["node_id"],
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            geometry=row.geometry,
        )

    for _, row in edges.iterrows():
        _add_directed_edge(graph, row, forward=True)
        _add_directed_edge(graph, row, forward=False)
    return graph


def edge_index_by_directed_id(graph: nx.MultiDiGraph) -> dict[str, tuple[str, str, int, dict[str, Any]]]:
    index: dict[str, tuple[str, str, int, dict[str, Any]]] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        index[data["directed_edge_id"]] = (u, v, key, data)
    return index


def ordered_edge_data(graph: nx.MultiDiGraph, directed_edge_ids: list[str]) -> list[dict[str, Any]]:
    index = edge_index_by_directed_id(graph)
    missing = [edge_id for edge_id in directed_edge_ids if edge_id not in index]
    if missing:
        raise KeyError(f"Directed edge IDs not found in graph: {missing}")
    return [index[edge_id][3] for edge_id in directed_edge_ids]


def _add_directed_edge(graph: nx.MultiDiGraph, row, forward: bool) -> None:
    u = row["u"] if forward else row["v"]
    v = row["v"] if forward else row["u"]
    suffix = "fwd" if forward else "rev"
    geometry = row.geometry if forward else reverse_line(row.geometry)
    ascent_key = "ascent_fwd_m" if forward else "ascent_rev_m"
    descent_key = "descent_fwd_m" if forward else "descent_rev_m"
    edge_id = str(row["edge_id"])
    directed_edge_id = f"{edge_id}:{suffix}"

    data = {
        "directed_edge_id": directed_edge_id,
        "undirected_edge_id": edge_id,
        "geometry": geometry,
        "length_m": float(row.get("length_m", geometry.length) or geometry.length),
        "ascent_m": float(row.get(ascent_key, 0.0) or 0.0),
        "descent_m": float(row.get(descent_key, 0.0) or 0.0),
        "source_primary": row.get("source_primary", "unknown"),
        "source_confidence": float(row.get("source_confidence", 0.5) or 0.5),
    }
    for optional in ["highway", "name", "trail_type", "source_feature_id", "source_row_index", "source_raw_properties_json"]:
        if optional in row and row[optional] is not None:
            data[optional] = row[optional]
    if forward:
        profile_keys = {
            "profile_distance_m": "profile_distance_m",
            "profile_elevation_raw_m": "profile_elevation_raw_m",
            "profile_elevation_smooth_m": "profile_elevation_smooth_m",
        }
    else:
        profile_keys = {
            "profile_distance_rev_m": "profile_distance_m",
            "profile_elevation_raw_rev_m": "profile_elevation_raw_m",
            "profile_elevation_smooth_rev_m": "profile_elevation_smooth_m",
        }
    for source_key, target_key in profile_keys.items():
        if source_key in row and row[source_key] is not None:
            data[target_key] = row[source_key]
    if "elevation_fallback_count" in row and row["elevation_fallback_count"] is not None:
        data["elevation_fallback_count"] = row["elevation_fallback_count"]
    graph.add_edge(u, v, **data)
