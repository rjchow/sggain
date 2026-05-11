from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString
from shapely.ops import transform

from sggain.gis.crs import SVY21, WGS84
from sggain.gis.geometry import concatenate_lines
from sggain.routing.graph import ordered_edge_data
from sggain.routing.search import RouteResult

_TO_WGS84 = pyproj.Transformer.from_crs(SVY21, WGS84, always_xy=True).transform


def route_linestring(route: RouteResult, graph, to_wgs84: bool = False) -> LineString:
    edges = ordered_edge_data(graph, route.directed_edge_ids)
    line, warnings = concatenate_lines([edge["geometry"] for edge in edges])
    route.warnings.extend(warning for warning in warnings if warning not in route.warnings)
    return transform(_TO_WGS84, line) if to_wgs84 else line


def build_route_samples(route: RouteResult, graph, spacing_m: float = 10) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cumulative = 0.0
    current_elevation = 0.0
    sample_index = 0
    edges = ordered_edge_data(graph, route.directed_edge_ids)

    for edge_number, edge in enumerate(edges):
        line = edge["geometry"]
        length = float(edge["length_m"])
        local_distances = list(np.arange(0, length, spacing_m))
        if not local_distances or not np.isclose(local_distances[-1], length):
            local_distances.append(length)
        if edge_number > 0 and local_distances and np.isclose(local_distances[0], 0):
            local_distances = local_distances[1:]

        elevations_raw, elevations_smooth = _edge_elevations(edge, local_distances, current_elevation)
        for local_distance, raw, smooth in zip(local_distances, elevations_raw, elevations_smooth, strict=True):
            point = line.interpolate(float(local_distance))
            lon, lat = _TO_WGS84(point.x, point.y)
            records.append(
                {
                    "route_id": route.route_id,
                    "sample_index": sample_index,
                    "lon": float(lon),
                    "lat": float(lat),
                    "x_svy21": float(point.x),
                    "y_svy21": float(point.y),
                    "cum_distance_m": float(cumulative + local_distance),
                    "cum_distance_km": float((cumulative + local_distance) / 1000),
                    "elevation_raw_m": float(raw),
                    "elevation_smooth_m": float(smooth),
                    "grade_pct": 0.0,
                    "grade_smooth_pct": 0.0,
                    "edge_id": edge["undirected_edge_id"],
                    "directed_edge_id": edge["directed_edge_id"],
                    "undirected_edge_id": edge["undirected_edge_id"],
                    "source_primary": edge.get("source_primary", "unknown"),
                    "source_confidence": float(edge.get("source_confidence", 0.5)),
                    "source_feature_id": edge.get("source_feature_id"),
                    "highway": edge.get("highway"),
                    "path_name": edge.get("name"),
                    "trail_type": edge.get("trail_type"),
                }
            )
            sample_index += 1
        cumulative += length
        current_elevation = float(elevations_smooth[-1]) if elevations_smooth else current_elevation

    samples = pd.DataFrame(records)
    if samples.empty:
        return samples
    distances = samples["cum_distance_m"].to_numpy(dtype=float)
    raw = samples["elevation_raw_m"].to_numpy(dtype=float)
    smooth = samples["elevation_smooth_m"].to_numpy(dtype=float)
    samples["grade_pct"] = _grades(raw, distances)
    samples["grade_smooth_pct"] = _grades(smooth, distances)
    return samples


def detect_steep_sections(samples: pd.DataFrame, threshold_pct: float = 12) -> list[dict[str, Any]]:
    if samples.empty:
        return []
    mask = samples["grade_smooth_pct"].abs() >= threshold_pct
    sections: list[dict[str, Any]] = []
    start_index: int | None = None
    previous_index: int | None = None
    for index, is_steep in mask.items():
        if is_steep and start_index is None:
            start_index = int(index)
        if not is_steep and start_index is not None:
            sections.append(_section_payload(samples, start_index, int(previous_index), threshold_pct))
            start_index = None
        previous_index = int(index)
    if start_index is not None and previous_index is not None:
        sections.append(_section_payload(samples, start_index, previous_index, threshold_pct))
    return sections


def _edge_elevations(edge: dict[str, Any], local_distances: list[float], start_elevation: float) -> tuple[list[float], list[float]]:
    if "profile_distance_m" in edge and "profile_elevation_smooth_m" in edge:
        profile_distances = np.asarray(edge["profile_distance_m"], dtype=float)
        smooth = np.interp(local_distances, profile_distances, np.asarray(edge["profile_elevation_smooth_m"], dtype=float))
        raw_source = edge.get("profile_elevation_raw_m", edge["profile_elevation_smooth_m"])
        raw = np.interp(local_distances, profile_distances, np.asarray(raw_source, dtype=float))
        return raw.tolist(), smooth.tolist()
    length = max(float(edge["length_m"]), 1e-9)
    net_gain = float(edge.get("ascent_m", 0.0)) - float(edge.get("descent_m", 0.0))
    values = [start_elevation + net_gain * (float(distance) / length) for distance in local_distances]
    return values, values


def _grades(values: np.ndarray, distances: np.ndarray) -> np.ndarray:
    grades = np.zeros(len(values), dtype=float)
    if len(values) < 2:
        return grades
    delta_d = np.diff(distances)
    delta_z = np.diff(values)
    valid = delta_d > 0
    step = np.zeros(len(delta_z), dtype=float)
    step[valid] = (delta_z[valid] / delta_d[valid]) * 100
    grades[1:] = step
    return grades


def _section_payload(samples: pd.DataFrame, start: int, end: int, threshold_pct: float) -> dict[str, Any]:
    subset = samples.loc[start:end]
    return {
        "start_distance_m": float(subset.iloc[0]["cum_distance_m"]),
        "end_distance_m": float(subset.iloc[-1]["cum_distance_m"]),
        "max_abs_grade_pct": float(subset["grade_smooth_pct"].abs().max()),
        "threshold_pct": float(threshold_pct),
    }
