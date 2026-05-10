from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from shapely.geometry import LineString, Point

from sggain.gis.geometry import densify_line, reverse_line


@dataclass
class ElevationModel:
    linear: LinearNDInterpolator | None
    nearest: NearestNDInterpolator

    @classmethod
    def from_points(cls, points: gpd.GeoDataFrame) -> "ElevationModel":
        if points.empty:
            raise ValueError("Cannot build elevation model from no points")
        xy = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
        z = points["elevation_m"].astype(float).to_numpy()
        linear: LinearNDInterpolator | None
        try:
            linear = LinearNDInterpolator(xy, z)
        except Exception:
            linear = None
        nearest = NearestNDInterpolator(xy, z)
        return cls(linear=linear, nearest=nearest)

    def interpolate(self, xy: np.ndarray) -> tuple[np.ndarray, int]:
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be an Nx2 array")
        if self.linear is None:
            return np.asarray(self.nearest(xy), dtype=float), int(len(xy))
        values = np.asarray(self.linear(xy), dtype=float)
        mask = np.isnan(values)
        fallback_count = int(mask.sum())
        if fallback_count:
            values[mask] = np.asarray(self.nearest(xy[mask]), dtype=float)
        return values, fallback_count


def densify_contours_to_points(
    contours: gpd.GeoDataFrame,
    spacing_m: float = 100,
    max_points: int | None = None,
) -> gpd.GeoDataFrame:
    estimated_points = int((contours.geometry.length / max(spacing_m, 1e-9)).sum()) + len(contours)
    stride = 1
    if max_points and estimated_points > max_points:
        stride = int(np.ceil(estimated_points / max_points))
    rows: list[dict[str, Any]] = []
    point_counter = 0
    for _, row in contours.iterrows():
        elevation = float(row["elevation_m"])
        for _, point in densify_line(row.geometry, spacing_m):
            point_counter += 1
            if stride > 1 and point_counter % stride != 1:
                continue
            rows.append({"elevation_m": elevation, "geometry": Point(point.x, point.y)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=contours.crs)


def sample_edge_profile(
    line: LineString,
    model: ElevationModel,
    spacing_m: float = 10,
    smooth_window_m: float = 30,
) -> dict[str, Any]:
    samples = densify_line(line, spacing_m)
    distances = np.asarray([distance for distance, _ in samples], dtype=float)
    xy = np.asarray([[point.x, point.y] for _, point in samples], dtype=float)
    raw, fallback_count = model.interpolate(xy)
    smooth = smooth_elevations(raw, distances, smooth_window_m)
    grades = compute_grades(smooth, distances)
    ascent, descent = compute_ascent_descent(smooth)
    return {
        "distance_m": distances.tolist(),
        "elevation_raw_m": raw.tolist(),
        "elevation_smooth_m": smooth.tolist(),
        "grade_smooth_pct": grades.tolist(),
        "ascent_m": ascent,
        "descent_m": descent,
        "fallback_count": fallback_count,
    }


def smooth_elevations(values: np.ndarray, distances_m: np.ndarray, window_m: float = 30) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    distances_m = np.asarray(distances_m, dtype=float)
    if len(values) <= 2:
        return values.copy()
    diffs = np.diff(distances_m)
    positive_diffs = diffs[diffs > 0]
    median_spacing = float(np.median(positive_diffs)) if len(positive_diffs) else window_m
    window_samples = max(1, int(round(window_m / median_spacing)))
    if window_samples % 2 == 0:
        window_samples += 1
    return (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def compute_ascent_descent(values: np.ndarray | list[float]) -> tuple[float, float]:
    diffs = np.diff(np.asarray(values, dtype=float))
    ascent = float(diffs[diffs > 0].sum()) if len(diffs) else 0.0
    descent = float((-diffs[diffs < 0]).sum()) if len(diffs) else 0.0
    return ascent, descent


def compute_grades(values: np.ndarray | list[float], distances_m: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    distances_m = np.asarray(distances_m, dtype=float)
    grades = np.zeros(len(values), dtype=float)
    if len(values) < 2:
        return grades
    delta_d = np.diff(distances_m)
    delta_z = np.diff(values)
    valid = delta_d > 0
    step_grades = np.zeros(len(delta_z), dtype=float)
    step_grades[valid] = (delta_z[valid] / delta_d[valid]) * 100
    grades[1:] = step_grades
    return grades


def enrich_edges_with_elevation(
    edges: gpd.GeoDataFrame,
    model: ElevationModel,
    spacing_m: float = 10,
    smooth_window_m: float = 30,
) -> gpd.GeoDataFrame:
    enriched = edges.copy()
    profiles: list[dict[str, Any]] = []
    for line in enriched.geometry:
        fwd = sample_edge_profile(line, model, spacing_m, smooth_window_m)
        rev = sample_edge_profile(reverse_line(line), model, spacing_m, smooth_window_m)
        profiles.append(
            {
                "ascent_fwd_m": fwd["ascent_m"],
                "descent_fwd_m": fwd["descent_m"],
                "ascent_rev_m": rev["ascent_m"],
                "descent_rev_m": rev["descent_m"],
                "elevation_fallback_count": fwd["fallback_count"] + rev["fallback_count"],
                "profile_distance_m": fwd["distance_m"],
                "profile_elevation_raw_m": fwd["elevation_raw_m"],
                "profile_elevation_smooth_m": fwd["elevation_smooth_m"],
                "profile_distance_rev_m": rev["distance_m"],
                "profile_elevation_raw_rev_m": rev["elevation_raw_m"],
                "profile_elevation_smooth_rev_m": rev["elevation_smooth_m"],
            }
        )
    for key in profiles[0].keys() if profiles else []:
        enriched[key] = [profile[key] for profile in profiles]
    return enriched
