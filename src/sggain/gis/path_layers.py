from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from sggain.config import AppConfig
from sggain.gis.crs import SVY21, WGS84
from sggain.gis.geometry import explode_lines, line_endpoints
from sggain.gis.io import read_vector
from sggain.sources.osm import load_osm_paths


SOURCE_CONFIDENCE = {
    "nparks_tracks": 1.0,
    "central_nature_reserve": 0.95,
    "park_connector_loop": 0.9,
    "osm": 0.65,
}


def build_paths_from_config(cfg: AppConfig) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    raw_dir = cfg.data_dir / "raw" / "data_gov"
    for source_name in ["nparks_tracks", "central_nature_reserve", "park_connector_loop"]:
        path = _find_downloaded_vector(raw_dir, source_name)
        if path is None:
            continue
        gdf = read_vector(path)
        if source_name == "nparks_tracks" and "ALLOW_WALKING" in gdf.columns:
            gdf = gdf.loc[gdf["ALLOW_WALKING"].map(_truthy)]
        frames.append(normalize_path_layer(gdf, source_name, SOURCE_CONFIDENCE[source_name]))

    osm_paths = load_osm_paths(cfg)
    if osm_paths is not None and not osm_paths.empty:
        frames.append(normalize_path_layer(osm_paths, "osm", SOURCE_CONFIDENCE["osm"]))

    if not frames:
        raise FileNotFoundError("No walkable path layers found. Run `sggain fetch` or configure local files first.")
    return pd.concat(frames, ignore_index=True).pipe(gpd.GeoDataFrame, geometry="geometry", crs=SVY21)


def normalize_path_layer(gdf: gpd.GeoDataFrame, source_primary: str, source_confidence: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["source_primary", "source_confidence", "geometry"], geometry="geometry", crs=SVY21)
    working = gdf.copy()
    if working.crs is None:
        working = working.set_crs(WGS84)
    working = working.to_crs(SVY21)
    rows: list[dict[str, Any]] = []
    for index, row in working.iterrows():
        for line in explode_lines(row.geometry):
            if line.length <= 0:
                continue
            record = row.drop(labels=["geometry"]).to_dict()
            record["source_primary"] = source_primary
            record["source_confidence"] = float(source_confidence)
            record["source_feature_id"] = str(record.get("id", index))
            record["geometry"] = line
            rows.append(record)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=SVY21)


def build_edge_node_tables(
    paths: gpd.GeoDataFrame,
    node_precision_m: float = 0.01,
    split_intersections: bool = False,
    endpoint_snap_tolerance_m: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if paths.empty:
        raise ValueError("Cannot build graph tables from no paths")

    noded_lines = _node_intersections(paths) if split_intersections else paths
    node_ids: dict[tuple[float, float], str] = {}
    node_points: dict[str, Point] = {}
    node_buckets: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    source_counts: defaultdict[str, int] = defaultdict(int)

    for _, row in noded_lines.iterrows():
        start, end = line_endpoints(row.geometry)
        u = _node_id(start, node_ids, node_points, node_precision_m, endpoint_snap_tolerance_m, node_buckets)
        v = _node_id(end, node_ids, node_points, node_precision_m, endpoint_snap_tolerance_m, node_buckets)
        if u == v:
            continue
        geometry = _line_with_snapped_endpoints(row.geometry, node_points[u], node_points[v])
        source = row["source_primary"]
        source_counts[source] += 1
        edge_id = f"{source}_{source_counts[source]:06d}"
        edge_rows.append(
            {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "length_m": float(geometry.length),
                "ascent_fwd_m": float(row.get("ascent_fwd_m", 0.0) or 0.0),
                "descent_fwd_m": float(row.get("descent_fwd_m", 0.0) or 0.0),
                "ascent_rev_m": float(row.get("ascent_rev_m", 0.0) or 0.0),
                "descent_rev_m": float(row.get("descent_rev_m", 0.0) or 0.0),
                "source_primary": source,
                "source_confidence": float(row.get("source_confidence", 0.5) or 0.5),
                "source_feature_id": row.get("source_feature_id"),
                "highway": row.get("highway"),
                "name": row.get("name") or row.get("TRAIL_NAME") or row.get("PARK") or row.get("PCN_LOOP"),
                "trail_type": row.get("TRAIL_TYPE") or row.get("TYPE"),
                "geometry": geometry,
            }
        )

    node_rows = [{"node_id": node_id, "geometry": point} for node_id, point in sorted(node_points.items())]
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=paths.crs)
    edges = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=paths.crs)
    return nodes, edges


def _node_intersections(paths: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # GEOS unary_union nodes linework at crossings. We then assign source metadata
    # back by nearest original line so route constraints keep public-source context.
    merged = unary_union(list(paths.geometry))
    segments = explode_lines(merged)
    rows: list[dict[str, Any]] = []
    spatial_index = paths.sindex
    for segment in segments:
        candidates_idx = list(spatial_index.query(segment.buffer(0.05), predicate="intersects"))
        candidates = paths.iloc[candidates_idx] if candidates_idx else paths
        distances = candidates.geometry.distance(segment)
        best = candidates.iloc[int(distances.to_numpy().argmin())]
        rows.append(
            {
                "source_primary": best["source_primary"],
                "source_confidence": best["source_confidence"],
                "source_feature_id": best.get("source_feature_id"),
                "geometry": segment,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=paths.crs)


def _node_id(
    coord: tuple[float, float],
    node_ids: dict[tuple[float, float], str],
    node_points: dict[str, Point],
    precision_m: float,
    snap_tolerance_m: float = 0.0,
    node_buckets: defaultdict[tuple[int, int], list[str]] | None = None,
) -> str:
    key = (round(coord[0] / precision_m) * precision_m, round(coord[1] / precision_m) * precision_m)
    point = Point(key)
    if snap_tolerance_m > 0 and node_buckets is not None:
        nearest = _nearest_snap_node(point, node_points, node_buckets, snap_tolerance_m)
        if nearest is not None:
            return nearest
    if key not in node_ids:
        node_id = f"n{len(node_ids) + 1:06d}"
        node_ids[key] = node_id
        node_points[node_id] = point
        if snap_tolerance_m > 0 and node_buckets is not None:
            node_buckets[_snap_bucket(point, snap_tolerance_m)].append(node_id)
    return node_ids[key]


def _nearest_snap_node(
    point: Point,
    node_points: dict[str, Point],
    node_buckets: defaultdict[tuple[int, int], list[str]],
    tolerance_m: float,
) -> str | None:
    bx, by = _snap_bucket(point, tolerance_m)
    best_id: str | None = None
    best_distance = tolerance_m
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for node_id in node_buckets.get((bx + dx, by + dy), []):
                distance = point.distance(node_points[node_id])
                if distance <= best_distance:
                    best_distance = distance
                    best_id = node_id
    return best_id


def _snap_bucket(point: Point, tolerance_m: float) -> tuple[int, int]:
    return (int(point.x // tolerance_m), int(point.y // tolerance_m))


def _line_with_snapped_endpoints(line: LineString, start: Point, end: Point) -> LineString:
    coords = list(line.coords)
    coords[0] = (start.x, start.y)
    coords[-1] = (end.x, end.y)
    return LineString(coords)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "walk", "allowed"}


def _find_downloaded_vector(raw_dir: Path, name: str) -> Path | None:
    for path in sorted(raw_dir.glob(f"{name}.*")):
        if path.suffix.lower() not in {".json", ".txt"}:
            return path
    return None
