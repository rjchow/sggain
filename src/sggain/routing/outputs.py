from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import gpxpy.gpx
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sggain.gis.crs import WGS84
from sggain.routing.search import RouteResult
from sggain.viz.data import build_route_samples, route_linestring


def write_route_outputs(routes: list[RouteResult], graph, output_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _remove_stale_route_outputs(out)
    summaries = [route.to_dict() for route in routes]
    pd.DataFrame(summaries).to_csv(out / "routes.csv", index=False)
    for route in routes:
        samples = build_route_samples(route, graph)
        samples.to_parquet(out / f"{route.route_id}_samples.parquet", index=False)
        _write_route_geojson(route, graph, out / f"{route.route_id}.geojson")
        _write_edge_sequence(route, out / f"{route.route_id}_edges.json")
        _write_gpx(route, samples, out / f"{route.route_id}.gpx")
        _write_profile_png(route, samples, out / f"{route.route_id}_profile.png")
    return summaries


def routes_to_json(routes: list[RouteResult], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([route.to_dict() for route in routes], indent=2))


def routes_from_json(path: str | Path) -> list[RouteResult]:
    return [RouteResult.from_dict(item) for item in json.loads(Path(path).read_text())]


def _write_route_geojson(route: RouteResult, graph, path: Path) -> None:
    line = route_linestring(route, graph, to_wgs84=True)
    properties = route.to_dict()
    for key in ["directed_edge_ids", "undirected_edge_ids", "node_ids", "warnings"]:
        properties.pop(key, None)
    gdf = gpd.GeoDataFrame([properties], geometry=[line], crs=WGS84)
    gdf.to_file(path, driver="GeoJSON")


def _write_edge_sequence(route: RouteResult, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "route_id": route.route_id,
                "directed_edge_ids": route.directed_edge_ids,
                "undirected_edge_ids": route.undirected_edge_ids,
                "node_ids": route.node_ids,
            },
            indent=2,
        )
    )


def _write_gpx(route: RouteResult, samples: pd.DataFrame, path: Path) -> None:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=route.route_id)
    segment = gpxpy.gpx.GPXTrackSegment()
    for row in samples.itertuples(index=False):
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=float(row.lat),
                longitude=float(row.lon),
                elevation=float(row.elevation_smooth_m),
            )
        )
    track.segments.append(segment)
    gpx.tracks.append(track)
    path.write_text(gpx.to_xml())


def _write_profile_png(route: RouteResult, samples: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(samples["cum_distance_km"], samples["elevation_smooth_m"], color="#265c42", linewidth=2)
    ax.set_title(route.route_id)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _remove_stale_route_outputs(out: Path) -> None:
    for pattern in [
        "*.geojson",
        "*_edges.json",
        "*.gpx",
        "*_profile.png",
        "*_samples.parquet",
        "routes.csv",
        "routes.json",
    ]:
        for path in out.glob(pattern):
            if path.is_file():
                path.unlink()
