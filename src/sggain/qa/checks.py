from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sggain.routing.graph import ordered_edge_data
from sggain.routing.search import RouteResult
from sggain.viz.data import route_linestring


def check_no_repeated_undirected_edges(routes: list[RouteResult]) -> dict[str, Any]:
    offenders = [route.route_id for route in routes if len(route.undirected_edge_ids) != len(set(route.undirected_edge_ids))]
    return _result("no repeated undirected edges", not offenders, f"{len(offenders)} offending routes")


def check_budget_compliance(routes: list[RouteResult]) -> dict[str, Any]:
    offenders = [route.route_id for route in routes if route.distance_m > route.budget_km * 1000 + 1e-6]
    return _result("route distance within budget", not offenders, f"{len(offenders)} offending routes")


def check_route_gaps(routes: list[RouteResult], graph) -> dict[str, Any]:
    offenders = []
    for route in routes:
        before = set(route.warnings)
        line = route_linestring(route, graph)
        new_gap_warnings = [warning for warning in route.warnings if warning not in before and warning.startswith("gap ")]
        if line.is_empty or line.length <= 0 or new_gap_warnings:
            offenders.append(route.route_id)
    return _result("ordered route geometry has no unexplained gaps", not offenders, f"{len(offenders)} invalid geometries")


def check_artifacts(routes: list[RouteResult], outputs_dir: Path) -> dict[str, Any]:
    missing = []
    for route in routes:
        for suffix in [".geojson", "_edges.json", ".gpx", "_profile.png", "_samples.parquet"]:
            path = outputs_dir / "routes" / f"{route.route_id}{suffix}"
            if not path.exists():
                missing.append(str(path))
        sample_json = outputs_dir / "viz" / "assets" / "route_samples" / f"{route.route_id}.json"
        detail = outputs_dir / "viz" / "routes" / f"{route.route_id}.html"
        if not sample_json.exists():
            missing.append(str(sample_json))
        if not detail.exists():
            missing.append(str(detail))
    for path in [
        outputs_dir / "routes" / "routes.csv",
        outputs_dir / "viz" / "index.html",
        outputs_dir / "viz" / "assets" / "route_summary.json",
        outputs_dir / "viz" / "assets" / "routes.geojson",
    ]:
        if not path.exists():
            missing.append(str(path))
    return _result("route artifacts complete", not missing, f"{len(missing)} missing artifacts")


def check_elevation_plausible(outputs_dir: Path, min_m: float = -20, max_m: float = 200) -> dict[str, Any]:
    offenders = []
    for path in (outputs_dir / "routes").glob("*_samples.parquet"):
        samples = pd.read_parquet(path)
        if not samples["elevation_smooth_m"].between(min_m, max_m).all():
            offenders.append(path.name)
    return _result("sampled elevations plausible", not offenders, f"{len(offenders)} sample files outside range")


def check_nearest_fallback_reported(graph) -> dict[str, Any]:
    total = 0
    seen = False
    for _, _, data in graph.edges(data=True):
        if "elevation_fallback_count" in data:
            seen = True
            total += int(data["elevation_fallback_count"] or 0)
    message = f"fallback count reported: {total}" if seen else "no elevation fallback metadata on graph edges"
    return _result("nearest-neighbour interpolation fallback reported", seen, message)


def check_html_rendered(outputs_dir: Path) -> dict[str, Any]:
    index = outputs_dir / "viz" / "index.html"
    details = list((outputs_dir / "viz" / "routes").glob("*.html"))
    return _result("index.html and detail pages render", index.exists() and bool(details), f"{len(details)} detail pages")


def _result(name: str, ok: bool, message: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "message": message}
