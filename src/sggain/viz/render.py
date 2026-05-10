from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
import pandas as pd

from sggain.config import AppConfig
from sggain.routing.search import RouteResult
from sggain.viz.data import build_route_samples, detect_steep_sections
from sggain.viz.map_data import route_summary_records, routes_to_feature_collection


def render_visualization(routes: list[RouteResult], graph, output_dir: str | Path, cfg: AppConfig | None = None) -> None:
    out = Path(output_dir)
    assets = out / "assets"
    samples_dir = assets / "route_samples"
    detail_dir = out / "routes"
    samples_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_visualization_outputs(samples_dir, detail_dir)

    threshold = 12.0 if cfg is None else float(cfg.get("visualization.steep_grade_threshold_pct", 12))
    tile_url = (
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        if cfg is None
        else cfg.get("visualization.tile_url", "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png")
    )
    tile_attribution = (
        "&copy; OpenStreetMap contributors &copy; CARTO"
        if cfg is None
        else cfg.get("visualization.tile_attribution", "&copy; OpenStreetMap contributors &copy; CARTO")
    )
    tile_max_zoom = 20 if cfg is None else int(cfg.get("visualization.tile_max_zoom", 20))
    max_routes_shown_default = 100 if cfg is None else int(cfg.get("visualization.max_routes_shown_default", 100))
    tile_options = {"tile_url": tile_url, "tile_attribution": tile_attribution, "tile_max_zoom": tile_max_zoom}
    route_samples: dict[str, list[dict[str, Any]]] = {}
    steep_sections: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        samples = build_route_samples(route, graph, spacing_m=10)
        sample_records = _json_ready(samples.to_dict(orient="records"))
        route_samples[route.route_id] = sample_records
        steep_sections[route.route_id] = _json_ready(detect_steep_sections(samples, threshold))
        (samples_dir / f"{route.route_id}.json").write_text(_dump_json(sample_records))

    summaries = _json_ready(route_summary_records(routes, route_samples))
    summaries_by_id = {item["route_id"]: item for item in summaries}
    feature_collection = _json_ready(routes_to_feature_collection(routes, graph, route_samples))
    (assets / "route_summary.json").write_text(_dump_json(summaries))
    (assets / "routes.geojson").write_text(_dump_json(feature_collection))

    env = _environment()
    index_template = env.get_template("index.html.j2")
    detail_template = env.get_template("route_detail.html.j2")
    (out / "index.html").write_text(
        index_template.render(
            routes=summaries,
            route_geojson=feature_collection,
            route_samples=route_samples,
            steep_sections=steep_sections,
            title="Singapore Gain Routes",
            max_routes_shown_default=max_routes_shown_default,
            **tile_options,
        )
    )
    for route in routes:
        (detail_dir / f"{route.route_id}.html").write_text(
            detail_template.render(
                route=summaries_by_id[route.route_id],
                samples=route_samples[route.route_id],
                steep_sections=steep_sections[route.route_id],
                title=summaries_by_id[route.route_id]["display_name"],
                **tile_options,
            )
        )


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("sggain.viz", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _remove_stale_visualization_outputs(samples_dir: Path, detail_dir: Path) -> None:
    for path in samples_dir.glob("*.json"):
        if path.is_file():
            path.unlink()
    for path in detail_dir.glob("*.html"):
        if path.is_file():
            path.unlink()


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_ready(value.item())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
