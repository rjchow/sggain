from __future__ import annotations

from sggain.config import AppConfig
from sggain.qa.checks import (
    check_artifacts,
    check_budget_compliance,
    check_elevation_plausible,
    check_html_rendered,
    check_nearest_fallback_reported,
    check_no_repeated_undirected_edges,
    check_route_gaps,
)
from sggain.routing.search import RouteResult


def run_diagnostics(routes: list[RouteResult], graph, cfg: AppConfig) -> list[dict]:
    return [
        check_no_repeated_undirected_edges(routes),
        check_budget_compliance(routes),
        check_route_gaps(routes, graph),
        check_artifacts(routes, cfg.outputs_dir),
        check_elevation_plausible(
            cfg.outputs_dir,
            min_m=cfg.get("elevation.plausible_min_m", -20),
            max_m=cfg.get("elevation.plausible_max_m", 200),
        ),
        check_nearest_fallback_reported(graph),
        check_html_rendered(cfg.outputs_dir),
    ]
