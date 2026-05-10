from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project_root": ".",
    "data_dir": "data",
    "outputs_dir": "outputs",
    "budgets_km": [5, 10, 15, 20],
    "modes": ["loop", "point_to_point"],
    "path_processing": {
        "split_intersections": False,
        "node_precision_m": 0.01,
        "endpoint_snap_tolerance_m": 5.0,
    },
    "contours": {
        "folderpath_patterns": ["contour", "height"],
        "densify_spacing_m": 100,
        "max_interpolation_points": 50000,
    },
    "elevation": {
        "edge_sample_spacing_m": 10,
        "smooth_window_m": 30,
        "plausible_min_m": -20,
        "plausible_max_m": 200,
    },
    "routing": {
        "beam_width": 400,
        "max_routes_per_budget": 100,
        "max_start_nodes": None,
        "start_node_grid_m": 1500,
        "max_start_nodes_per_grid": 4,
        "route_diversity_grid_m": 3000,
        "max_routes_per_grid": 1,
        "frontier_grid_m": 3000,
        "max_frontier_states_per_grid": 3,
        "jaccard_similarity_threshold": 0.85,
        "length_jaccard_similarity_threshold": 0.78,
        "length_containment_similarity_threshold": 0.88,
        "min_ascent_m": 30,
        "landmark_path_name_patterns": ["Summit Path", "South View Path", "Cave Path"],
    },
    "visualization": {
        "steep_grade_threshold_pct": 12,
        "sample_spacing_m": 10,
        "max_routes_shown_default": 100,
        "tile_url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "tile_attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
        "tile_max_zoom": 20,
    },
    "sources": {
        "data_gov": {
            "nparks_tracks": "d_306cc1018cb733346681883ee6d73054",
            "central_nature_reserve": "d_84e6db6c43ffb8b17803334ec2b5c95d",
            "park_connector_loop": "d_a69ef89737379f231d2ae93fd1c5707f",
            "national_map_line": "d_10480c0b59e65663dfae1028ff4aa8bb",
        },
        "osm": {
            "enabled": True,
            "singapore_bbox_wgs84": [103.55, 1.15, 104.12, 1.55],
            "geofabrik_url": "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf",
            "local_path": None,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class AppConfig:
    values: dict[str, Any]
    config_path: Path | None = None

    @property
    def root(self) -> Path:
        base = self.config_path.parent if self.config_path else Path.cwd()
        root_value = Path(self.values.get("project_root", "."))
        return root_value if root_value.is_absolute() else (base / root_value).resolve()

    @property
    def data_dir(self) -> Path:
        return self.resolve_path(self.values["data_dir"])

    @property
    def outputs_dir(self) -> Path:
        return self.resolve_path(self.values["outputs_dir"])

    def resolve_path(self, value: str | Path | None) -> Path:
        if value is None:
            raise ValueError("Cannot resolve a null path")
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir / "raw" / "data_gov",
            self.data_dir / "raw" / "osm",
            self.data_dir / "cache",
            self.data_dir / "processed" / "paths",
            self.data_dir / "processed" / "elevation",
            self.data_dir / "processed" / "graph",
            self.data_dir / "fixtures",
            self.outputs_dir / "routes",
            self.outputs_dir / "maps",
            self.outputs_dir / "profiles",
            self.outputs_dir / "viz" / "assets" / "route_samples",
            self.outputs_dir / "viz" / "routes",
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).resolve() if path else None
    overrides: dict[str, Any] = {}
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        overrides = yaml.safe_load(config_path.read_text()) or {}
    values = _deep_merge(DEFAULT_CONFIG, overrides)
    return AppConfig(values=values, config_path=config_path)
