from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio
import requests

from sggain.config import AppConfig
from sggain.gis.crs import WGS84
from sggain.gis.io import configure_large_geojson_reads

WALKABLE_HIGHWAYS = {
    "footway",
    "path",
    "pedestrian",
    "steps",
    "track",
}


def fetch_osm_extract(cfg: AppConfig) -> Path | None:
    if not cfg.get("sources.osm.enabled", True):
        return None
    local_path = cfg.get("sources.osm.local_path")
    if local_path:
        path = cfg.resolve_path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured OSM local_path does not exist: {path}")
        return path

    url = cfg.get("sources.osm.geofabrik_url")
    if not url:
        return None
    destination = cfg.data_dir / "raw" / "osm" / Path(url).name
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def load_osm_paths(cfg: AppConfig) -> gpd.GeoDataFrame | None:
    if not cfg.get("sources.osm.enabled", True):
        return None
    path = _configured_or_downloaded_path(cfg)
    if path is None or not path.exists():
        return None
    try:
        gdf = _read_osm_vector(path, bbox=cfg.get("sources.osm.singapore_bbox_wgs84"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read OSM extract {path}. Install GDAL with OSM/PBF support or configure a converted GeoJSON/GPKG."
        ) from exc
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return _filter_walkable_osm(gdf)


def _configured_or_downloaded_path(cfg: AppConfig) -> Path | None:
    local_path = cfg.get("sources.osm.local_path")
    if local_path:
        return cfg.resolve_path(local_path)
    url = cfg.get("sources.osm.geofabrik_url")
    if not url:
        return None
    return cfg.data_dir / "raw" / "osm" / Path(url).name


def _read_osm_vector(path: Path, bbox: list[float] | None = None) -> gpd.GeoDataFrame:
    configure_large_geojson_reads()
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json", ".gpkg", ".shp", ".parquet"}:
        return gpd.read_parquet(path) if suffix == ".parquet" else gpd.read_file(path)
    layers = pyogrio.list_layers(path)
    layer_names = [str(row[0]) for row in layers]
    layer = "lines" if "lines" in layer_names else layer_names[0]
    kwargs = {"layer": layer}
    if bbox:
        kwargs["bbox"] = tuple(float(value) for value in bbox)
    try:
        return pyogrio.read_dataframe(path, columns=["highway", "access", "foot", "name"], **kwargs)
    except Exception:
        return pyogrio.read_dataframe(path, **kwargs)


def _filter_walkable_osm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "highway" not in gdf.columns:
        return gdf.iloc[0:0].copy()
    highway = gdf["highway"].map(_first_value).astype(str).str.lower()
    mask = highway.isin(WALKABLE_HIGHWAYS)
    if "access" in gdf.columns:
        mask &= ~gdf["access"].fillna("").astype(str).str.lower().isin({"private", "no"})
    if "foot" in gdf.columns:
        foot = gdf["foot"].fillna("").astype(str).str.lower()
        mask &= ~foot.isin({"no", "private"})
    return gdf.loc[mask].copy()


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    if pd.isna(value):
        return None
    return value
