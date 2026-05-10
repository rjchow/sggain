from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pyogrio

from sggain.gis.crs import WGS84


def configure_large_geojson_reads() -> None:
    pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": "0"})


def read_vector(path: str | Path, default_crs: str = WGS84) -> gpd.GeoDataFrame:
    configure_large_geojson_reads()
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)
    return gdf


def write_geojson(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs(WGS84).to_file(output, driver="GeoJSON")
