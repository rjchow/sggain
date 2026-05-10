from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import pandas as pd

from sggain.gis.geometry import explode_lines

ELEVATION_FIELDS = ("ELEVATION", "ELEV", "HEIGHT", "CONTOUR", "LEVEL", "NAME")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def inspect_folderpaths(gdf: gpd.GeoDataFrame) -> list[str]:
    if "FOLDERPATH" not in gdf.columns:
        return []
    values = gdf["FOLDERPATH"].dropna().astype(str).unique().tolist()
    return sorted(values)


def parse_contour_elevation(row: Mapping[str, Any] | pd.Series, geometry) -> float | None:
    for field in ELEVATION_FIELDS:
        value = _case_insensitive_get(row, field)
        parsed = _parse_numeric(value)
        if parsed is not None:
            return parsed

    if geometry is not None and hasattr(geometry, "has_z") and geometry.has_z:
        z_values = []
        for line in explode_lines(geometry):
            for coord in line.coords:
                if len(coord) >= 3 and float(coord[2]) != 0:
                    z_values.append(float(coord[2]))
        if z_values:
            return float(pd.Series(z_values).median())
    return None


def extract_contour_lines(gdf: gpd.GeoDataFrame, folderpath_patterns: Sequence[str]) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    working = gdf.copy()
    if "FOLDERPATH" in working.columns and folderpath_patterns:
        pattern = "|".join(re.escape(item) for item in folderpath_patterns)
        mask = working["FOLDERPATH"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
        working = working.loc[mask].copy()

    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        elevation = parse_contour_elevation(row, row.geometry)
        if elevation is None:
            continue
        for line in explode_lines(row.geometry):
            record = row.drop(labels=["geometry"]).to_dict()
            record["elevation_m"] = float(elevation)
            record["geometry"] = line
            rows.append(record)

    if not rows:
        return gpd.GeoDataFrame(columns=list(working.columns) + ["elevation_m"], geometry="geometry", crs=working.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=working.crs)


def _case_insensitive_get(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if key in row:
        return row[key]
    lower_key = key.lower()
    for existing_key in row.keys():
        if str(existing_key).lower() == lower_key:
            return row[existing_key]
    return None


def _parse_numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)) and float(value) != 0:
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    return float(match.group(0))
