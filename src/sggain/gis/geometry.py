from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from shapely.geometry import LineString, MultiLineString


def explode_lines(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 0]
    return []


def reverse_line(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def line_endpoints(line: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = list(line.coords)
    return (float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))


def densify_line(line: LineString, spacing_m: float) -> list[tuple[float, object]]:
    if line.length == 0:
        return [(0.0, line.interpolate(0))]
    distances = list(np.arange(0, line.length, spacing_m))
    if not distances or not np.isclose(distances[-1], line.length):
        distances.append(float(line.length))
    return [(float(distance), line.interpolate(float(distance))) for distance in distances]


def concatenate_lines(lines: Iterable[LineString], gap_tolerance_m: float = 2.0) -> tuple[LineString, list[str]]:
    coords: list[tuple[float, float]] = []
    warnings: list[str] = []
    previous = None
    for line in lines:
        line_coords = [(float(x), float(y)) for x, y, *_ in line.coords]
        if not line_coords:
            continue
        if previous is not None:
            gap = LineString([previous, line_coords[0]]).length
            if gap > gap_tolerance_m:
                warnings.append(f"gap {gap:.2f} m before segment {len(coords)}")
        if coords and coords[-1] == line_coords[0]:
            coords.extend(line_coords[1:])
        else:
            coords.extend(line_coords)
        previous = coords[-1]
    if len(coords) == 1:
        coords.append(coords[0])
    return LineString(coords), warnings
