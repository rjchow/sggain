import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from sggain.gis.contours import parse_contour_elevation
from sggain.gis.elevation import (
    ElevationModel,
    compute_ascent_descent,
    densify_contours_to_points,
    sample_edge_profile,
    smooth_elevations,
)


def test_parse_contour_elevation_from_field_name_and_z():
    assert parse_contour_elevation({"ELEVATION": "35m"}, None) == 35.0
    assert parse_contour_elevation({"NAME": "Contour 72.5 m"}, None) == 72.5

    z_line = LineString([(0, 0, 0), (1, 0, 12)])
    assert parse_contour_elevation({}, z_line) == 12.0


def test_interpolation_uses_nearest_fallback_and_samples_edge_profile():
    contour_lines = gpd.GeoDataFrame(
        {"elevation_m": [0.0, 100.0]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 100), (100, 100)]),
        ],
        crs="EPSG:3414",
    )
    points = densify_contours_to_points(contour_lines, spacing_m=50)
    model = ElevationModel.from_points(points)

    profile = sample_edge_profile(LineString([(10, 10), (10, 90)]), model, spacing_m=10, smooth_window_m=30)

    assert profile["fallback_count"] >= 0
    assert len(profile["distance_m"]) >= 9
    assert profile["elevation_smooth_m"][0] < profile["elevation_smooth_m"][-1]


def test_smoothing_and_ascent_descent_are_directional():
    raw = np.array([0.0, 10.0, 5.0, 25.0, 20.0])
    smoothed = smooth_elevations(raw, distances_m=np.array([0, 10, 20, 30, 40]), window_m=30)
    ascent, descent = compute_ascent_descent(smoothed)

    assert len(smoothed) == len(raw)
    assert ascent > 0
    assert descent >= 0
    assert math.isclose(
        compute_ascent_descent(smoothed[::-1])[0],
        descent,
        rel_tol=0.2,
    )
