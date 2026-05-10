import geopandas as gpd
from shapely.geometry import LineString

from sggain.sources.osm import _filter_walkable_osm


def test_osm_filter_keeps_pedestrian_paths_and_excludes_ordinary_roads():
    gdf = gpd.GeoDataFrame(
        {"highway": ["footway", "path", "residential", "primary"], "access": [None, None, None, None]},
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(0, 2), (1, 2)]),
            LineString([(0, 3), (1, 3)]),
        ],
        crs="EPSG:4326",
    )

    filtered = _filter_walkable_osm(gdf)

    assert filtered["highway"].tolist() == ["footway", "path"]
