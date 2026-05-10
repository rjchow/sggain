import geopandas as gpd
from shapely.geometry import LineString

from sggain.gis.path_layers import build_edge_node_tables, normalize_path_layer


def test_build_edge_node_tables_defaults_to_endpoint_noding_for_live_scale_data():
    raw = gpd.GeoDataFrame(
        {"name": ["east_west", "north_south"]},
        geometry=[
            LineString([(0, 50), (100, 50)]),
            LineString([(50, 0), (50, 100)]),
        ],
        crs="EPSG:3414",
    )
    paths = normalize_path_layer(raw, "fixture", 1.0)

    nodes, edges = build_edge_node_tables(paths)

    assert len(edges) == 2
    assert len(nodes) == 4


def test_build_edge_node_tables_can_snap_nearby_endpoints_and_geometry():
    raw = gpd.GeoDataFrame(
        {"name": ["approach", "steps"]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(12, 0), (20, 0)]),
        ],
        crs="EPSG:3414",
    )
    paths = normalize_path_layer(raw, "fixture", 1.0)

    nodes, edges = build_edge_node_tables(paths, endpoint_snap_tolerance_m=3.0)

    assert len(nodes) == 3
    first = edges.loc[edges["name"] == "approach"].iloc[0]
    second = edges.loc[edges["name"] == "steps"].iloc[0]
    assert first.v == second.u
    assert list(second.geometry.coords)[0] == list(first.geometry.coords)[-1]
