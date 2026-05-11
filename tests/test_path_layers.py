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


def test_normalize_path_layer_preserves_raw_feature_lookup():
    raw = gpd.GeoDataFrame(
        {"OBJECTID": [57452], "TRAIL_NAME": ["Summit Step"], "TRAIL_TYPE": [1]},
        geometry=[LineString([(103.7773, 1.3535), (103.7766, 1.3547)])],
        crs="EPSG:4326",
    )

    paths = normalize_path_layer(raw, "central_nature_reserve", 0.95)

    assert paths.iloc[0]["source_feature_id"] == "57452"
    assert paths.iloc[0]["source_row_index"] == "0"
    assert '"TRAIL_NAME": "Summit Step"' in paths.iloc[0]["source_raw_properties_json"]
    assert '"OBJECTID": 57452' in paths.iloc[0]["source_raw_properties_json"]


def test_build_edge_node_tables_uses_named_fallbacks_when_name_is_nan():
    raw = gpd.GeoDataFrame(
        {"name": [float("nan")], "TRAIL_NAME": ["Summit Step"], "TRAIL_TYPE": [1]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs="EPSG:3414",
    )
    paths = normalize_path_layer(raw, "central_nature_reserve", 0.95)

    _nodes, edges = build_edge_node_tables(paths)

    assert edges.iloc[0]["name"] == "Summit Step"
    assert edges.iloc[0]["trail_type"] == "1"
