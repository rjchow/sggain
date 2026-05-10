import json

import geopandas as gpd
from shapely.geometry import LineString, Point

from sggain.routing.graph import build_multidigraph
from sggain.routing.outputs import write_route_outputs
from sggain.routing.search import RouteResult


def test_outputs_write_summary_geojson_edge_sequence_gpx_png_and_samples(tmp_path):
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b"]},
        geometry=[Point(0, 0), Point(100, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab"],
            "u": ["a"],
            "v": ["b"],
            "length_m": [100.0],
            "ascent_fwd_m": [10.0],
            "descent_fwd_m": [2.0],
            "ascent_rev_m": [2.0],
            "descent_rev_m": [10.0],
            "source_primary": ["fixture"],
            "source_confidence": [1.0],
        },
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)
    route = RouteResult(
        route_id="r001",
        budget_km=0.2,
        mode="point_to_point",
        rank=1,
        distance_m=100.0,
        ascent_m=10.0,
        descent_m=2.0,
        gain_density_m_per_km=100.0,
        source_confidence=1.0,
        directed_edge_ids=["ab:fwd"],
        undirected_edge_ids=["ab"],
        node_ids=["a", "b"],
        warnings=[],
    )

    written = write_route_outputs([route], graph, tmp_path)

    assert (tmp_path / "routes.csv").exists()
    assert (tmp_path / "r001.geojson").exists()
    assert (tmp_path / "r001_edges.json").exists()
    assert (tmp_path / "r001.gpx").exists()
    assert (tmp_path / "r001_profile.png").exists()
    assert (tmp_path / "r001_samples.parquet").exists()
    assert written[0]["route_id"] == "r001"

    sequence = json.loads((tmp_path / "r001_edges.json").read_text())
    assert sequence["directed_edge_ids"] == ["ab:fwd"]


def test_outputs_remove_stale_route_artifacts_before_writing(tmp_path):
    stale_files = [
        tmp_path / "old.geojson",
        tmp_path / "old_edges.json",
        tmp_path / "old.gpx",
        tmp_path / "old_profile.png",
        tmp_path / "old_samples.parquet",
        tmp_path / "routes.json",
    ]
    for path in stale_files:
        path.write_text("stale")

    write_route_outputs([], None, tmp_path)

    assert not any(path.exists() for path in stale_files)
    assert (tmp_path / "routes.csv").exists()
