import json

import geopandas as gpd
from shapely.geometry import LineString, Point

from sggain.routing.graph import build_multidigraph
from sggain.routing.search import RouteResult
from sggain.viz.data import build_route_samples, detect_steep_sections
from sggain.viz.map_data import route_display_name, routes_to_feature_collection
from sggain.viz.render import render_visualization


def _route_and_graph():
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
            "ascent_fwd_m": [20.0],
            "descent_fwd_m": [0.0],
            "ascent_rev_m": [0.0],
            "descent_rev_m": [20.0],
            "source_primary": ["fixture"],
            "source_confidence": [1.0],
            "source_feature_id": ["fixture-row-7"],
            "highway": [float("nan")],
            "name": ["Summit Path"],
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
        ascent_m=20.0,
        descent_m=0.0,
        gain_density_m_per_km=200.0,
        source_confidence=1.0,
        directed_edge_ids=["ab:fwd"],
        undirected_edge_ids=["ab"],
        node_ids=["a", "b"],
        warnings=[],
    )
    return route, graph


def test_route_samples_and_geojson_are_ordered():
    route, graph = _route_and_graph()

    samples = build_route_samples(route, graph, spacing_m=25)
    collection = routes_to_feature_collection([route], graph)

    assert samples["route_id"].eq("r001").all()
    assert samples["cum_distance_m"].is_monotonic_increasing
    assert samples.iloc[-1]["cum_distance_m"] == 100.0
    assert "highway" in samples.columns
    assert "path_name" in samples.columns
    assert collection["features"][0]["geometry"]["type"] == "LineString"


def test_steep_sections_and_html_rendering(tmp_path):
    route, graph = _route_and_graph()
    samples = build_route_samples(route, graph, spacing_m=10)
    steep = detect_steep_sections(samples, threshold_pct=12)
    route_artifact_dir = tmp_path.parent / "routes"
    route_artifact_dir.mkdir()
    (route_artifact_dir / "r001.gpx").write_text("<gpx></gpx>")
    (route_artifact_dir / "r001.geojson").write_text("{}")

    render_visualization([route], graph, tmp_path)

    assert steep
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "routes" / "r001.html").exists()
    assert (tmp_path / "assets" / "route_summary.json").exists()
    assert (tmp_path / "assets" / "routes.geojson").exists()
    assert (tmp_path / "assets" / "route_samples" / "r001.json").exists()
    assert (tmp_path / "routes" / "r001.gpx").exists()
    assert (tmp_path / "routes" / "r001.geojson").exists()
    assert "tile.openstreetmap.org" not in (tmp_path / "index.html").read_text()
    detail_html = (tmp_path / "routes" / "r001.html").read_text()
    assert "tile.openstreetmap.org" not in detail_html
    assert 'href="r001.gpx"' in detail_html
    assert 'href="r001.geojson"' in detail_html
    assert "../../routes/" not in detail_html
    assert "Data Attribution" in detail_html
    assert "Fixture" in detail_html
    assert "100.0% of route network" in detail_html
    assert "SLA National Map Line" in detail_html
    assert "Source Records Used" in detail_html
    assert '"source_feature_id": "fixture-row-7"' in detail_html
    assert '"derived_edge_ids": [' in detail_html
    assert '"latlon_bounds": {' in detail_html
    assert '"elevation_summary": {' in detail_html
    assert '"route_sample_points": [' in detail_html
    assert '"lat":' in detail_html
    assert '"lon":' in detail_html
    assert '"elevation_smooth_m":' in detail_html
    assert "Source Diagnostics" not in detail_html
    assert "fetch('assets/" not in (tmp_path / "index.html").read_text()
    assert "ROUTE_GEOJSON" in (tmp_path / "index.html").read_text()
    assert "ROUTE_SAMPLES" in (tmp_path / "index.html").read_text()
    assert '<option value="density" selected>Gain density</option>' in (tmp_path / "index.html").read_text()
    assert 'id="maxRoutes" type="number" min="1" value="100"' in (tmp_path / "index.html").read_text()
    assert "onEachFeature" in (tmp_path / "index.html").read_text()
    assert "selectRouteById" in (tmp_path / "index.html").read_text()
    assert "display_name" in (tmp_path / "index.html").read_text()
    assert "NaN" not in (tmp_path / "index.html").read_text()
    assert "NaN" not in (tmp_path / "assets" / "route_samples" / "r001.json").read_text()

    payload = json.loads((tmp_path / "assets" / "route_samples" / "r001.json").read_text())
    assert payload[0]["route_id"] == "r001"
    assert payload[0]["source_feature_id"] == "fixture-row-7"
    summary = json.loads((tmp_path / "assets" / "route_summary.json").read_text())
    assert summary[0]["display_name"].startswith("Summit Path")
    assert "20 m gain" in summary[0]["display_name"]


def test_render_visualization_removes_stale_detail_and_sample_files(tmp_path):
    stale_detail = tmp_path / "routes" / "old.html"
    stale_sample = tmp_path / "assets" / "route_samples" / "old.json"
    stale_detail.parent.mkdir(parents=True)
    stale_sample.parent.mkdir(parents=True)
    stale_detail.write_text("stale")
    stale_sample.write_text("stale")

    render_visualization([], None, tmp_path)

    assert not stale_detail.exists()
    assert not stale_sample.exists()


def test_display_name_uses_area_when_path_name_is_missing():
    route, _graph = _route_and_graph()
    route.mode = "point_to_point"
    samples = [
        {"lat": 1.3540, "lon": 103.7770, "path_name": None},
        {"lat": 1.3550, "lon": 103.7780, "path_name": None},
    ]

    name = route_display_name(route, samples)

    assert name.startswith("Bukit Timah Nature Reserve route")
    assert "Point To Point" not in name
