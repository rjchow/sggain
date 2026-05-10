import geopandas as gpd
from shapely.geometry import LineString, Point

from sggain.routing.graph import build_multidigraph
from sggain.routing.search import search_routes


def _fixture_graph():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d"]},
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0), Point(100, 100)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "bc", "bd", "dc", "ca"],
            "u": ["a", "b", "b", "d", "c"],
            "v": ["b", "c", "d", "c", "a"],
            "length_m": [100.0, 100.0, 100.0, 141.0, 200.0],
            "ascent_fwd_m": [20.0, 5.0, 30.0, 25.0, 1.0],
            "descent_fwd_m": [1.0, 2.0, 1.0, 3.0, 5.0],
            "ascent_rev_m": [1.0, 2.0, 1.0, 3.0, 5.0],
            "descent_rev_m": [20.0, 5.0, 30.0, 25.0, 1.0],
            "source_primary": ["fixture"] * 5,
            "source_confidence": [1.0] * 5,
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(100, 0), (100, 100)]),
            LineString([(100, 100), (200, 0)]),
            LineString([(200, 0), (0, 0)]),
        ],
        crs="EPSG:3414",
    )
    return build_multidigraph(nodes, edges)


def test_search_respects_budget_and_never_repeats_undirected_edges():
    routes = search_routes(
        _fixture_graph(),
        budgets_km=[0.35],
        modes=["point_to_point"],
        beam_width=20,
        max_routes_per_budget=5,
    )

    assert routes
    assert all(route.distance_m <= 350.0 for route in routes)
    assert all(len(route.undirected_edge_ids) == len(set(route.undirected_edge_ids)) for route in routes)
    assert routes == sorted(
        routes,
        key=lambda r: (-r.ascent_m, -r.gain_density_m_per_km, -r.source_confidence),
    )


def test_search_can_return_loop_routes():
    routes = search_routes(
        _fixture_graph(),
        budgets_km=[0.5],
        modes=["loop"],
        beam_width=40,
        max_routes_per_budget=3,
    )

    assert routes
    assert all(route.node_ids[0] == route.node_ids[-1] for route in routes)


def test_search_can_limit_start_nodes_by_outgoing_ascent_score():
    routes = search_routes(
        _fixture_graph(),
        budgets_km=[0.25],
        modes=["point_to_point"],
        beam_width=10,
        max_routes_per_budget=3,
        max_start_nodes=1,
    )

    assert routes
    assert all(route.node_ids[0] == "b" for route in routes)


def test_search_deduplicates_prefix_routes_by_length_containment():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d"]},
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0), Point(300, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "bc", "cd"],
            "u": ["a", "b", "c"],
            "v": ["b", "c", "d"],
            "length_m": [100.0, 100.0, 100.0],
            "ascent_fwd_m": [30.0, 10.0, 10.0],
            "descent_fwd_m": [0.0, 0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0, 0.0],
            "descent_rev_m": [10.0, 10.0, 10.0],
            "source_primary": ["fixture"] * 3,
            "source_confidence": [1.0] * 3,
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (300, 0)]),
        ],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.35],
        modes=["point_to_point"],
        beam_width=10,
        max_routes_per_budget=5,
        max_start_nodes=1,
        length_containment_similarity_threshold=0.95,
    )

    assert len(routes) == 1
    assert routes[0].undirected_edge_ids == ["ab", "bc", "cd"]


def test_search_route_grid_cap_surfaces_lower_gain_geographic_areas():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d", "x", "y"]},
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(200, 0),
            Point(300, 0),
            Point(5000, 0),
            Point(5100, 0),
        ],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "bc", "cd", "xy"],
            "u": ["a", "b", "c", "x"],
            "v": ["b", "c", "d", "y"],
            "length_m": [100.0, 100.0, 100.0, 100.0],
            "ascent_fwd_m": [50.0, 45.0, 40.0, 5.0],
            "descent_fwd_m": [0.0, 0.0, 0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0, 0.0, 0.0],
            "descent_rev_m": [50.0, 45.0, 40.0, 5.0],
            "source_primary": ["fixture"] * 4,
            "source_confidence": [1.0] * 4,
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (300, 0)]),
            LineString([(5000, 0), (5100, 0)]),
        ],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.2],
        modes=["point_to_point"],
        beam_width=20,
        max_routes_per_budget=4,
        max_start_nodes=None,
        route_diversity_grid_m=1000,
        max_routes_per_grid=1,
    )

    assert any(route.node_ids[0] == "x" for route in routes)


def test_search_includes_configured_landmark_path_even_when_start_node_is_not_top_scoring():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "x", "y"]},
        geometry=[Point(0, 0), Point(100, 0), Point(5000, 0), Point(5100, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "xy"],
            "u": ["a", "x"],
            "v": ["b", "y"],
            "length_m": [100.0, 100.0],
            "ascent_fwd_m": [50.0, 5.0],
            "descent_fwd_m": [0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0],
            "descent_rev_m": [50.0, 5.0],
            "source_primary": ["fixture", "fixture"],
            "source_confidence": [1.0, 1.0],
            "name": ["High Gain Path", "Summit Path"],
        },
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(5000, 0), (5100, 0)])],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.2],
        modes=["point_to_point"],
        beam_width=10,
        max_routes_per_budget=2,
        max_start_nodes=1,
        landmark_path_name_patterns=["Summit Path"],
    )

    assert any("xy" in route.undirected_edge_ids for route in routes)


def test_landmark_path_selection_prefers_steps_segments():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "x", "y"]},
        geometry=[Point(0, 0), Point(100, 0), Point(5000, 0), Point(5100, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["footway_summit", "steps_summit"],
            "u": ["a", "x"],
            "v": ["b", "y"],
            "length_m": [100.0, 100.0],
            "ascent_fwd_m": [20.0, 5.0],
            "descent_fwd_m": [0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0],
            "descent_rev_m": [20.0, 5.0],
            "source_primary": ["fixture", "fixture"],
            "source_confidence": [1.0, 1.0],
            "name": ["Summit Path", "Summit Path"],
            "highway": ["footway", "steps"],
        },
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(5000, 0), (5100, 0)])],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.2],
        modes=["point_to_point"],
        beam_width=10,
        max_routes_per_budget=1,
        max_start_nodes=None,
        landmark_path_name_patterns=["Summit Path"],
    )

    assert "steps_summit" in routes[0].undirected_edge_ids


def test_landmark_path_is_not_pruned_before_it_can_extend():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["h1", "h2", "h3", "h4", "s1", "s2", "s3", "s4"]},
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(200, 0),
            Point(300, 0),
            Point(5000, 0),
            Point(5100, 0),
            Point(5200, 0),
            Point(5300, 0),
        ],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["h12", "h23", "h34", "summit_steps", "summit_footway", "summit_tail"],
            "u": ["h1", "h2", "h3", "s1", "s2", "s3"],
            "v": ["h2", "h3", "h4", "s2", "s3", "s4"],
            "length_m": [100.0] * 6,
            "ascent_fwd_m": [60.0, 55.0, 50.0, 6.0, 20.0, 20.0],
            "descent_fwd_m": [0.0] * 6,
            "ascent_rev_m": [0.0] * 6,
            "descent_rev_m": [60.0, 55.0, 50.0, 6.0, 20.0, 20.0],
            "source_primary": ["fixture"] * 6,
            "source_confidence": [1.0] * 6,
            "name": ["High Gain"] * 3 + ["Summit Path", "Summit Path", "Summit Connector"],
            "highway": ["footway", "footway", "footway", "steps", "footway", "footway"],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (300, 0)]),
            LineString([(5000, 0), (5100, 0)]),
            LineString([(5100, 0), (5200, 0)]),
            LineString([(5200, 0), (5300, 0)]),
        ],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.35],
        modes=["point_to_point"],
        beam_width=2,
        max_routes_per_budget=2,
        max_start_nodes=1,
        landmark_path_name_patterns=["Summit Path"],
    )

    landmark_routes = [route for route in routes if "summit_steps" in route.undirected_edge_ids]
    assert landmark_routes
    assert len(landmark_routes[0].undirected_edge_ids) > 1


def test_frontier_grid_cap_keeps_lower_gain_areas_alive_during_expansion():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d", "x", "y", "z"]},
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(200, 0),
            Point(300, 0),
            Point(5000, 0),
            Point(5100, 0),
            Point(5200, 0),
        ],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "bc", "cd", "xy", "yz"],
            "u": ["a", "b", "c", "x", "y"],
            "v": ["b", "c", "d", "y", "z"],
            "length_m": [100.0] * 5,
            "ascent_fwd_m": [50.0, 45.0, 40.0, 30.0, 5.0],
            "descent_fwd_m": [0.0] * 5,
            "ascent_rev_m": [0.0] * 5,
            "descent_rev_m": [50.0, 45.0, 40.0, 30.0, 5.0],
            "source_primary": ["fixture"] * 5,
            "source_confidence": [1.0] * 5,
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (300, 0)]),
            LineString([(5000, 0), (5100, 0)]),
            LineString([(5100, 0), (5200, 0)]),
        ],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.25],
        modes=["point_to_point"],
        beam_width=2,
        max_routes_per_budget=2,
        max_start_nodes=None,
        route_diversity_grid_m=1000,
        max_routes_per_grid=1,
        frontier_grid_m=1000,
        max_frontier_states_per_grid=1,
    )

    assert any(route.undirected_edge_ids == ["xy", "yz"] for route in routes)


def test_search_does_not_repeat_exact_route_families_across_budgets():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d", "e", "f", "g"]},
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(1000, 0),
            Point(1100, 0),
            Point(2000, 0),
            Point(2100, 0),
            Point(2200, 0),
        ],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "cd", "ef", "fg"],
            "u": ["a", "c", "e", "f"],
            "v": ["b", "d", "f", "g"],
            "length_m": [100.0, 100.0, 100.0, 100.0],
            "ascent_fwd_m": [50.0, 40.0, 35.0, 35.0],
            "descent_fwd_m": [0.0, 0.0, 0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0, 0.0, 0.0],
            "descent_rev_m": [50.0, 40.0, 35.0, 35.0],
            "source_primary": ["fixture"] * 4,
            "source_confidence": [1.0] * 4,
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(1000, 0), (1100, 0)]),
            LineString([(2000, 0), (2100, 0)]),
            LineString([(2100, 0), (2200, 0)]),
        ],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.15, 0.25],
        modes=["point_to_point"],
        beam_width=20,
        max_routes_per_budget=2,
        max_start_nodes=None,
        route_diversity_grid_m=None,
    )

    signatures = [(route.mode, tuple(route.undirected_edge_ids)) for route in routes]
    assert len(signatures) == len(set(signatures))
    assert any(route.budget_km == 0.25 and route.undirected_edge_ids == ["ef", "fg"] for route in routes)


def test_search_filters_routes_below_minimum_ascent():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b", "c", "d"]},
        geometry=[Point(0, 0), Point(100, 0), Point(1000, 0), Point(1100, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["ab", "cd"],
            "u": ["a", "c"],
            "v": ["b", "d"],
            "length_m": [100.0, 100.0],
            "ascent_fwd_m": [29.0, 31.0],
            "descent_fwd_m": [0.0, 0.0],
            "ascent_rev_m": [0.0, 0.0],
            "descent_rev_m": [29.0, 31.0],
            "source_primary": ["fixture", "fixture"],
            "source_confidence": [1.0, 1.0],
        },
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(1000, 0), (1100, 0)])],
        crs="EPSG:3414",
    )
    graph = build_multidigraph(nodes, edges)

    routes = search_routes(
        graph,
        budgets_km=[0.2],
        modes=["point_to_point"],
        beam_width=10,
        max_routes_per_budget=5,
        max_start_nodes=None,
        route_diversity_grid_m=None,
        min_ascent_m=30.0,
    )

    assert routes
    assert all(route.ascent_m >= 30.0 for route in routes)
    assert all("ab" not in route.undirected_edge_ids for route in routes)
