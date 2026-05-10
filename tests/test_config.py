from sggain.config import load_config


def test_default_routing_config_prefers_islandwide_coverage():
    cfg = load_config()

    assert cfg.get("routing.max_routes_per_budget") >= 100
    assert cfg.get("routing.max_start_nodes") is None
    assert cfg.get("routing.max_start_nodes_per_grid") <= 4
    assert cfg.get("routing.max_routes_per_grid") == 1
    assert cfg.get("routing.route_diversity_grid_m") >= 3000
    assert cfg.get("routing.min_ascent_m") == 30


def test_default_dashboard_shows_more_than_a_token_route_sample():
    cfg = load_config()

    assert cfg.get("visualization.max_routes_shown_default") >= 100
