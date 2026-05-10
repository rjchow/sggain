import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point

from sggain.routing.graph import build_multidigraph


def test_graph_adds_two_oriented_directed_edges_per_segment():
    nodes = gpd.GeoDataFrame(
        {"node_id": ["a", "b"]},
        geometry=[Point(0, 0), Point(100, 0)],
        crs="EPSG:3414",
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["e1"],
            "u": ["a"],
            "v": ["b"],
            "length_m": [100.0],
            "ascent_fwd_m": [12.0],
            "descent_fwd_m": [3.0],
            "ascent_rev_m": [3.0],
            "descent_rev_m": [12.0],
            "source_primary": ["fixture"],
            "source_confidence": [0.9],
            "highway": ["steps"],
            "name": ["Summit Path"],
        },
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3414",
    )

    graph = build_multidigraph(nodes, edges)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_edges() == 2

    fwd = graph.get_edge_data("a", "b")[0]
    rev = graph.get_edge_data("b", "a")[0]

    assert fwd["ascent_m"] == 12.0
    assert rev["ascent_m"] == 3.0
    assert list(rev["geometry"].coords) == [(100.0, 0.0), (0.0, 0.0)]
    assert fwd["undirected_edge_id"] == rev["undirected_edge_id"] == "e1"
    assert fwd["highway"] == "steps"
    assert fwd["name"] == "Summit Path"
