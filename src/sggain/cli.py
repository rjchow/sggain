from __future__ import annotations

import builtins
import json
import pickle
from pathlib import Path
from typing import Optional

import geopandas as gpd
import typer
from rich.console import Console
from rich.table import Table

from sggain.config import AppConfig, load_config
from sggain.gis.contours import extract_contour_lines, inspect_folderpaths
from sggain.gis.elevation import ElevationModel, densify_contours_to_points, enrich_edges_with_elevation
from sggain.gis.io import configure_large_geojson_reads, read_vector
from sggain.gis.path_layers import build_edge_node_tables, build_paths_from_config
from sggain.routing.graph import build_multidigraph
from sggain.routing.outputs import routes_from_json, routes_to_json, write_route_outputs
from sggain.routing.search import search_routes
from sggain.sources.data_gov import fetch_all_datasets
from sggain.sources.osm import fetch_osm_extract
from sggain.viz.render import render_visualization
from sggain.qa.diagnostics import run_diagnostics

app = typer.Typer(help="Find and visualize high-elevation-gain Singapore walking routes.")
console = Console()


def _load(config_path: Optional[Path]) -> AppConfig:
    cfg = load_config(config_path)
    cfg.ensure_directories()
    return cfg


@app.command()
def fetch(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Download configured data.gov.sg datasets and optional OSM extract."""
    cfg = _load(config)
    records = fetch_all_datasets(cfg)
    osm_path = fetch_osm_extract(cfg)
    for record in records:
        console.print(f"[green]downloaded[/] {record['name']}: {record['path']}")
    if osm_path:
        console.print(f"[green]osm[/] {osm_path}")


@app.command()
def inspect(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Print National Map Line FOLDERPATH values and contour candidates."""
    cfg = _load(config)
    path = _data_gov_path(cfg, "national_map_line")
    gdf = read_vector(path)
    values = inspect_folderpaths(gdf)
    candidates = extract_contour_lines(gdf, cfg.get("contours.folderpath_patterns", ["contour", "height"]))

    table = Table(title="National Map Line FOLDERPATH values")
    table.add_column("FOLDERPATH")
    for value in values:
        table.add_row(value)
    console.print(table)
    console.print(f"Likely contour features: {len(candidates)}")


@app.command("build-paths")
def build_paths(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Normalize walkable path layers and create node/edge tables."""
    cfg = _load(config)
    paths = build_paths_from_config(cfg)
    nodes, edges = build_edge_node_tables(
        paths,
        node_precision_m=cfg.get("path_processing.node_precision_m", 0.01),
        split_intersections=cfg.get("path_processing.split_intersections", False),
        endpoint_snap_tolerance_m=cfg.get("path_processing.endpoint_snap_tolerance_m", 5.0),
    )
    out = cfg.data_dir / "processed" / "paths"
    paths.to_parquet(out / "paths.parquet")
    nodes.to_parquet(out / "nodes.parquet")
    edges.to_parquet(out / "edges.parquet")
    console.print(f"Wrote {len(nodes)} nodes and {len(edges)} path edges to {out}")


@app.command("build-elevation")
def build_elevation(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Build contour interpolation points and enrich path edges with ascent/descent."""
    cfg = _load(config)
    contours_raw = read_vector(_data_gov_path(cfg, "national_map_line"))
    contours = extract_contour_lines(contours_raw, cfg.get("contours.folderpath_patterns", ["contour", "height"]))
    if contours.empty:
        raise typer.BadParameter("No contour lines with parseable elevation were found.")
    if contours.crs is None:
        contours = contours.set_crs("EPSG:4326")
    contours = contours.to_crs("EPSG:3414")
    points = densify_contours_to_points(
        contours,
        cfg.get("contours.densify_spacing_m", 100),
        max_points=cfg.get("contours.max_interpolation_points", 50000),
    )
    model = ElevationModel.from_points(points)

    path_dir = cfg.data_dir / "processed" / "paths"
    edges = gpd.read_parquet(path_dir / "edges.parquet")
    enriched = enrich_edges_with_elevation(
        edges,
        model,
        spacing_m=cfg.get("elevation.edge_sample_spacing_m", 10),
        smooth_window_m=cfg.get("elevation.smooth_window_m", 30),
    )
    out = cfg.data_dir / "processed" / "elevation"
    contours.to_parquet(out / "contours.parquet")
    points.to_parquet(out / "elevation_points.parquet")
    enriched.to_parquet(path_dir / "edges_with_elevation.parquet")
    console.print(f"Wrote {len(points)} elevation points and enriched {len(enriched)} edges")


@app.command("build-graph")
def build_graph(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Build a NetworkX MultiDiGraph from processed nodes and edges."""
    cfg = _load(config)
    path_dir = cfg.data_dir / "processed" / "paths"
    nodes = gpd.read_parquet(path_dir / "nodes.parquet")
    edge_path = path_dir / "edges_with_elevation.parquet"
    if not edge_path.exists():
        edge_path = path_dir / "edges.parquet"
    edges = gpd.read_parquet(edge_path)
    graph = build_multidigraph(nodes, edges)
    out = cfg.data_dir / "processed" / "graph" / "graph.pkl"
    with out.open("wb") as handle:
        pickle.dump(graph, handle)
    console.print(f"Wrote graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} directed edges")


@app.command()
def search(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Search high-ascent routes under configured budgets."""
    cfg = _load(config)
    graph = _read_graph(cfg)
    routes = search_routes(
        graph,
        budgets_km=cfg.get("budgets_km", [5, 10, 15, 20]),
        modes=cfg.get("modes", ["loop", "point_to_point"]),
        beam_width=cfg.get("routing.beam_width", 80),
        max_routes_per_budget=cfg.get("routing.max_routes_per_budget", 25),
        max_start_nodes=cfg.get("routing.max_start_nodes", 2000),
        start_node_grid_m=cfg.get("routing.start_node_grid_m", 1500),
        max_start_nodes_per_grid=cfg.get("routing.max_start_nodes_per_grid", 25),
        route_diversity_grid_m=cfg.get("routing.route_diversity_grid_m", 2500),
        max_routes_per_grid=cfg.get("routing.max_routes_per_grid", 3),
        frontier_grid_m=cfg.get("routing.frontier_grid_m", 2500),
        max_frontier_states_per_grid=cfg.get("routing.max_frontier_states_per_grid", 20),
        jaccard_similarity_threshold=cfg.get("routing.jaccard_similarity_threshold", 0.85),
        length_jaccard_similarity_threshold=cfg.get("routing.length_jaccard_similarity_threshold", 0.78),
        length_containment_similarity_threshold=cfg.get("routing.length_containment_similarity_threshold", 0.88),
        landmark_path_name_patterns=cfg.get("routing.landmark_path_name_patterns", []),
        min_ascent_m=cfg.get("routing.min_ascent_m", 30),
    )
    out = cfg.outputs_dir / "routes"
    write_route_outputs(routes, graph, out)
    routes_to_json(routes, out / "routes.json")
    console.print(f"Wrote {len(routes)} routes to {out}")


@app.command()
def visualize(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Render the static visualization dashboard from searched routes."""
    cfg = _load(config)
    graph = _read_graph(cfg)
    routes = routes_from_json(cfg.outputs_dir / "routes" / "routes.json")
    render_visualization(routes, graph, cfg.outputs_dir / "viz", cfg)
    console.print(f"Wrote dashboard to {cfg.outputs_dir / 'viz' / 'index.html'}")


@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Serve outputs/viz locally with Python's static file server."""
    import http.server
    import socketserver

    cfg = _load(config)
    viz_dir = cfg.outputs_dir / "viz"
    console.print(f"Serving {viz_dir} at http://127.0.0.1:{port}")
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=viz_dir, **kwargs)
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()


@app.command()
def qa(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Run local artifact and route consistency checks."""
    cfg = _load(config)
    graph = _read_graph(cfg)
    routes = routes_from_json(cfg.outputs_dir / "routes" / "routes.json")
    results = run_diagnostics(routes, graph, cfg)
    for item in results:
        style = "green" if item["ok"] else "red"
        console.print(f"[{style}]{item['name']}[/]: {item['message']}")
    if not builtins.all(item["ok"] for item in results):
        raise typer.Exit(1)


@app.command()
def all(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Run the full local pipeline and end by rendering visualization."""
    cfg = _load(config)
    fetch_all_datasets(cfg)
    fetch_osm_extract(cfg)
    paths = build_paths_from_config(cfg)
    nodes, edges = build_edge_node_tables(
        paths,
        node_precision_m=cfg.get("path_processing.node_precision_m", 0.01),
        split_intersections=cfg.get("path_processing.split_intersections", False),
        endpoint_snap_tolerance_m=cfg.get("path_processing.endpoint_snap_tolerance_m", 5.0),
    )
    path_dir = cfg.data_dir / "processed" / "paths"
    paths.to_parquet(path_dir / "paths.parquet")
    nodes.to_parquet(path_dir / "nodes.parquet")
    edges.to_parquet(path_dir / "edges.parquet")
    build_elevation(config)
    build_graph(config)
    search(config)
    visualize(config)


def _read_graph(cfg: AppConfig):
    with (cfg.data_dir / "processed" / "graph" / "graph.pkl").open("rb") as handle:
        return pickle.load(handle)


def _data_gov_path(cfg: AppConfig, name: str) -> Path:
    dataset_id = cfg.get(f"sources.data_gov.{name}")
    metadata_path = cfg.data_dir / "raw" / "data_gov" / f"{name}.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
        return Path(payload["path"])
    candidates = sorted((cfg.data_dir / "raw" / "data_gov").glob(f"{name}.*"))
    candidates = [path for path in candidates if path.suffix.lower() not in {".json", ".txt"}]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No downloaded file found for {name} ({dataset_id}). Run `sggain fetch` first.")
