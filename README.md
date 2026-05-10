# sg-hiking-gain

`sggain` is a personal local Python CLI for finding and visualizing Singapore walking and hiking routes with high cumulative elevation gain.

It downloads public path and contour data, builds a local walkable graph, searches routes under distance budgets, and renders static Leaflet/Plotly HTML dashboards. It intentionally does not include a backend API, hosted app, auth system, database server, scheduler, Docker deployment, mobile app, or runtime LLM agent.

## Quick Start

```bash
uv run sggain --help
uv run sggain all --config config.example.yaml
```

The end-to-end command writes route artifacts under `outputs/routes/` and visualization files under `outputs/viz/`.
