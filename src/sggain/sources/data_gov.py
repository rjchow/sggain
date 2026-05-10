from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from sggain.config import AppConfig

POLL_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"


def poll_download_url(dataset_id: str, timeout: int = 30, max_polls: int = 20) -> str:
    url = POLL_URL.format(dataset_id=dataset_id)
    for attempt in range(max_polls):
        response = requests.get(url, timeout=timeout)
        if response.status_code == 429:
            time.sleep(_retry_after_seconds(response, attempt))
            continue
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {})
        status = str(data.get("status", "")).lower()
        if data.get("url"):
            return str(data["url"])
        if status in {"ready", "success", "complete"} and data.get("downloadUrl"):
            return str(data["downloadUrl"])
        if status in {"failed", "error"}:
            raise RuntimeError(f"data.gov.sg download failed for {dataset_id}: {payload}")
        time.sleep(min(2**attempt, 10))
    raise TimeoutError(f"Timed out waiting for data.gov.sg dataset {dataset_id}")


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout, stream=True) as response:
        if response.status_code == 429:
            raise RuntimeError("Rate limit exceeded while downloading data.gov.sg file")
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def fetch_all_datasets(cfg: AppConfig) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    raw_dir = cfg.data_dir / "raw" / "data_gov"
    for name, dataset_id in cfg.get("sources.data_gov", {}).items():
        cached = _cached_record(raw_dir, name, dataset_id)
        if cached is not None:
            records.append(cached)
            continue
        url = poll_download_url(dataset_id)
        suffix = _suffix_from_url(url) or ".geojson"
        destination = raw_dir / f"{name}{suffix}"
        if not destination.exists():
            download_file(url, destination)
        metadata = {"name": name, "dataset_id": dataset_id, "path": _safe_relative_path(destination, cfg.root)}
        (raw_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2))
        records.append({"name": name, "dataset_id": dataset_id, "url": "", "path": str(destination)})
    return records


def _suffix_from_url(url: str) -> str:
    path = Path(urlparse(url).path)
    suffixes = "".join(path.suffixes)
    return suffixes or ".geojson"


def _cached_record(raw_dir: Path, name: str, dataset_id: str) -> dict[str, str] | None:
    metadata_path = raw_dir / f"{name}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        path = _resolve_metadata_path(raw_dir, str(metadata.get("path", "")))
        if path.exists():
            return {
                "name": str(metadata.get("name", name)),
                "dataset_id": str(metadata.get("dataset_id", dataset_id)),
                "url": "",
                "path": str(path),
            }
    for candidate in sorted(raw_dir.glob(f"{name}.*")):
        if candidate.suffix.lower() in {".json", ".txt"}:
            continue
        return {"name": name, "dataset_id": dataset_id, "url": "", "path": str(candidate)}
    return None


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _resolve_metadata_path(raw_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = raw_dir.parents[2]
    root_relative = root / path
    if root_relative.exists():
        return root_relative
    return raw_dir / path


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    if header is not None:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    return float(min(2**attempt, 60))
