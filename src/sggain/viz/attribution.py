from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

DATA_GOV_VIEW_URL = "https://data.gov.sg/datasets/{dataset_id}/view"

SOURCE_METADATA: dict[str, dict[str, str | None]] = {
    "nparks_tracks": {
        "label": "NParks Tracks",
        "publisher": "National Parks Board via data.gov.sg",
        "dataset_id": "d_306cc1018cb733346681883ee6d73054",
        "url": DATA_GOV_VIEW_URL.format(dataset_id="d_306cc1018cb733346681883ee6d73054"),
        "description": "Walking-permitted NParks track segments used for route network geometry.",
    },
    "central_nature_reserve": {
        "label": "Central Nature Reserve Hiking Trails",
        "publisher": "National Parks Board via data.gov.sg",
        "dataset_id": "d_84e6db6c43ffb8b17803334ec2b5c95d",
        "url": DATA_GOV_VIEW_URL.format(dataset_id="d_84e6db6c43ffb8b17803334ec2b5c95d"),
        "description": "Central Nature Reserve hiking trail geometry used for route network segments.",
    },
    "park_connector_loop": {
        "label": "Park Connector Loop",
        "publisher": "National Parks Board via data.gov.sg",
        "dataset_id": "d_a69ef89737379f231d2ae93fd1c5707f",
        "url": DATA_GOV_VIEW_URL.format(dataset_id="d_a69ef89737379f231d2ae93fd1c5707f"),
        "description": "Park connector loop segments used where they form part of selected routes.",
    },
    "national_map_line": {
        "label": "SLA National Map Line",
        "publisher": "Singapore Land Authority via data.gov.sg",
        "dataset_id": "d_10480c0b59e65663dfae1028ff4aa8bb",
        "url": DATA_GOV_VIEW_URL.format(dataset_id="d_10480c0b59e65663dfae1028ff4aa8bb"),
        "description": "Contour lines parsed and interpolated for elevation, ascent, descent, and profile charts.",
    },
    "osm": {
        "label": "OpenStreetMap walkable paths",
        "publisher": "OpenStreetMap contributors via Geofabrik",
        "dataset_id": None,
        "url": "https://download.geofabrik.de/asia/malaysia-singapore-brunei.html",
        "description": "Walkable OSM ways used as enrichment for paths, tracks, footways, and steps.",
    },
    "fixture": {
        "label": "Fixture",
        "publisher": "Synthetic test fixture",
        "dataset_id": None,
        "url": None,
        "description": "Synthetic route network data used by tests.",
    },
}


def source_attribution_records(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path_sources": _path_source_records(samples),
        "source_records": _source_records_used(samples),
        "elevation_source": _source_record("national_map_line"),
    }


def _source_records_used(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    total_distance = _total_sample_distance(samples)

    for current, following in zip(samples, samples[1:], strict=False):
        delta = max(0.0, _float(following.get("cum_distance_m")) - _float(current.get("cum_distance_m")))
        if delta <= 0:
            continue
        source_key = _source_key(current.get("source_primary"))
        source_feature_id = _source_feature_id(current)
        group_key = (source_key, source_feature_id)
        record = grouped.setdefault(
            group_key,
            {
                "source_label": _source_record(source_key)["label"],
                "source_primary": source_key,
                "source_feature_id": source_feature_id,
                "distance_m": 0.0,
                "share_pct": 0.0,
                "source_confidence": _float(current.get("source_confidence"), 0.5),
                "derived_edge_ids": [],
                "attributes": _compact_attributes(current),
            },
        )
        record["distance_m"] += delta
        edge_id = current.get("edge_id") or current.get("undirected_edge_id")
        if edge_id and edge_id not in record["derived_edge_ids"]:
            record["derived_edge_ids"].append(edge_id)

    for record in grouped.values():
        record["distance_m"] = round(record["distance_m"], 1)
        record["share_pct"] = round((record["distance_m"] / total_distance) * 100, 1) if total_distance > 0 else 0.0

    return sorted(grouped.values(), key=lambda item: (-item["distance_m"], item["source_primary"], item["source_feature_id"]))


def _path_source_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples:
        return []

    distance_by_source: dict[str, float] = defaultdict(float)
    confidence_by_source: dict[str, float] = defaultdict(float)
    total_distance = 0.0
    for current, following in zip(samples, samples[1:], strict=False):
        delta = max(0.0, _float(following.get("cum_distance_m")) - _float(current.get("cum_distance_m")))
        if delta <= 0:
            continue
        source_key = _source_key(current.get("source_primary"))
        confidence = _float(current.get("source_confidence"), 0.5)
        distance_by_source[source_key] += delta
        confidence_by_source[source_key] += confidence * delta
        total_distance += delta

    if total_distance <= 0:
        counts = Counter(_source_key(sample.get("source_primary")) for sample in samples)
        total_count = sum(counts.values()) or 1
        return [
            {
                **_source_record(source_key),
                "distance_m": 0.0,
                "share_pct": round((count / total_count) * 100, 1),
                "confidence": round(_mean_confidence(samples, source_key), 2),
            }
            for source_key, count in counts.most_common()
        ]

    records = []
    for source_key, distance_m in sorted(distance_by_source.items(), key=lambda item: (-item[1], item[0])):
        records.append(
            {
                **_source_record(source_key),
                "distance_m": round(distance_m, 1),
                "share_pct": round((distance_m / total_distance) * 100, 1),
                "confidence": round(confidence_by_source[source_key] / distance_m, 2),
            }
        )
    return records


def _total_sample_distance(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        return 0.0
    return max(0.0, _float(samples[-1].get("cum_distance_m")) - _float(samples[0].get("cum_distance_m")))


def _source_feature_id(sample: dict[str, Any]) -> str:
    value = sample.get("source_feature_id")
    if value is None or str(value).strip().lower() in {"", "nan"}:
        value = sample.get("edge_id") or sample.get("undirected_edge_id") or "unknown"
    return str(value)


def _compact_attributes(sample: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_primary",
        "source_feature_id",
        "path_name",
        "highway",
        "trail_type",
        "source_confidence",
    ]
    return {key: sample.get(key) for key in keys if _present(sample.get(key))}


def _present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _source_record(source_key: str) -> dict[str, Any]:
    metadata = SOURCE_METADATA.get(source_key)
    if metadata is None:
        label = source_key.replace("_", " ").title()
        metadata = {
            "label": label,
            "publisher": "Processed local route data",
            "dataset_id": None,
            "url": None,
            "description": f"Route network segments tagged as {source_key}.",
        }
    return {"source_key": source_key, **metadata}


def _source_key(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_confidence(samples: list[dict[str, Any]], source_key: str) -> float:
    values = [
        _float(sample.get("source_confidence"), 0.5)
        for sample in samples
        if _source_key(sample.get("source_primary")) == source_key
    ]
    return sum(values) / len(values) if values else 0.5
