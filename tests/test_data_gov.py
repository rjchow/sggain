import json
from pathlib import Path

from sggain.config import AppConfig, DEFAULT_CONFIG
from sggain.sources import data_gov


def test_fetch_all_datasets_reuses_existing_metadata_without_polling(tmp_path, monkeypatch):
    data_file = tmp_path / "data" / "raw" / "data_gov" / "nparks_tracks.geojson"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("{}")
    (data_file.parent / "nparks_tracks.json").write_text(
        json.dumps(
            {
                "name": "nparks_tracks",
                "dataset_id": "existing",
                "url": "https://example.test/nparks.geojson",
                "path": str(data_file),
            }
        )
    )
    cfg = AppConfig(
        values={
            **DEFAULT_CONFIG,
            "project_root": str(tmp_path),
            "sources": {"data_gov": {"nparks_tracks": "existing"}, "osm": {"enabled": False}},
        }
    )

    def fail_poll(_dataset_id):
        raise AssertionError("polling should be skipped for cached datasets")

    monkeypatch.setattr(data_gov, "poll_download_url", fail_poll)

    records = data_gov.fetch_all_datasets(cfg)

    assert records[0]["path"] == str(data_file)


def test_fetch_all_datasets_writes_sanitized_metadata(tmp_path, monkeypatch):
    cfg = AppConfig(
        values={
            **DEFAULT_CONFIG,
            "project_root": str(tmp_path),
            "sources": {"data_gov": {"nparks_tracks": "dataset-id"}, "osm": {"enabled": False}},
        }
    )

    signed_url = (
        "https://s3.example.test/nparks.geojson?"
        "AWSAccessKeyId=TEMPORARY-CREDENTIAL&Signature=redacted&x-amz-security-token=redacted"
    )
    monkeypatch.setattr(data_gov, "poll_download_url", lambda _dataset_id: signed_url)

    def fake_download(_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}")
        return destination

    monkeypatch.setattr(data_gov, "download_file", fake_download)

    data_gov.fetch_all_datasets(cfg)

    metadata = json.loads((tmp_path / "data" / "raw" / "data_gov" / "nparks_tracks.json").read_text())
    assert "url" not in metadata
    assert not Path(metadata["path"]).is_absolute()
    assert "AWSAccessKeyId" not in json.dumps(metadata)


def test_poll_download_url_retries_after_rate_limit(monkeypatch):
    calls = []
    sleeps = []

    class Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("raise_for_status should not be called for retryable 429")

        def json(self):
            return self._payload

    def fake_get(url, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            return Response(429, headers={"Retry-After": "0"})
        return Response(200, {"data": {"status": "ready", "url": "https://example.test/data.geojson"}})

    monkeypatch.setattr(data_gov.requests, "get", fake_get)
    monkeypatch.setattr(data_gov.time, "sleep", lambda seconds: sleeps.append(seconds))

    url = data_gov.poll_download_url("dataset-id", max_polls=2)

    assert url == "https://example.test/data.geojson"
    assert len(calls) == 2
    assert sleeps == [0.0]
