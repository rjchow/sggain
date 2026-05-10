from typer.testing import CliRunner

from sggain import cli


def test_qa_uses_builtin_all_instead_of_all_command(monkeypatch, tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"project_root: {tmp_path}\ndata_dir: data\noutputs_dir: outputs\n")

    monkeypatch.setattr(cli, "_read_graph", lambda _cfg: object())
    monkeypatch.setattr(cli, "routes_from_json", lambda _path: [])
    monkeypatch.setattr(cli, "run_diagnostics", lambda _routes, _graph, _cfg: [{"name": "ok", "ok": True, "message": "fine"}])

    result = runner.invoke(cli.app, ["qa", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "ok" in result.output
