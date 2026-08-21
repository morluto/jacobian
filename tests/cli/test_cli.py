from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jacobian.cli import app


def test_cli_exposes_only_stateless_math_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("catalog", "inspect", "run"):
        assert command in result.stdout
    for removed in ("init", "update", "--state-dir", "artifact-put"):
        assert removed not in result.stdout


def test_cli_catalog_inspect_and_run_are_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    catalog_call = runner.invoke(app, ["catalog"])
    inspect_call = runner.invoke(app, ["inspect", "matrix.determinant.compute"])
    run_call = runner.invoke(
        app,
        [
            "run",
            "matrix.determinant.compute",
            "--json",
            json.dumps(
                {
                    "matrix": {
                        "matrix_schema_version": "1",
                        "domain": "QQ",
                        "entries": [[{"num": "1", "den": "1"}]],
                    }
                }
            ),
        ],
    )
    assert list(tmp_path.iterdir()) == []

    assert catalog_call.exit_code == inspect_call.exit_code == run_call.exit_code == 0
    descriptor = json.loads(inspect_call.stdout)
    assert descriptor in json.loads(catalog_call.stdout)["operations"]
    assert json.loads(run_call.stdout)["output"]["determinant"] == {
        "num": "1",
        "den": "1",
    }


@pytest.mark.parametrize("arguments", [(), ("--json", "{}", "--file", "input.json")])
def test_cli_run_requires_exactly_one_payload_source(
    arguments: tuple[str, ...],
) -> None:
    result = CliRunner().invoke(
        app, ["run", "integer.compute.extended_gcd", *arguments]
    )

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert error["message"] == ("pass exactly one of --json or --file")
