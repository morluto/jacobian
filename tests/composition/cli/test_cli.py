from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from jacobian.cli import CliState, JacobianGroup, app


def test_cli_exposes_only_stateless_math_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("catalog", "inspect", "run"):
        assert command in result.stdout
    for removed in ("init", "update", "--state-dir", "artifact-put"):
        assert removed not in result.stdout


def test_cli_catalog_inspect_and_run_are_inline(tmp_path: Path) -> None:
    runner = CliRunner()
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

    assert catalog_call.exit_code == inspect_call.exit_code == run_call.exit_code == 0
    descriptor = json.loads(inspect_call.stdout)
    assert descriptor in json.loads(catalog_call.stdout)["operations"]
    assert json.loads(run_call.stdout)["output"]["result"]["determinant"] == {
        "num": "1",
        "den": "1",
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [(), ("--json", "{}", "--file", "input.json")])
def test_cli_run_requires_exactly_one_payload_source(
    arguments: tuple[str, ...],
) -> None:
    result = CliRunner().invoke(app, ["run", "integer.compute.gcd", *arguments])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["message"] == (
        "pass exactly one of --json or --file"
    )


def test_cli_cleanup_failure_propagates_after_successful_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = typer.Typer(cls=JacobianGroup)

    @test_app.callback()
    def configure_test_state(context: typer.Context) -> None:
        context.obj = CliState()

    @test_app.command("succeed")
    def succeed() -> None:
        typer.echo("command completed")

    cleanup_error = RuntimeError("state close failure")
    monkeypatch.setattr(
        CliState, "close", lambda _state: (_ for _ in ()).throw(cleanup_error)
    )

    result = CliRunner().invoke(test_app, ["succeed"])

    assert result.exit_code == 1
    assert result.exception is cleanup_error
