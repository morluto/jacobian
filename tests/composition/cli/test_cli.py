from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from jacobian.cli import CliState, JacobianGroup, app
from jacobian.runtime import CheckerAuthorityMode


def test_cli_help_exposes_only_math_and_operator_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("init", "catalog", "inspect", "run", "provider-measure"):
        assert command in result.stdout
    for deleted in ("search-enumerate", "experiment-inspect", "artifact-put"):
        assert deleted not in result.stdout


def test_cli_init_reports_installed_operation_count(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["--state-dir", str(tmp_path), "init"],
    )

    assert result.exit_code == 0, result.stderr
    assert f"Initialized Jacobian state in {tmp_path.resolve()}" in result.stdout
    assert "Installed " in result.stdout
    assert " mathematical operations." in result.stdout


def test_cli_catalog_and_inspect_share_installed_declaration(tmp_path: Path) -> None:
    runner = CliRunner()
    common = [
        "--state-dir",
        str(tmp_path),
        "--checker-authority",
        "NONE",
    ]

    catalog_call = runner.invoke(app, [*common, "catalog"])
    inspect_call = runner.invoke(
        app,
        [*common, "inspect", "matrix.determinant.compute"],
    )

    assert catalog_call.exit_code == 0, catalog_call.stderr
    assert inspect_call.exit_code == 0, inspect_call.stderr
    catalog = json.loads(catalog_call.stdout)
    descriptor = json.loads(inspect_call.stdout)
    assert descriptor["capability_id"] == "matrix.determinant.compute"
    assert descriptor in catalog["capabilities"]


def test_cli_run_executes_one_installed_operation_from_inline_json(
    tmp_path: Path,
) -> None:
    payload = {
        "matrix": {
            "matrix_schema_version": "1",
            "domain": "QQ",
            "entries": [
                [
                    {"num": "1", "den": "1"},
                    {"num": "2", "den": "1"},
                ],
                [
                    {"num": "3", "den": "1"},
                    {"num": "4", "den": "1"},
                ],
            ],
        }
    }

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path),
            "--checker-authority",
            "NONE",
            "run",
            "matrix.determinant.compute",
            "--json",
            json.dumps(payload),
        ],
    )

    assert result.exit_code == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["execution"]["status"] == "COMPLETED"
    assert response["output"]["result"]["determinant"] == {
        "num": "-2",
        "den": "1",
    }


def test_cli_run_reads_strict_json_file(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"left":"12","right":"18"}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path / "state"),
            "--checker-authority",
            "NONE",
            "run",
            "integer.compute.gcd",
            "--file",
            str(payload),
        ],
    )

    assert result.exit_code == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["execution"]["status"] == "COMPLETED"


@pytest.mark.parametrize("arguments", [(), ("--json", "{}", "--file", "input.json")])
def test_cli_run_requires_exactly_one_payload_source(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path),
            "run",
            "integer.compute.gcd",
            *arguments,
        ],
    )

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["message"] == "pass exactly one of --json or --file"


def test_cli_cleanup_failure_propagates_after_successful_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = typer.Typer(cls=JacobianGroup)

    @test_app.callback()
    def configure_test_state(context: typer.Context) -> None:
        context.obj = CliState(
            tmp_path,
            checker_authority=CheckerAuthorityMode.NONE,
        )

    @test_app.command("succeed")
    def succeed() -> None:
        typer.echo("command completed")

    cleanup_error = RuntimeError("state close failure")

    def fail_close(_state: CliState) -> None:
        raise cleanup_error

    monkeypatch.setattr(CliState, "close", fail_close)

    result = CliRunner().invoke(test_app, ["succeed"])

    assert result.exit_code == 1
    assert result.exception is cleanup_error
    assert result.stdout == "command completed\n"
