from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from tests.support.selected_runtime import selected_runtime_opener
from typer.testing import CliRunner

from jacobian.cli import CliState, JacobianGroup, app, create_cli_app
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.domains.number_theory import number_theory_operations


def test_cli_help_exposes_only_math_and_operator_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("init", "update", "catalog", "inspect", "run"):
        assert command in result.stdout
    for deleted in ("search-enumerate", "experiment-inspect", "artifact-put"):
        assert deleted not in result.stdout


def test_cli_init_reports_installed_operation_count(tmp_path: Path) -> None:
    """Complete-portfolio smoke: jacobian init compiles the built-in catalog."""
    result = CliRunner().invoke(
        app,
        ["--state-dir", str(tmp_path), "init"],
    )

    assert result.exit_code == 0, result.stderr
    assert f"Initialized Jacobian state in {tmp_path.resolve()}" in result.stdout
    assert "Compiled " in result.stdout
    assert " mathematical operations." in result.stdout

    repeated = CliRunner().invoke(
        app,
        ["--state-dir", str(tmp_path), "init"],
    )

    assert repeated.exit_code == 0, repeated.stderr
    assert (
        f"Jacobian state is already current in {tmp_path.resolve()}" in repeated.stdout
    )
    assert "Catalog contains " in repeated.stdout
    assert "Compiled " not in repeated.stdout


def test_cli_catalog_and_inspect_share_installed_declaration(tmp_path: Path) -> None:
    runner = CliRunner()
    selected = create_cli_app(
        runtime_opener=selected_runtime_opener(matrix_operations())
    )
    common = [
        "--state-dir",
        str(tmp_path),
    ]

    catalog_call = runner.invoke(selected, [*common, "catalog"])
    inspect_call = runner.invoke(
        selected,
        [*common, "inspect", "matrix.determinant.compute"],
    )

    assert catalog_call.exit_code == 0, catalog_call.stderr
    assert inspect_call.exit_code == 0, inspect_call.stderr
    catalog = json.loads(catalog_call.stdout)
    descriptor = json.loads(inspect_call.stdout)
    assert descriptor["operation_id"] == "matrix.determinant.compute"
    assert descriptor in catalog["operations"]


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
    selected = create_cli_app(
        runtime_opener=selected_runtime_opener(matrix_operations())
    )

    result = CliRunner().invoke(
        selected,
        [
            "--state-dir",
            str(tmp_path),
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
    selected = create_cli_app(
        runtime_opener=selected_runtime_opener(number_theory_operations())
    )

    result = CliRunner().invoke(
        selected,
        [
            "--state-dir",
            str(tmp_path / "state"),
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
        context.obj = CliState(tmp_path)

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
