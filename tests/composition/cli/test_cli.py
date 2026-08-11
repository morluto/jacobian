from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from jacobian.cli import CliState, JacobianGroup, _public_error, app
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.storage.errors import StorageLimitError, UnsupportedStateVersionError


def test_cli_init_reports_reference_domains_and_polytope_formats(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        ["--state-dir", str(tmp_path), "init", "--json"],
    )

    assert result.exit_code == 0
    catalog = json.loads(result.stdout)
    # Required reference surfaces; do not freeze the full key set (merge magnet).
    required = {
        "erdos_straus",
        "graph_paths",
        "matrices",
        "finite_polytopes",
        "finite_magmas",
        "lean4",
        "rational_polynomial_maps",
        "simple_undirected_graphs",
    }
    assert required <= set(catalog)
    assert catalog["erdos_straus"]["witness_checker_ids"][
        "erdos_straus.decomposition_table"
    ].startswith("checker://sha256/")
    assert catalog["finite_polytopes"]["certificate_checker_id"].startswith(
        "checker://sha256/"
    )
    assert catalog["lean4"]["lean_version"] == "4.31.0"
    assert catalog["lean4"]["profiles"]["MATHLIB"]["mathlib_commit"] == (
        "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
    )
    assert catalog["lean4"]["profiles"]["MATHLIB"]["checker_timeout_seconds"] == 225


def test_cli_init_has_a_human_readable_default_summary(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["--state-dir", str(tmp_path), "init"])

    assert result.exit_code == 0
    assert len(result.stdout) < 1_000
    assert f"Initialized Jacobian state in {tmp_path.resolve()}" in result.stdout
    assert "reference domains" in result.stdout
    assert "capabilities" in result.stdout
    assert "graph_paths" in result.stdout


def test_cli_help_exposes_v02_operations() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "structure-canonicalize",
        "search-enumerate",
        "search-run",
        "experiment-inspect",
        "experiment-cancel",
        "experiment-pause",
        "experiment-resume",
        "conjecture-repair",
        "conjecture-generate",
        "parameter-generalize",
        "transform-apply",
        "transform-verify",
        "polytope-separate",
        "provider-measure",
    ):
        assert command in result.stdout


def test_cli_measures_exact_provider_without_implicit_cold_install(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path),
            "--checker-authority",
            "NONE",
            "provider-measure",
            "graph.compute.properties",
        ],
    )

    assert result.exit_code == 0, result.stdout
    measurement = json.loads(result.stdout)
    assert measurement["provider_runtime"]["provider"] == "jacobian.networkx"
    assert measurement["measurement_version"] == "2"
    assert measurement["installed_size"]["status"] == "COMPLETED"
    assert measurement["installed_size"]["bytes"] > 0
    assert measurement["cold_install"]["status"] == "SKIPPED"
    assert measurement["cold_start"]["status"] == "COMPLETED"
    assert measurement["cold_start"]["peak_rss_bytes"] > 0
    assert measurement["reproduction_case"]["status"] == "COMPLETED"


def test_cli_missing_input_file_returns_an_actionable_json_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    state_dir = tmp_path / "state"

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(state_dir),
            "artifact-put",
            "schema://missing",
            "semantics://missing",
            str(missing),
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "error": {
            "code": "INPUT_FILE_UNAVAILABLE",
            "message": "Jacobian could not read the input file.",
            "hint": "Check that the path exists and is readable, then retry.",
        }
    }
    assert "Traceback" not in result.stderr
    assert str(missing) not in result.stderr
    assert not state_dir.exists()


def test_cli_invalid_json_returns_an_actionable_json_error(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": NaN}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path / "state"),
            "artifact-put",
            "schema://missing",
            "semantics://missing",
            str(payload),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "error": {
            "code": "INVALID_INPUT",
            "message": "Jacobian could not use the supplied input.",
            "hint": (
                "Check the command arguments and JSON payload against the "
                "documented schema, then retry."
            ),
        }
    }
    assert "NaN" not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_cleanup_failure_does_not_replace_translated_command_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": NaN}', encoding="utf-8")

    def failing_close(_state: CliState) -> None:
        raise RuntimeError("state close failure")

    monkeypatch.setattr(CliState, "close", failing_close)

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path / "state"),
            "artifact-put",
            "schema://missing",
            "semantics://missing",
            str(payload),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "error": {
            "code": "INVALID_INPUT",
            "message": "Jacobian could not use the supplied input.",
            "hint": (
                "Check the command arguments and JSON payload against the "
                "documented schema, then retry."
            ),
        }
    }
    assert result.exception is not None
    translated_exit = result.exception.__context__
    assert translated_exit is not None
    assert translated_exit.__notes__ == ["CLI cleanup also failed: state close failure"]


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
    original_close = CliState.close

    def close_then_fail(state: CliState) -> None:
        original_close(state)
        raise cleanup_error

    monkeypatch.setattr(CliState, "close", close_then_fail)

    result = CliRunner().invoke(
        test_app,
        ["succeed"],
    )

    assert result.exit_code == 1
    assert result.exception is cleanup_error
    assert result.stdout == "command completed\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("as_directory", "expected_code"),
    [
        (False, "INVALID_INPUT"),
        (True, "INPUT_FILE_UNAVAILABLE"),
    ],
)
def test_cli_json_shape_and_read_errors_use_the_json_envelope(
    tmp_path: Path,
    *,
    as_directory: bool,
    expected_code: str,
) -> None:
    payload = tmp_path / "payload.json"
    if as_directory:
        payload.mkdir()
    else:
        payload.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path / "state"),
            "artifact-put",
            "schema://missing",
            "semantics://missing",
            str(payload),
        ],
    )

    assert result.exit_code in {1, 2}
    assert json.loads(result.stderr)["error"]["code"] == expected_code
    assert "Traceback" not in result.stderr
    assert "Usage:" not in result.stderr


def test_cli_storage_limit_has_a_capacity_recovery_action() -> None:
    error, exit_code = _public_error(StorageLimitError("fixture internal limit"))

    assert exit_code == 1
    assert error == {
        "code": "STORAGE_LIMIT_REACHED",
        "message": "The input or stored data exceeds a configured size limit.",
        "hint": (
            "Reduce the payload size or free space in the state directory, then retry."
        ),
    }
    assert "fixture" not in str(error)


def test_cli_unsupported_state_version_is_typed_and_preserves_state() -> None:
    error, exit_code = _public_error(
        UnsupportedStateVersionError(2, minimum_revision=3)
    )

    assert exit_code == 1
    assert error["code"] == "UNSUPPORTED_STATE_VERSION"
    assert "revision 2" in error["message"]
    assert "floor 3" in error["message"]
    assert "fresh state directory" in error["hint"]


def test_cli_enumeration_completes_before_the_local_process_exits(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    reference = runtime.portfolio.references["matrices"]
    claim = runtime.core.artifacts.put(
        schema_uri=reference.claim_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "jacobian.integer-matrices",
            "domain_version": "1",
            "semantics_uri": reference.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "is_nonsingular", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    bounds = tmp_path / "bounds.json"
    bounds.write_text(
        json.dumps({"rows": 1, "cols": 1, "entries": [0]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--state-dir",
            str(tmp_path),
            "search-enumerate",
            claim.artifact_uri,
            reference.plugin_id,
            str(bounds),
            "--candidates-max",
            "1",
            "--wall-seconds",
            "30",
            "--page-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state"] == "COMPLETED"
    assert payload["stop_reason"] == "COMPLETE"
    assert payload["verification"] == "UNVERIFIED"
