from __future__ import annotations

import threading
from pathlib import Path

import pytest
from tests.support.provider_external_sat import drat_trim_runtime_available

from jacobian.bounded_process import bounded_process_cancellation
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import SatAssignmentArtifact, SatProofArtifact
from jacobian.providers.external_solver_runtime import cadical_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime
from jacobian.sat_smt.cadical import install_cadical_capabilities


def _fake_cadical(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake-cadical"
    executable.write_text(
        (
            "#!/usr/bin/python3\n"
            "import pathlib\n"
            "import sys\n"
            "import time\n"
            "if '--version' in sys.argv:\n"
            "    print('3.0.1')\n"
            "    raise SystemExit(0)\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _runtime_with_fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executable: Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
) -> JacobianRuntime:
    unavailable = cadical_provider_runtime(tmp_path / "not-installed")
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.cadical_provider_runtime",
        lambda *_args, **_kwargs: unavailable,
    )
    runtime = create_runtime(
        tmp_path / "store",
        checker_authority=checker_authority,
    )
    provider = cadical_provider_runtime(executable)
    assert provider.availability is CapabilityProviderAvailability.AVAILABLE
    for adapter in install_cadical_capabilities(
        runtime.core.sat,
        provider,
        executable=executable,
    ):
        runtime.core.capabilities.register(adapter)
    return runtime


def _invoke(
    runtime: JacobianRuntime,
    capability_id: str,
    cnf_uri: str,
    *,
    wall_seconds: int = 2,
    conflicts: int | None = None,
):
    budget: dict[str, int] = {"wall_seconds": wall_seconds}
    if conflicts is not None:
        budget["conflicts"] = conflicts
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            mode=CapabilityMode.EXPLORE,
            input={
                "cnf_uri": cnf_uri,
                "resource_budget": budget,
            },
        )
    )


def test_model_find_materializes_only_an_unverified_bound_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "assert pathlib.Path(sys.argv[-1]).read_bytes() == "
        "b'p cnf 2 2\\n1 0\\n-2 0\\n'\n"
        "assert sys.argv[sys.argv.index('-c') + 1] == '50'\n"
        "print('s SATISFIABLE')\n"
        "print('v 1 -2 0')\n"
        "raise SystemExit(10)",
    )
    runtime = _runtime_with_fake(
        tmp_path,
        monkeypatch,
        executable,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    cnf = runtime.core.sat.put_cnf(
        variable_names=("y", "x"),
        clauses=((2,), (-1,)),
    )

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri, conflicts=50)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "ASSIGNMENT_PRODUCED"
    assert result.output["solver_status"] == "SATISFIABLE"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["assignment"] == {"x": True, "y": False}
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.assurance.verification_record_uri is None
    assignment_uri = result.output["assignment_uri"]
    stored = runtime.core.store.get(assignment_uri)
    assignment = SatAssignmentArtifact.model_validate(stored.payload)
    assert assignment.values == (True, False)
    assert assignment.cnf.cnf_artifact_uri == cnf.artifact_uri
    assert assignment.resource_budget.wall_seconds == 2
    assert assignment.resource_budget.conflicts == 50
    assert assignment.producer.provider == "cadical"
    assert stored.manifest.parents == (cnf.artifact_uri,)

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.model.verify",
            mode=CapabilityMode.VERIFY,
            input={"assignment_uri": assignment_uri},
        )
    )
    assert verified.output["status"] == "VERIFIED_SATISFYING"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_model_find_returns_named_values_after_lexicographic_remapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "assert pathlib.Path(sys.argv[-1]).read_bytes() == "
        "b'p cnf 3 3\\n1 0\\n2 0\\n-3 0\\n'\n"
        "print('s SATISFIABLE')\n"
        "print('v 1 2 -3 0')\n"
        "raise SystemExit(10)",
    )
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("n1", "n2", "n10"),
        clauses=((1,), (-2,), (3,)),
    )

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.output["assignment"] == {
        "n1": True,
        "n10": True,
        "n2": False,
    }
    assert "named variable map" in result.output["detail"]


def test_proof_find_normalizes_deletions_without_self_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "assert '--no-binary' in sys.argv\n"
        "assert pathlib.Path(sys.argv[-2]).read_bytes() == "
        "b'p cnf 1 2\\n-1 0\\n1 0\\n'\n"
        "proof = pathlib.Path(sys.argv[-1])\n"
        "proof.write_bytes(b'0\\nd -1 0\\n')\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x",),
        clauses=((1,), (-1,)),
    )

    result = _invoke(runtime, "sat.unsat_proof.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "PROOF_PRODUCED"
    assert result.output["solver_status"] == "UNSATISFIABLE"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.assurance.verification_record_uri is None
    proof_uri = result.output["proof_uri"]
    stored = runtime.core.store.get(proof_uri)
    proof = SatProofArtifact.model_validate(stored.payload)
    assert proof.raw_bytes() == b"0\n"
    assert "removed 1 operational deletion step(s)" in result.output["detail"]
    assert proof.proof_format == "DRAT"
    assert proof.proof_format_version == "drat-text/v1"
    assert stored.manifest.parents == (cnf.artifact_uri,)


@pytest.mark.skipif(
    not drat_trim_runtime_available(),
    reason="the pinned DRAT-trim runtime is unavailable",
)
def test_cadical_deletion_heavy_proof_replays_in_strict_checker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "proof = pathlib.Path(sys.argv[-1])\n"
        "proof.write_bytes((b'd -2 0\\n' * 50000) + b'0\\n')\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    runtime = _runtime_with_fake(
        tmp_path,
        monkeypatch,
        executable,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x", "y"),
        clauses=((1,), (-1,), (2,)),
    )

    produced = _invoke(runtime, "sat.unsat_proof.find", cnf.artifact_uri)
    proof_uri = produced.output["proof_uri"]
    assert runtime.core.sat.resolve_proof(proof_uri).proof.raw_bytes() == b"0\n"

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.unsat_proof.verify",
            mode=CapabilityMode.VERIFY,
            input={"proof_uri": proof_uri},
        )
    )

    assert verified.output["status"] == "VERIFIED_UNSAT"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize(
    ("capability_id", "body", "expected_status", "solver_status"),
    [
        (
            "sat.model.find",
            "print('s UNSATISFIABLE')\nraise SystemExit(20)",
            "NO_ASSIGNMENT_PRODUCED",
            "UNSATISFIABLE",
        ),
        (
            "sat.unsat_proof.find",
            "print('s SATISFIABLE')\nprint('v 1 0')\nraise SystemExit(10)",
            "NO_PROOF_PRODUCED",
            "SATISFIABLE",
        ),
        (
            "sat.model.find",
            "print('s UNKNOWN')\nraise SystemExit(0)",
            "NO_ASSIGNMENT_PRODUCED",
            "UNKNOWN",
        ),
    ],
)
def test_opposite_or_unknown_solver_status_never_becomes_a_conclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability_id: str,
    body: str,
    expected_status: str,
    solver_status: str,
) -> None:
    executable = _fake_cadical(tmp_path, body)
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, capability_id, cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == expected_status
    assert result.output["solver_status"] == solver_status
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.artifact_uris == (cnf.artifact_uri,)
    assert result.assurance.verification_record_uri is None


def test_timeout_is_operational_and_materializes_no_solver_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(tmp_path, "time.sleep(5)")
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(
        runtime,
        "sat.model.find",
        cnf.artifact_uri,
        wall_seconds=1,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output == {}
    assert result.diagnostics[0].code == "CADICAL_TIMEOUT"
    assert result.artifact_uris == (cnf.artifact_uri,)
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC


def test_client_cancellation_terminates_solver_and_materializes_no_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(tmp_path, "time.sleep(30)")
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))
    cancellation_event = threading.Event()
    cancellation_event.set()

    with bounded_process_cancellation(cancellation_event):
        result = _invoke(
            runtime,
            "sat.model.find",
            cnf.artifact_uri,
            wall_seconds=150,
        )

    assert result.execution.status is ExecutionStatus.CANCELLED
    assert result.diagnostics[0].code == "CADICAL_CANCELLED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_partial_model_fails_closed_without_an_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('s SATISFIABLE')\nprint('v 1 0')\nraise SystemExit(10)",
    )
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x", "y"),
        clauses=((1, 2),),
    )

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output == {}
    assert result.diagnostics[0].code == "INVALID_CADICAL_MODEL"
    assert result.artifact_uris == (cnf.artifact_uri,)


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (
            "print('s UNSATISFIABLE')\nraise SystemExit(10)",
            "INVALID_CADICAL_OUTPUT",
        ),
        (
            "print('solver crashed')\nraise SystemExit(7)",
            "CADICAL_EXECUTION_FAILED",
        ),
    ],
)
def test_inconsistent_protocol_or_nondocumented_exit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    expected_code: str,
) -> None:
    executable = _fake_cadical(tmp_path, body)
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == expected_code
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_excessive_stdout_fails_closed_without_an_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('x' * 4096)\nraise SystemExit(10)",
    )
    monkeypatch.setattr("jacobian.sat_smt.cadical.CADICAL_STDOUT_LIMIT", 1024)
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CADICAL_OUTPUT_LIMIT_EXCEEDED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_oversized_proof_fails_closed_before_reading_or_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "pathlib.Path(sys.argv[-1]).write_bytes(b'x' * 9)\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    monkeypatch.setattr("jacobian.sat_smt.cadical.CADICAL_PROOF_LIMIT", 8)
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x",),
        clauses=((1,), (-1,)),
    )

    result = _invoke(runtime, "sat.unsat_proof.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CADICAL_DURABLE_PROOF_LIMIT_EXCEEDED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_proof_symlink_is_never_followed_or_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "pathlib.Path(sys.argv[-1]).symlink_to('/etc/passwd')\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x",),
        clauses=((1,), (-1,)),
    )

    result = _invoke(runtime, "sat.unsat_proof.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_CADICAL_PROOF_FILE"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_runtime_tampering_after_probe_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('s SATISFIABLE')\nprint('v 1 0')\nraise SystemExit(10)",
    )
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))
    executable.write_text("#!/bin/sh\nexit 10\n", encoding="utf-8")
    executable.chmod(0o755)

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CADICAL_RUNTIME_CHANGED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_invocation_environment_does_not_require_the_callers_locale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "assert os.environ.get('LC_ALL') == 'C'\n"
        "print('s SATISFIABLE')\n"
        "print('v 1 0')\n"
        "raise SystemExit(10)",
    )
    text = executable.read_text(encoding="utf-8").replace(
        "import pathlib\n",
        "import os\nimport pathlib\n",
    )
    executable.write_text(text, encoding="utf-8")
    executable.chmod(0o755)
    runtime = _runtime_with_fake(tmp_path, monkeypatch, executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "ASSIGNMENT_PRODUCED"


def test_wrong_cadical_version_is_unavailable(
    tmp_path: Path,
) -> None:
    executable = _fake_cadical(tmp_path, "raise SystemExit(0)")
    executable.write_text(
        executable.read_text(encoding="utf-8").replace("3.0.1", "3.0.0"),
        encoding="utf-8",
    )

    runtime = cadical_provider_runtime(executable)

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.version is None
    assert runtime.digest is None
    assert runtime.diagnostic is not None
    assert "3.0.1" in runtime.diagnostic
