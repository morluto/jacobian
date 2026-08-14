from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.provider_external_sat import drat_trim_runtime_available
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.bounded_process import bounded_process_cancellation
from jacobian.contracts.operations import (
    OperationRequest,
    ProviderAvailability,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import SatAssignmentArtifact, SatProofArtifact
from jacobian.providers.external_solver_runtime import (
    cadical_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.sat_smt.cadical import install_cadical_operations
from jacobian.sat_smt.sat_operations import (
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)


def _fake_cadical(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake-cadical"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
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


def _install_fake_cadical(
    services: DomainTestServices,
    executable: Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
) -> None:
    provider = cadical_provider_runtime(executable)
    assert provider.availability is ProviderAvailability.AVAILABLE
    with atomic_installation(services.core):
        for adapter in install_cadical_operations(
            services.core.sat,
            provider,
            executable=executable,
        ):
            services.installation.register_operation(adapter)
        if checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED:
            assignment, _assignment_installation = install_sat_assignment_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.sat,
                services.verification,
                services.core.checkers,
                authorize_checker=True,
            )
            assert assignment is not None
            services.installation.register_operation(assignment)
            proof, _proof_installation = install_sat_unsat_proof_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.sat,
                services.verification,
                services.core.checkers,
                drat_trim_provider_runtime(),
                authorize_checker=True,
            )
            if proof is not None:
                services.installation.register_operation(proof)


@pytest.fixture
def fake_cadical_services(
    tmp_path: Path,
) -> Iterator[Callable[..., DomainTestServices]]:
    with ExitStack() as stack:
        opened = 0

        def factory(
            executable: Path,
            *,
            checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
        ) -> DomainTestServices:
            nonlocal opened
            opened += 1
            services = stack.enter_context(
                open_domain_services(
                    tmp_path / f"store-{opened}",
                    checker_authority=checker_authority,
                )
            )
            _install_fake_cadical(
                services,
                executable,
                checker_authority=checker_authority,
            )
            return services

        yield factory


def _invoke(
    runtime: DomainTestServices,
    operation_id: str,
    cnf_uri: str,
    *,
    wall_seconds: int = 2,
    conflicts: int | None = None,
):
    budget: dict[str, int] = {"wall_seconds": wall_seconds}
    if conflicts is not None:
        budget["conflicts"] = conflicts
    return runtime.core.operations.invoke(
        OperationRequest(
            operation_id=operation_id,
            input={
                "cnf_uri": cnf_uri,
                "resource_budget": budget,
            },
        )
    )


def test_model_find_materializes_only_an_unverified_bound_assignment(
    tmp_path: Path,
    fake_cadical_services: Callable[..., DomainTestServices],
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
    runtime = fake_cadical_services(
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
    assert result.output["assignment"] == {"x": True, "y": False}
    assignment_uri = result.output["assignment_uri"]
    stored = runtime.core.store.get(assignment_uri)
    assignment = SatAssignmentArtifact.model_validate(stored.payload)
    assert assignment.values == (True, False)
    assert assignment.cnf.cnf_artifact_uri == cnf.artifact_uri
    assert assignment.resource_budget.wall_seconds == 2
    assert assignment.resource_budget.conflicts == 50
    assert assignment.producer.provider == "cadical"
    assert stored.manifest.parents == (cnf.artifact_uri,)

    verified = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="sat.model.verify",
            input={"assignment_uri": assignment_uri},
        )
    )
    assert verified.output["status"] == "VERIFIED_SATISFYING"
    assert verified.verification_record_uri is not None


def test_model_find_returns_named_values_after_lexicographic_remapping(
    tmp_path: Path,
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "assert pathlib.Path(sys.argv[-1]).read_bytes() == "
        "b'p cnf 3 3\\n1 0\\n2 0\\n-3 0\\n'\n"
        "print('s SATISFIABLE')\n"
        "print('v 1 2 -3 0')\n"
        "raise SystemExit(10)",
    )
    runtime = fake_cadical_services(executable)
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
    fake_cadical_services: Callable[..., DomainTestServices],
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
    runtime = fake_cadical_services(executable)
    cnf = runtime.core.sat.put_cnf(
        variable_names=("x",),
        clauses=((1,), (-1,)),
    )

    result = _invoke(runtime, "sat.unsat_proof.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "PROOF_PRODUCED"
    assert result.output["solver_status"] == "UNSATISFIABLE"
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
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "proof = pathlib.Path(sys.argv[-1])\n"
        "proof.write_bytes((b'd -2 0\\n' * 50000) + b'0\\n')\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    runtime = fake_cadical_services(
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

    verified = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="sat.unsat_proof.verify",
            input={"proof_uri": proof_uri},
        )
    )

    assert verified.output["status"] == "VERIFIED_UNSAT"
    assert verified.verification_record_uri is not None


@pytest.mark.parametrize(
    ("operation_id", "body", "expected_status", "solver_status"),
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
    fake_cadical_services: Callable[..., DomainTestServices],
    operation_id: str,
    body: str,
    expected_status: str,
    solver_status: str,
) -> None:
    executable = _fake_cadical(tmp_path, body)
    runtime = fake_cadical_services(executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, operation_id, cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == expected_status
    assert result.output["solver_status"] == solver_status
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_timeout_is_operational_and_materializes_no_solver_evidence(
    tmp_path: Path,
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(tmp_path, "time.sleep(5)")
    runtime = fake_cadical_services(executable)
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


def test_client_cancellation_terminates_solver_and_materializes_no_evidence(
    tmp_path: Path,
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(tmp_path, "time.sleep(30)")
    runtime = fake_cadical_services(executable)
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
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('s SATISFIABLE')\nprint('v 1 0')\nraise SystemExit(10)",
    )
    runtime = fake_cadical_services(executable)
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
    fake_cadical_services: Callable[..., DomainTestServices],
    body: str,
    expected_code: str,
) -> None:
    executable = _fake_cadical(tmp_path, body)
    runtime = fake_cadical_services(executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == expected_code
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_excessive_stdout_fails_closed_without_an_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('x' * 4096)\nraise SystemExit(10)",
    )
    monkeypatch.setattr("jacobian.sat_smt.cadical.CADICAL_STDOUT_LIMIT", 1024)
    runtime = fake_cadical_services(executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CADICAL_OUTPUT_LIMIT_EXCEEDED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_oversized_proof_fails_closed_before_reading_or_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "pathlib.Path(sys.argv[-1]).write_bytes(b'x' * 9)\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    monkeypatch.setattr("jacobian.sat_smt.cadical.CADICAL_PROOF_LIMIT", 8)
    runtime = fake_cadical_services(executable)
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
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "pathlib.Path(sys.argv[-1]).symlink_to('/etc/passwd')\n"
        "print('s UNSATISFIABLE')\n"
        "raise SystemExit(20)",
    )
    runtime = fake_cadical_services(executable)
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
    fake_cadical_services: Callable[..., DomainTestServices],
) -> None:
    executable = _fake_cadical(
        tmp_path,
        "print('s SATISFIABLE')\nprint('v 1 0')\nraise SystemExit(10)",
    )
    runtime = fake_cadical_services(executable)
    cnf = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))
    executable.write_text("#!/bin/sh\nexit 10\n", encoding="utf-8")
    executable.chmod(0o755)

    result = _invoke(runtime, "sat.model.find", cnf.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CADICAL_RUNTIME_CHANGED"
    assert result.artifact_uris == (cnf.artifact_uri,)


def test_invocation_environment_does_not_require_the_callers_locale(
    tmp_path: Path,
    fake_cadical_services: Callable[..., DomainTestServices],
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
    runtime = fake_cadical_services(executable)
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

    assert runtime.availability is ProviderAvailability.UNAVAILABLE
    assert runtime.version is None
    assert runtime.digest is None
    assert runtime.diagnostic is not None
    assert "3.0.1" in runtime.diagnostic
