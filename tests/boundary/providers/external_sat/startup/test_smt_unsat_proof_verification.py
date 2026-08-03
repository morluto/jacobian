from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from tests.unit.contracts.artifacts import sha256_file as _sha256_file

import jacobian_checkers.smt
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.evidence import CertificateEnvelope
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.smt import SmtResourceBudget
from jacobian.contracts.verification import VerificationRecord
from jacobian.providers.external_solver_runtime import carcara_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime
from jacobian.verification import CheckerExecutionError
from jacobian.verification._helpers import _environment_digest

_FIXTURES = (
    Path(__file__).resolve().parents[5]
    / "tests"
    / "boundary"
    / "providers"
    / "external_sat"
    / "fixtures"
)
_PROBLEM = (_FIXTURES / "qf_uf_equality_unsat.smt2").read_text(encoding="ascii")
_PROOF = (_FIXTURES / "qf_uf_equality_unsat.alethe").read_bytes()


def _fake_carcara(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "carcara"
    executable.write_text(
        (
            "#!/usr/bin/python3\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('carcara 1.1.0 [git master 394edbb]')\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:] == ['check', '--help']:\n"
            "    print('--strict-parsing --parse-hole-args '\n"
            "          '--allow-int-real-subtyping --expand-let-bindings')\n"
            "    raise SystemExit(0)\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    manifest = executable.with_name(executable.name + ".jacobian-runtime.json")
    manifest.write_text(
        (
            "{\n"
            '  "runtime_manifest_version": "1",\n'
            '  "provider": "carcara",\n'
            '  "version": "1.1.0",\n'
            '  "source_repository": "https://github.com/ufmg-smite/carcara",\n'
            '  "source_commit": '
            '"394edbb15ba95c47893f1d821fddde7e016af178",\n'
            '  "compatible_cvc5_version": "1.3.4",\n'
            f'  "executable_sha256": "{_sha256_file(executable)}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return executable


def _producer() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="cvc5",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1.3.4",
        digest="sha256:" + "d" * 64,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="BSD-3-Clause",
        features=("alethe-proof-production",),
        configuration={
            "profile": "jacobian.smtlib2.qf-unsat/v1",
            "proof_format": "cvc5.alethe/1.3.4",
        },
    )


def _runtime_with_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executable: Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
) -> JacobianRuntime:
    runtime = carcara_provider_runtime(executable)
    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.carcara_provider_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    return create_runtime(
        tmp_path / "store",
        checker_authority=checker_authority,
    )


@pytest.mark.parametrize(
    "attack",
    ["missing_provenance", "wrong_commit", "digest_mismatch"],
)
def test_carcara_runtime_requires_exact_operator_provenance(
    tmp_path: Path,
    attack: str,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    manifest = executable.with_name(executable.name + ".jacobian-runtime.json")
    if attack == "missing_provenance":
        manifest.unlink()
    elif attack == "wrong_commit":
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "394edbb15ba95c47893f1d821fddde7e016af178",
                "0000000000000000000000000000000000000000",
            ),
            encoding="utf-8",
        )
    else:
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    runtime = carcara_provider_runtime(executable)

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.version is None
    assert runtime.digest is None
    assert runtime.diagnostic is not None


def _proof(runtime: JacobianRuntime) -> tuple[str, str]:
    problem = runtime.core.smt.put_problem(logic="QF_UF", smtlib_text=_PROBLEM)
    proof = runtime.core.smt.put_proof(
        problem_uri=problem.artifact_uri,
        proof=_PROOF,
        producer=_producer(),
        resource_budget=SmtResourceBudget(wall_seconds=5),
    )
    return problem.artifact_uri, proof.artifact_uri


def _verify(runtime: JacobianRuntime, proof_uri: str):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="smt.unsat_proof.verify",
            mode=CapabilityMode.VERIFY,
            input={"proof_uri": proof_uri},
        )
    )


def test_unsat_proof_is_verified_by_authorized_strict_carcara(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    runtime = _runtime_with_runtime(tmp_path, monkeypatch, executable)
    problem_uri, proof_uri = _proof(runtime)
    monkeypatch.setattr(
        jacobian_checkers.smt,
        "check_unsat_proof",
        lambda _request: {
            "accepted": False,
            "conclusion": "UNKNOWN",
            "arithmetic": "SYMBOLIC",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": "parent-process monkeypatch",
        },
    )

    result = _verify(runtime, proof_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == "VERIFIED_UNSAT"
    assert result.output["conclusion"] == "TRUE"
    assert result.output["problem_uri"] == problem_uri
    assert result.output["proof_uri"] == proof_uri
    certificate_uri = result.output["certificate_uri"]
    certificate = CertificateEnvelope.model_validate(
        runtime.core.store.get(certificate_uri).payload
    )
    assert certificate.certificate_type == "smt.unsat-proof"
    assert certificate.payload == {
        "problem_uri": problem_uri,
        "proof_uri": proof_uri,
    }
    record_uri = result.output["verification_record_uri"]
    assert record_uri is not None
    record_artifact = runtime.core.store.get(record_uri)
    record = VerificationRecord.model_validate(record_artifact.payload)
    assert record.checker_id == runtime.portfolio.smt_unsat_proof_checker.checker_id
    assert record.evidence_uri == certificate_uri
    checker = runtime.core.checkers.require_active(record.checker_id)
    assert checker.provider_runtime == runtime.portfolio.carcara_runtime
    assert record.environment_digest == _environment_digest(
        checker.executable_digest,
        checker.provider_runtime,
    )
    assert set(record_artifact.manifest.parents) == {
        problem_uri,
        proof_uri,
        certificate_uri,
    }


def test_holey_checker_report_never_establishes_sat_or_unsat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('holey')\nraise SystemExit(0)",
    )
    runtime = _runtime_with_runtime(tmp_path, monkeypatch, executable)
    _problem_uri, proof_uri = _proof(runtime)

    result = _verify(runtime, proof_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["status"] == "REJECTED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None


def test_proof_verify_requires_runtime_and_operator_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    without_references = _runtime_with_runtime(
        tmp_path / "without-references",
        monkeypatch,
        executable,
        checker_authority=CheckerAuthorityMode.NONE,
    )
    unavailable = carcara_provider_runtime(tmp_path / "missing")
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.carcara_provider_runtime",
        lambda *_args, **_kwargs: unavailable,
    )
    without_runtime = create_runtime(
        tmp_path / "without-runtime",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )

    assert without_references.portfolio.smt_unsat_proof_checker.checker_id is None
    assert without_runtime.portfolio.smt_unsat_proof_checker.checker_id is None
    for runtime in (without_references, without_runtime):
        assert "smt.unsat_proof.verify" not in {
            descriptor.capability_id
            for descriptor in runtime.core.capabilities.catalog().capabilities
        }


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_output_status"),
    [
        (
            subprocess.TimeoutExpired(cmd=("carcara",), timeout=1),
            ExecutionStatus.TIMEOUT,
            "TIMEOUT",
        ),
        (
            CheckerExecutionError("deliberate checker crash"),
            ExecutionStatus.ERROR,
            "ERROR",
        ),
    ],
)
def test_checker_operational_failure_never_creates_a_conclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    expected_status: ExecutionStatus,
    expected_output_status: str,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    runtime = _runtime_with_runtime(tmp_path, monkeypatch, executable)
    _problem_uri, proof_uri = _proof(runtime)

    def fail(**_kwargs: Any):
        raise exception

    monkeypatch.setattr(runtime.services.verification, "_run_checker", fail)
    result = _verify(runtime, proof_uri)

    assert result.execution.status is expected_status
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == expected_output_status
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None


def test_runtime_replacement_after_authorization_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_carcara(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    runtime = _runtime_with_runtime(tmp_path, monkeypatch, executable)
    _problem_uri, proof_uri = _proof(runtime)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    result = _verify(runtime, proof_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == "ERROR"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
