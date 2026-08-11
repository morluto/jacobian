from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.artifacts import sha256_file as _sha256_file

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.smt import SmtResourceBudget
from jacobian.sat_smt.smt import install_smt_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian_checkers.smt import check_unsat_proof

_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "boundary"
    / "providers"
    / "external_sat"
    / "fixtures"
)
_QF_UF_PROBLEM = (_FIXTURES / "qf_uf_equality_unsat.smt2").read_text(encoding="ascii")
_QF_UF_PROOF = (_FIXTURES / "qf_uf_equality_unsat.alethe").read_bytes()
ProofRequestFactory = Callable[..., dict[str, Any]]


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


def _checker_artifact(artifact: StoredArtifact) -> dict[str, Any]:
    return {
        "artifact_uri": artifact.artifact_uri,
        "object_digest": artifact.manifest.object_digest,
        "payload_digest": artifact.manifest.payload_digest,
        "schema_uri": artifact.manifest.schema_uri,
        "semantics_uri": artifact.manifest.semantics_uri,
        "parents": list(artifact.manifest.parents),
        "payload": artifact.payload,
    }


@pytest.fixture
def proof_request_factory(
    tmp_path: Path,
) -> Iterator[ProofRequestFactory]:
    store = ArtifactRepository(tmp_path / "store")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    smt = install_smt_artifacts(store, schemas, artifacts)
    semantics = store.get(smt.installation.semantics_uri)
    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )

    def build(
        proof_bytes: bytes = _QF_UF_PROOF,
        *,
        logic: str = "QF_UF",
        smtlib_text: str = _QF_UF_PROBLEM,
    ) -> dict[str, Any]:
        problem_result = smt.put_problem(
            logic=logic,  # type: ignore[arg-type]
            smtlib_text=smtlib_text,
        )
        problem = store.get(problem_result.artifact_uri)
        proof_result = smt.put_proof(
            problem_uri=problem_result.artifact_uri,
            proof=proof_bytes,
            producer=_producer(),
            resource_budget=SmtResourceBudget(wall_seconds=5),
        )
        proof = store.get(proof_result.artifact_uri)
        bindings = EvidenceBindings(
            claim_digest=problem.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=proof.manifest.object_digest,
        )
        payload = {
            "problem_uri": problem.artifact_uri,
            "proof_uri": proof.artifact_uri,
        }
        certificate = CertificateEnvelope(
            certificate_type="smt.unsat-proof",
            format_version="1",
            bindings=bindings,
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
            ),
            payload=payload,
        )
        certificate_result = artifacts.put(
            schema_uri=certificate_schema_uri,
            semantics_uri=smt.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(problem.artifact_uri, proof.artifact_uri),
            summary="SMT UNSAT proof verification certificate",
        )
        stored_certificate = store.get(certificate_result.artifact_uri)
        return {
            "request_version": "1",
            "claim": _checker_artifact(problem),
            "candidate": _checker_artifact(proof),
            "scope": None,
            "certificate": _checker_artifact(stored_certificate),
            "expected_bindings": bindings.model_dump(mode="json"),
        }

    try:
        yield build
    finally:
        store.close()


def _fake_checker(tmp_path: Path, body: str) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "called"
    executable = tmp_path / "carcara"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
            "import pathlib\n"
            "import sys\n"
            f"marker = pathlib.Path({str(marker)!r})\n"
            "marker.write_text('called', encoding='utf-8')\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, marker


def _install_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
    executable: Path,
) -> None:
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        _sha256_file(executable),
    )


def test_checker_reconstructs_exact_inputs_and_uses_only_strict_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "import os\n"
        "assert 'JACOBIAN_AMBIENT_TEST' not in os.environ\n"
        "assert sys.argv[1:7] == [\n"
        "    'check', '--strict-parsing', '--parse-hole-args',\n"
        "    '--allow-int-real-subtyping', '--expand-let-bindings', sys.argv[6]\n"
        "]\n"
        "assert '--ignore-unknown-rules' not in sys.argv\n"
        "assert '--allowed-rules' not in sys.argv\n"
        "proof = pathlib.Path(sys.argv[6]).read_bytes()\n"
        "problem = pathlib.Path(sys.argv[7]).read_bytes()\n"
        f"assert proof == {bytes(_QF_UF_PROOF)!r}\n"
        f"assert problem == {bytes(_QF_UF_PROBLEM, 'ascii')!r}\n"
        "print('valid')\n"
        "raise SystemExit(0)",
    )
    monkeypatch.setenv("JACOBIAN_AMBIENT_TEST", "must-not-leak")
    _install_runtime_environment(monkeypatch, executable)

    decision = check_unsat_proof(proof_request_factory())

    assert marker.read_text(encoding="utf-8") == "called"
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "SYMBOLIC"
    assert decision["method"] == "CHECKED_CERTIFICATE"


def test_only_exact_valid_with_clean_stderr_and_zero_exit_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    bodies = (
        "print('holey')\nraise SystemExit(0)",
        "print('valid')\nraise SystemExit(1)",
        "print('valid')\nprint('valid')\nraise SystemExit(0)",
        "print(' valid ')\nraise SystemExit(0)",
        "print('valid')\nprint('warning', file=sys.stderr)\nraise SystemExit(0)",
    )
    for index, body in enumerate(bodies):
        executable, _marker = _fake_checker(tmp_path / str(index), body)
        _install_runtime_environment(monkeypatch, executable)

        decision = check_unsat_proof(proof_request_factory())

        assert decision["accepted"] is False, body
        assert decision["conclusion"] == "UNKNOWN", body


def test_holes_and_unsupported_logics_are_rejected_before_carcara(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "raise AssertionError('must not run')",
    )
    _install_runtime_environment(monkeypatch, executable)
    holey = proof_request_factory(
        b'(\n(step t0 (cl) :rule hole :args ("unknown"))\n)\n'
    )
    lia = proof_request_factory(
        b"(\n)\n",
        logic="QF_LIA",
        smtlib_text=(_FIXTURES / "qf_lia_bounds_unsat.smt2").read_text(
            encoding="ascii"
        ),
    )

    for request in (holey, lia):
        decision = check_unsat_proof(request)
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
    assert not marker.exists()


def test_binding_and_lineage_attacks_are_rejected_before_carcara(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "raise AssertionError('must not run')",
    )
    _install_runtime_environment(monkeypatch, executable)
    original = proof_request_factory()
    mutations: list[dict[str, Any]] = []

    changed = deepcopy(original)
    changed["claim"]["payload"]["smtlib_text"] = changed["claim"]["payload"][
        "smtlib_text"
    ].replace("(= a b)", "(= b a)", 1)
    mutations.append(changed)

    changed = deepcopy(original)
    changed["candidate"]["payload"]["problem"]["smtlib_digest"] = "sha256:" + "a" * 64
    mutations.append(changed)

    changed = deepcopy(original)
    changed["candidate"]["parents"] = []
    mutations.append(changed)

    changed = deepcopy(original)
    changed["certificate"]["payload"]["payload"]["proof_uri"] = (
        "artifact://sha256/" + "a" * 64
    )
    mutations.append(changed)

    changed = deepcopy(original)
    changed["expected_bindings"]["candidate_digest"] = "sha256:" + "a" * 64
    mutations.append(changed)

    for request in mutations:
        decision = check_unsat_proof(request)
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
    assert not marker.exists()


def test_runtime_digest_mismatch_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "print('valid')\nraise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, executable)
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + "a" * 64,
    )

    decision = check_unsat_proof(proof_request_factory())

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
    assert not marker.exists()


def test_excessive_output_and_runtime_mutation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    noisy, _marker = _fake_checker(
        tmp_path / "noisy",
        "print('x' * 4096)\nraise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, noisy)
    monkeypatch.setattr("jacobian_checkers.smt.CARCARA_OUTPUT_LIMIT", 128)

    noisy_decision = check_unsat_proof(proof_request_factory())

    assert noisy_decision["accepted"] is False
    assert noisy_decision["conclusion"] == "UNKNOWN"

    mutated, _marker = _fake_checker(
        tmp_path / "mutated",
        "pathlib.Path(sys.argv[0]).write_text('#!/bin/sh\\nexit 0\\n')\n"
        "print('valid')\n"
        "raise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, mutated)

    mutated_decision = check_unsat_proof(proof_request_factory())

    assert mutated_decision["accepted"] is False
    assert mutated_decision["conclusion"] == "UNKNOWN"


def test_carcara_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "import time\ntime.sleep(5)",
    )
    _install_runtime_environment(monkeypatch, executable)
    # Leave enough time for the interpreter to start and create the marker;
    # timeout still occurs well before the five-second worker sleep.
    monkeypatch.setattr("jacobian_checkers.smt.CARCARA_TIMEOUT_SECONDS", 1.0)

    decision = check_unsat_proof(proof_request_factory())

    assert marker.read_text(encoding="utf-8") == "called"
    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
