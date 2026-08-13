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
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.contracts.sat import SatResourceBudget
from jacobian.sat_smt.sat import install_sat_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian_checkers.sat import check_unsat_proof

_DEFAULT_PROOF = b"-1 0\n0\n"
ProofRequestFactory = Callable[[bytes], dict[str, Any]]


def _producer() -> ProviderObservation:
    return ProviderObservation(
        provider="cadical",
        availability=ProviderAvailability.AVAILABLE,
        version="3.0.1",
        digest="sha256:" + "d" * 64,
        digest_kind=ProviderDigestKind.EXECUTABLE,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T2,
        license_id="MIT",
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
    sat = install_sat_artifacts(store, schemas, artifacts)
    cnf_result = sat.put_cnf(
        variable_names=("x", "y"),
        clauses=((1, 2), (-1, 2), (1, -2), (-1, -2)),
    )
    cnf = store.get(cnf_result.artifact_uri)
    semantics = store.get(sat.installation.semantics_uri)
    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )

    def build(proof_bytes: bytes) -> dict[str, Any]:
        proof_result = sat.put_proof(
            cnf_uri=cnf_result.artifact_uri,
            proof=proof_bytes,
            producer=_producer(),
            resource_budget=SatResourceBudget(wall_seconds=5),
        )
        proof = store.get(proof_result.artifact_uri)
        bindings = EvidenceBindings(
            claim_digest=cnf.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=proof.manifest.object_digest,
        )
        payload = {
            "cnf_uri": cnf.artifact_uri,
            "proof_uri": proof.artifact_uri,
        }
        certificate = CertificateEnvelope(
            certificate_type="sat.unsat-proof",
            format_version="1",
            bindings=bindings,
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
            ),
            payload=payload,
        )
        certificate_artifact = artifacts.put(
            schema_uri=certificate_schema_uri,
            semantics_uri=sat.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(cnf.artifact_uri, proof.artifact_uri),
            summary="SAT UNSAT proof certificate",
        )
        stored_certificate = store.get(certificate_artifact.artifact_uri)
        return {
            "request_version": "1",
            "claim": _checker_artifact(cnf),
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
    marker = tmp_path / "called"
    executable = tmp_path / "drat-trim"
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


def test_checker_reconstructs_exact_dimacs_and_forces_ascii_drat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "cnf = pathlib.Path(sys.argv[1]).read_bytes()\n"
        "proof = pathlib.Path(sys.argv[2]).read_bytes()\n"
        "assert cnf == b'p cnf 2 4\\n-1 -2 0\\n-1 2 0\\n1 -2 0\\n1 2 0\\n'\n"
        "assert proof.startswith(b'c jacobian drat-text/v1 force-ascii ')\n"
        "assert proof.endswith(b'-1 0\\n0\\n')\n"
        "print('s VERIFIED')\n"
        "raise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, executable)

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert marker.read_text(encoding="utf-8") == "called"
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["method"] == "CHECKED_CERTIFICATE"
    assert decision["coverage"] == "NOT_APPLICABLE"


def test_checker_accepts_cleanup_deletions_after_the_empty_clause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, _marker = _fake_checker(
        tmp_path,
        "print('s VERIFIED')\nraise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, executable)

    decision = check_unsat_proof(proof_request_factory(b"0\nd 1 0\nd -1 2 0\n"))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_reports_strict_deletion_warning_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, _marker = _fake_checker(
        tmp_path,
        "print('c WARNING: deleted clause on proof line 4 does not occur: [0] -2 0')\n"
        "raise SystemExit(80)",
    )
    _install_runtime_environment(monkeypatch, executable)

    decision = check_unsat_proof(proof_request_factory(b"d -2 0\n-1 0\n0\n"))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
    assert decision["detail"].startswith("DRAT_DELETION_WARNING_REJECTED:")


def test_binding_and_lineage_attacks_are_rejected_before_drat_trim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "raise AssertionError('must not run')",
    )
    _install_runtime_environment(monkeypatch, executable)
    original = proof_request_factory(_DEFAULT_PROOF)
    mutations: list[dict[str, Any]] = []

    changed = deepcopy(original)
    changed["claim"]["payload"]["clauses"] = list(
        reversed(changed["claim"]["payload"]["clauses"])
    )
    mutations.append(changed)

    changed = deepcopy(original)
    changed["candidate"]["payload"]["cnf"]["dimacs_digest"] = "sha256:" + "a" * 64
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


def test_malformed_or_concatenated_proof_is_rejected_before_drat_trim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "raise AssertionError('must not run')",
    )
    _install_runtime_environment(monkeypatch, executable)

    for proof_bytes in (
        b"-1 0\n0\n1 0\n",
        b"1 -1 0\n0\n",
        b"1 1 0\n0\n",
        b"1 0 2\n",
        b"\xff 0\n",
    ):
        decision = check_unsat_proof(proof_request_factory(proof_bytes))

        assert decision["accepted"] is False, proof_bytes
        assert decision["conclusion"] == "UNKNOWN", proof_bytes
    assert not marker.exists()


@pytest.mark.parametrize(
    "body",
    [
        "print('s NOT VERIFIED')\nraise SystemExit(0)",
        "print('s VERIFIED')\nraise SystemExit(1)",
        "print('s VERIFIED')\nprint('s VERIFIED')\nraise SystemExit(0)",
    ],
)
def test_only_one_verified_status_with_zero_exit_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, _marker = _fake_checker(tmp_path, body)
    _install_runtime_environment(monkeypatch, executable)

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_excessive_checker_output_is_rejected_without_reading_it_into_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, _marker = _fake_checker(
        tmp_path,
        "print('x' * 4096)\nraise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, executable)
    monkeypatch.setattr("jacobian_checkers.sat.DRAT_TRIM_OUTPUT_LIMIT", 128)

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_runtime_digest_mismatch_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    executable, marker = _fake_checker(
        tmp_path,
        "print('s VERIFIED')\nraise SystemExit(0)",
    )
    _install_runtime_environment(monkeypatch, executable)
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + "a" * 64,
    )

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
    assert not marker.exists()


def test_missing_runtime_authorization_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    proof_request_factory: ProofRequestFactory,
) -> None:
    monkeypatch.delenv("JACOBIAN_CHECKER_EXECUTABLE", raising=False)
    monkeypatch.delenv("JACOBIAN_CHECKER_RUNTIME_DIGEST", raising=False)

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_drat_timeout_fails_closed(
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
    monkeypatch.setattr("jacobian_checkers.sat.DRAT_TRIM_TIMEOUT_SECONDS", 1.0)

    decision = check_unsat_proof(proof_request_factory(_DEFAULT_PROOF))

    assert marker.read_text(encoding="utf-8") == "called"
    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
