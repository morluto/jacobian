from __future__ import annotations

import base64
import hashlib

import pytest
from pydantic import ValidationError

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.smt import (
    SmtAletheProofArtifact,
    SmtExplorationBudget,
    SmtProblemArtifact,
    SmtProblemBinding,
    SmtUnsatProofFindOutput,
    SmtUnsatProofVerificationOutput,
)

_PROBLEM_URI = "artifact://sha256/" + "1" * 64
_PROOF_URI = "artifact://sha256/" + "2" * 64
_DIGEST = "sha256:" + "3" * 64
_SMTLIB = (
    "(set-logic QF_UF)\n"
    "(declare-sort U 0)\n"
    "(declare-fun a () U)\n"
    "(assert (not (= a a)))\n"
    "(check-sat)\n"
)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _runtime() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="cvc5",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1.3.4",
        digest=_DIGEST,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="BSD-3-Clause",
        license_files=("cvc5.dist-info/licenses/COPYING",),
        features=("alethe-proof-production",),
        configuration={
            "profile": "jacobian.smtlib2.qf-unsat/v1",
            "proof_format": "cvc5.alethe/1.3.4",
        },
    )


def _problem() -> SmtProblemArtifact:
    return SmtProblemArtifact.from_text(logic="QF_UF", smtlib_text=_SMTLIB)


def _binding() -> SmtProblemBinding:
    problem = _problem()
    return SmtProblemBinding(
        problem_artifact_uri=_PROBLEM_URI,
        problem_object_digest=_DIGEST,
        problem_payload_digest=_DIGEST,
        logic=problem.logic,
        profile=problem.profile,
        input_language=problem.input_language,
        smtlib_digest=problem.smtlib_digest,
    )


def test_problem_contract_pins_one_exact_quantifier_free_query() -> None:
    problem = _problem()

    assert problem.logic == "QF_UF"
    assert problem.smtlib_digest == _sha256(_SMTLIB.encode("ascii"))
    assert problem.query_scope == "SINGLE_CHECK_SAT"

    for invalid in (
        _SMTLIB.replace("QF_UF", "ALL"),
        _SMTLIB.replace("(check-sat)\n", "(check-sat)\n(check-sat)\n"),
        _SMTLIB.replace("(check-sat)\n", "(push 1)\n(check-sat)\n"),
        _SMTLIB.replace("(check-sat)\n", "(get-model)\n"),
        _SMTLIB.replace("(check-sat)\n", "(check-sat)\r\n"),
    ):
        with pytest.raises(ValidationError):
            SmtProblemArtifact.from_text(logic="QF_UF", smtlib_text=invalid)


def test_synchronous_smt_budget_stays_below_remote_deadline() -> None:
    assert SmtExplorationBudget(wall_seconds=150).wall_seconds == 150
    with pytest.raises(ValidationError):
        SmtExplorationBudget(wall_seconds=151)


def test_command_scanner_ignores_comments_strings_and_quoted_symbols() -> None:
    text = (
        "(set-logic QF_UF)\n"
        "; (push 1) is only a comment\n"
        "(declare-fun |semi;paren()| () String)\n"
        '(assert (= |semi;paren()| "literal (check-sat) ; text"))\n'
        "(check-sat)\n"
    )

    problem = SmtProblemArtifact.from_text(logic="QF_UF", smtlib_text=text)

    assert problem.logic == "QF_UF"


def test_proof_contract_preserves_bytes_and_marks_holes_without_verifying() -> None:
    proof = b'(\n(step t0 (cl) :rule hole :args ("untranslated rewrite"))\n)\n'

    artifact = SmtAletheProofArtifact.from_bytes(
        problem=_binding(),
        proof=proof,
        producer=_runtime(),
        resource_budget=SmtExplorationBudget(wall_seconds=5).artifact_budget(),
    )

    assert artifact.raw_bytes() == proof
    assert artifact.proof_digest == _sha256(proof)
    assert artifact.alethe_hole_count == 1
    assert artifact.contains_holes is True

    payload = artifact.model_dump(mode="json")
    payload["alethe_hole_count"] = 0
    with pytest.raises(ValidationError):
        SmtAletheProofArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["proof_base64"] = base64.b64encode(proof + b" ").decode("ascii")
    with pytest.raises(ValidationError):
        SmtAletheProofArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["producer"]["digest_kind"] = "SOURCE_TREE"
    with pytest.raises(ValidationError):
        SmtAletheProofArtifact.model_validate(payload)


def test_producer_output_does_not_project_a_mathematical_conclusion() -> None:
    output = SmtUnsatProofFindOutput(
        status="PROOF_PRODUCED",
        solver_status="UNSATISFIABLE",
        problem_uri=_PROBLEM_URI,
        proof_uri=_PROOF_URI,
        contains_holes=True,
        alethe_hole_count=2,
        detail="raw proof evidence only",
    )

    assert "conclusion" not in output.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SmtUnsatProofFindOutput(
            status="NO_PROOF_PRODUCED",
            solver_status="UNSATISFIABLE",
            problem_uri=_PROBLEM_URI,
            proof_uri=_PROOF_URI,
            contains_holes=False,
            alethe_hole_count=0,
            detail="inconsistent",
        )


def test_only_independently_verified_smt_unsat_output_carries_true() -> None:
    record_uri = "artifact://sha256/" + "4" * 64
    checker_id = "checker://sha256/" + "5" * 64
    certificate_uri = "artifact://sha256/" + "6" * 64

    verified = SmtUnsatProofVerificationOutput(
        status="VERIFIED_UNSAT",
        conclusion="TRUE",
        problem_uri=_PROBLEM_URI,
        proof_uri=_PROOF_URI,
        certificate_uri=certificate_uri,
        checker_id=checker_id,
        verification_record_uri=record_uri,
        detail="strict Carcara replay accepted the exact proof",
    )

    assert verified.verification_record_uri == record_uri
    with pytest.raises(ValidationError):
        SmtUnsatProofVerificationOutput(
            status="REJECTED",
            conclusion="TRUE",
            problem_uri=_PROBLEM_URI,
            proof_uri=_PROOF_URI,
            certificate_uri=certificate_uri,
            checker_id=checker_id,
            verification_record_uri=None,
            detail="Carcara reported holey",
        )
    with pytest.raises(ValidationError):
        SmtUnsatProofVerificationOutput(
            status="VERIFIED_UNSAT",
            conclusion="TRUE",
            problem_uri=_PROBLEM_URI,
            proof_uri=_PROOF_URI,
            certificate_uri=certificate_uri,
            checker_id=checker_id,
            verification_record_uri=None,
            detail="missing verification record",
        )
