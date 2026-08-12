"""Contract tests for non-authoritative Lean axiom inspection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.lean_proof_axioms import (
    LeanProofAxiomsArtifact,
    LeanProofAxiomsInspectRequest,
)

DIGEST = "sha256:" + "a" * 64


def _artifact(**overrides: object) -> LeanProofAxiomsArtifact:
    values: dict[str, object] = {
        "environment": "CORE",
        "environment_digest": DIGEST,
        "provider_runtime_digest": DIGEST,
        "lean_version": "4.31.0",
        "lean_commit": "lean-commit",
        "imports": ("Init.Prelude",),
        "statement": "True",
        "proof": "by trivial",
        "elaborated": True,
        "inspection_complete": True,
        "axioms_reported": True,
        "axioms": (),
        "sorry_count": 0,
        "admit_count": 0,
        "diagnostics": (),
    }
    values.update(overrides)
    return LeanProofAxiomsArtifact(**values)


def test_request_accepts_bounded_statement_and_proof() -> None:
    request = LeanProofAxiomsInspectRequest(
        statement="True",
        proof="by trivial",
    )

    assert request.environment.value == "CORE"


@pytest.mark.parametrize("statement", ["theorem t : True := trivial", "True\n"])
def test_request_rejects_declarations_and_multiline_statements(
    statement: str,
) -> None:
    with pytest.raises(ValidationError):
        LeanProofAxiomsInspectRequest(statement=statement, proof="by trivial")


def test_request_rejects_comment_injection() -> None:
    with pytest.raises(ValidationError, match="comments"):
        LeanProofAxiomsInspectRequest(
            statement="True",
            proof="by trivial -- hide another command",
        )


def test_artifact_accepts_complete_axiom_dependency_facts() -> None:
    artifact = _artifact()

    assert artifact.semantic_scope == "AXIOM_DEPENDENCY_ONLY"


def test_artifact_rejects_complete_inspection_without_axiom_report() -> None:
    with pytest.raises(ValidationError, match="axiom closure"):
        _artifact(axioms_reported=False, inspection_complete=True)


def test_artifact_rejects_unsorted_axioms() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        _artifact(axioms=("propext", "Classical.choice"))
