from pathlib import Path

import pytest

from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.registry import CheckerRegistry
from jacobian.storage.repository import ArtifactRepository

_CLAIM_SCHEMA_URI = "artifact://sha256/" + "1" * 64
_SEMANTICS_URI = "artifact://sha256/" + "2" * 64
_CANDIDATE_SCHEMA_URI = "artifact://sha256/" + "3" * 64


def _operation() -> CheckerOperation:
    return CheckerOperation(
        name="exact rational determinant test checker",
        entrypoint=(
            "jacobian_checkers.rational_determinants:check_rational_determinant"
        ),
        evidence_kind=EvidenceKind.WITNESS,
        format_id="matrix.rational_determinant",
        format_version="1",
        claim_schema_uris=(_CLAIM_SCHEMA_URI,),
        semantics_uris=(_SEMANTICS_URI,),
        candidate_schema_uris=(_CANDIDATE_SCHEMA_URI,),
        reason="test operator authorization",
    )


def test_disabled_checker_operation_remains_unauthorized(
    tmp_path: Path,
) -> None:
    store = ArtifactRepository(tmp_path / "store")
    installed = authorize_checker_operation(
        CheckerRegistry(store),
        _operation(),
        authorize=False,
    )

    assert not installed.authorized
    assert installed.checker_id is None
    with pytest.raises(ValueError, match="not authorized"):
        installed.require_checker_id()


def test_checker_authorization_uses_exact_declared_scope(tmp_path: Path) -> None:
    registry = CheckerRegistry(ArtifactRepository(tmp_path / "store"))
    installed = authorize_checker_operation(registry, _operation(), authorize=True)

    checker_id = installed.require_checker_id()
    registration = registry.get(checker_id)
    assert installed.authorized
    assert registration.evidence_kind is EvidenceKind.WITNESS
    assert registration.format_id == "matrix.rational_determinant"
    assert registration.claim_schema_uris == (_CLAIM_SCHEMA_URI,)
    assert registration.semantics_uris == (_SEMANTICS_URI,)
    assert registration.candidate_schema_uris == (_CANDIDATE_SCHEMA_URI,)


def test_checker_operation_rejects_unbound_authorization() -> None:
    with pytest.raises(ValueError, match="claim schema"):
        CheckerOperation(
            name="unbound checker",
            entrypoint="package.module:check",
            evidence_kind=EvidenceKind.WITNESS,
            format_id="example",
            format_version="1",
            claim_schema_uris=(),
            semantics_uris=(_SEMANTICS_URI,),
            candidate_schema_uris=(_CANDIDATE_SCHEMA_URI,),
            reason="test",
        )
