from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import python_distribution_provider_runtime
from jacobian.registry import (
    CheckerCompatibilityError,
    CheckerExecutableChangedError,
    CheckerNotFoundError,
    CheckerRegistry,
    CheckerRegistryError,
    CheckerRevokedError,
)
from jacobian.storage.repository import ArtifactRepository

CLAIM_SCHEMA_A = "artifact://sha256/" + "a" * 64
CLAIM_SCHEMA_B = "artifact://sha256/" + "b" * 64


def test_checker_selection_errors_explain_recovery(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)
    selection = {
        "evidence_kind": "WITNESS",
        "format_id": "example.witness",
        "format_version": "1",
        "claim_schema_uri": CLAIM_SCHEMA_A,
        "semantics_uri": CLAIM_SCHEMA_A,
        "candidate_schema_uri": CLAIM_SCHEMA_A,
    }

    with pytest.raises(
        CheckerNotFoundError,
        match="authorize a compatible checker before retrying",
    ):
        registry.select_compatible(**selection)

    for name in ("reject-all-v1", "reject-all-v2"):
        registry.authorize(
            name=name,
            entrypoint="jacobian_checkers.reject:check",
            evidence_kind="WITNESS",
            format_id="example.witness",
            format_version="1",
            claim_schema_uris=(CLAIM_SCHEMA_A,),
            semantics_uris=(CLAIM_SCHEMA_A,),
            candidate_schema_uris=(CLAIM_SCHEMA_A,),
        )

    with pytest.raises(
        CheckerCompatibilityError,
        match="Configure checker policy to select exactly one",
    ):
        registry.select_compatible(**selection)


def test_revoked_checker_cannot_authorize_new_verification(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)
    checker = registry.authorize(
        name="reject-all-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
    )

    assert registry.require_active(checker.checker_id) == checker

    registry.revoke(checker.checker_id, reason="test revocation")

    with pytest.raises(
        CheckerRevokedError,
        match="Select an active checker from the reference contract",
    ):
        registry.require_active(checker.checker_id)
    assert [event.action for event in registry.audit_log(checker.checker_id)] == [
        "AUTHORIZED",
        "REVOKED",
    ]


def test_catalog_binding_uses_persisted_checker_identity_without_remeasurement(
    tmp_path: Path,
) -> None:
    registry = CheckerRegistry(ArtifactRepository(tmp_path))
    checker = registry.authorize(
        name="reject-all-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
    )

    assert (
        registry.require_catalog_binding(
            checker.checker_id,
            implementation_digest=checker.implementation_digest,
        )
        == checker
    )
    with pytest.raises(CheckerExecutableChangedError, match="jacobian update"):
        registry.require_catalog_binding(
            checker.checker_id,
            implementation_digest="sha256:" + "0" * 64,
        )


def test_checker_policy_lock_must_precede_store_transaction(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)
    checker = registry.authorize(
        name="reject-all-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
    )

    with (
        store.transaction(),
        pytest.raises(
            CheckerRegistryError,
            match="policy must be locked before the store transaction",
        ),
    ):
        registry.revoke(checker.checker_id, reason="wrong lock order")

    with (
        store.transaction(),
        pytest.raises(
            CheckerRegistryError,
            match="policy must be locked before the store transaction",
        ),
        registry.verification_guard(
            checker.checker_id,
            expected_implementation_digest=checker.implementation_digest,
        ),
    ):
        pass

    second_registry = CheckerRegistry(store)
    with (
        store.transaction(),
        pytest.raises(
            CheckerRegistryError,
            match="policy must be locked before the store transaction",
        ),
    ):
        second_registry.revoke(checker.checker_id, reason="wrong lock order")

    with registry.policy_transaction(), store.transaction():
        registry.revoke(checker.checker_id, reason="ordered policy change")

    with pytest.raises(CheckerRevokedError):
        registry.require_active(checker.checker_id)


def test_concurrent_duplicate_authorize_is_serialized(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)
    barrier = threading.Barrier(8)
    registrations = []
    errors: list[Exception] = []

    def authorize() -> None:
        barrier.wait()
        try:
            registrations.append(
                registry.authorize(
                    name="reject-all-v1",
                    entrypoint="jacobian_checkers.reject:check",
                    evidence_kind="WITNESS",
                    format_id="example.witness",
                    format_version="1",
                    claim_schema_uris=(CLAIM_SCHEMA_A,),
                    semantics_uris=(CLAIM_SCHEMA_A,),
                    candidate_schema_uris=(CLAIM_SCHEMA_A,),
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=authorize) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len({registration.checker_id for registration in registrations}) == 1
    assert [
        event.action for event in registry.audit_log(registrations[0].checker_id)
    ] == ["AUTHORIZED"]


@pytest.mark.parametrize("corruption", ["registration", "digest_column"])
def test_checker_registry_rejects_identity_metadata_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)
    checker = registry.authorize(
        name="reject-all-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
    )

    with sqlite3.connect(store.db_path) as connection:
        if corruption == "registration":
            tampered = checker.model_dump(mode="json")
            tampered["claim_schema_uris"] = [CLAIM_SCHEMA_A, CLAIM_SCHEMA_B]
            connection.execute(
                """
                UPDATE checkers
                SET registration_json = ?
                WHERE checker_id = ?
                """,
                (canonicalize_json(tampered), checker.checker_id),
            )
        else:
            connection.execute(
                """
                UPDATE checkers
                    SET implementation_digest = ?
                WHERE checker_id = ?
                """,
                ("sha256:" + "0" * 64, checker.checker_id),
            )

    with pytest.raises(CheckerRegistryError, match="Checker registry data"):
        registry.get(checker.checker_id)


@pytest.mark.parametrize(
    ("evidence_kind", "claim_schemas", "semantics", "candidate_schemas", "targets"),
    [
        (EvidenceKind.WITNESS, (), (CLAIM_SCHEMA_A,), (CLAIM_SCHEMA_A,), ()),
        (EvidenceKind.WITNESS, (CLAIM_SCHEMA_A,), (), (CLAIM_SCHEMA_A,), ()),
        (EvidenceKind.WITNESS, (CLAIM_SCHEMA_A,), (CLAIM_SCHEMA_A,), (), ()),
    ],
)
def test_checker_authorization_requires_explicit_compatibility_scope(
    tmp_path: Path,
    evidence_kind: EvidenceKind,
    claim_schemas: tuple[str, ...],
    semantics: tuple[str, ...],
    candidate_schemas: tuple[str, ...],
    targets: tuple[str, ...],
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)

    with pytest.raises(CheckerRegistryError, match="Supply"):
        registry.authorize(
            name="reject-all-v1",
            entrypoint="jacobian_checkers.reject:check",
            evidence_kind=evidence_kind,
            format_id="example.witness",
            format_version="1",
            claim_schema_uris=claim_schemas,
            semantics_uris=semantics,
            candidate_schema_uris=candidate_schemas,
            target_schema_uris=targets,
            target_semantics_uris=targets,
        )


def test_checker_selection_uses_authorized_external_runtime_identity(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "external-checker"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    runtime = ProviderObservation(
        provider="external-checker",
        availability=ProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
        digest_kind=ProviderDigestKind.EXECUTABLE,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T2,
        license_id="MIT",
        configuration={"executable": str(executable.resolve())},
    )
    store = ArtifactRepository(tmp_path / "store")
    registry = CheckerRegistry(store)
    checker = registry.authorize(
        name="externally-backed-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
        provider_runtime=runtime,
    )

    assert (
        registry.require_active(checker.checker_id).implementation.provider_runtime
        == runtime
    )

    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    executable.chmod(0o755)
    selected = registry.select_compatible(
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uri=CLAIM_SCHEMA_A,
        semantics_uri=CLAIM_SCHEMA_A,
        candidate_schema_uri=CLAIM_SCHEMA_A,
    )

    assert selected == checker


def test_checker_registry_authorizes_python_distribution_runtime(
    tmp_path: Path,
) -> None:
    runtime = python_distribution_provider_runtime(
        "pydantic",
        distribution_name="pydantic",
        import_name="pydantic",
        required_attributes=("BaseModel",),
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        configuration={"import_name": "pydantic"},
    )
    assert runtime.availability is ProviderAvailability.AVAILABLE
    store = ArtifactRepository(tmp_path)
    registry = CheckerRegistry(store)

    checker = registry.authorize(
        name="distribution-backed-v1",
        entrypoint="jacobian_checkers.reject:check",
        evidence_kind="WITNESS",
        format_id="example.witness",
        format_version="1",
        claim_schema_uris=(CLAIM_SCHEMA_A,),
        semantics_uris=(CLAIM_SCHEMA_A,),
        candidate_schema_uris=(CLAIM_SCHEMA_A,),
        provider_runtime=runtime,
    )

    assert (
        registry.require_active(checker.checker_id).implementation.provider_runtime
        == runtime
    )
