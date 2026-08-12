"""Guarded durable publication for accepted verification records."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.registry import CheckerRegistry
from jacobian.storage.repository import ArtifactRepository


class VerificationRecordCommitter:
    """Commit a verification record only while its checker remains authorized."""

    def __init__(
        self,
        store: ArtifactRepository,
        checker_registry: CheckerRegistry,
    ) -> None:
        self.store = store
        self.checker_registry = checker_registry

    def commit(
        self,
        *,
        checker_id: str,
        implementation_digest: str,
        schema_uri: str,
        semantics_uri: str,
        payload: dict[str, Any],
        parents: tuple[str, ...],
        summary: str,
    ) -> ArtifactPutResult:
        """Atomically re-check authorization and persist the record."""

        with self.checker_registry.verification_guard(
            checker_id,
            expected_implementation_digest=implementation_digest,
        ):
            return self.store.put(
                schema_uri=schema_uri,
                semantics_uri=semantics_uri,
                payload=payload,
                parents=parents,
                summary=summary,
            )
