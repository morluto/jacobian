"""Focused operation binding support shared by owning test lanes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


@contextmanager
def bind_operation_group(
    tmp_path: Path,
    binder: Callable[..., tuple[Any, Any]],
) -> Iterator[tuple[Any, Any, ArtifactRepository]]:
    """Build and close a minimal store around one bound operation group."""

    store = ArtifactRepository(tmp_path / "store")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    checkers = CheckerRegistry(store)
    verification = VerificationService(store, checkers, schemas)
    adapters, bound = binder(
        store,
        schemas,
        artifacts,
        verification,
        checkers,
        authorize_checker=True,
    )
    try:
        yield adapters, bound, store
    finally:
        store.close()
