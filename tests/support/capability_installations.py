"""Focused capability installation support shared by owning test lanes."""

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
def install_capability_bundle(
    tmp_path: Path,
    installer: Callable[..., tuple[Any, Any]],
) -> Iterator[tuple[Any, Any, ArtifactRepository]]:
    """Build and close a minimal store around one installed capability bundle."""

    store = ArtifactRepository(tmp_path / "store")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    checkers = CheckerRegistry(store)
    verification = VerificationService(store, checkers)
    adapters, installed = installer(
        store,
        schemas,
        artifacts,
        verification,
        checkers,
        authorize_checker=True,
    )
    try:
        yield adapters, installed, store
    finally:
        store.close()
