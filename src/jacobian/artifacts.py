"""Validated public artifact operations."""

from __future__ import annotations

import logging
from typing import Any

from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    SchemaValidationError,
)
from jacobian.storage.errors import (
    ArtifactNotFoundError,
    StorageError,
    StorageLimitError,
)
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class ArtifactValidationError(ValueError):
    """Artifact input failed schema, semantics, or canonical validation."""


class ArtifactService:
    """Validate domain payloads before committing immutable artifacts."""

    def __init__(self, store: ArtifactRepository, schemas: SchemaRegistry) -> None:
        self.store = store
        self.schemas = schemas

    def put(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        payload: Any,
        parents: tuple[str, ...] | list[str] = (),
        summary: str = "",
        producer_write: bool = False,
    ) -> ArtifactPutResult:
        """Validate and store one artifact under its schema and semantics."""

        if self.schemas.is_producer_only(schema_uri) and not producer_write:
            raise ArtifactValidationError(
                "This artifact schema is producer-only. Invoke its owning "
                "operation instead of writing the artifact directly."
            )
        try:
            normalized = self.schemas.validate(schema_uri, payload)
            self.store.get_descriptor(
                semantics_uri,
                expected_kind="semantics",
            )
        except (SchemaRegistryError, StorageError, ValueError) as exc:
            _LOGGER.warning("artifact input validation failed", exc_info=exc)
            if isinstance(exc, (StorageError, SchemaRegistryError)) and not isinstance(
                exc,
                SchemaValidationError,
            ):
                detail = (
                    "The schema or semantics descriptor is unavailable. Check both "
                    "descriptor URIs, then retry."
                )
            elif isinstance(exc, SchemaValidationError):
                if exc.required_field is not None:
                    field = exc.required_field
                    detail = (
                        f"The artifact payload is missing required field {field!r}. "
                        "Add it using the reference contract, then retry."
                    )
                else:
                    detail = (
                        "The artifact payload does not match its schema. Check the "
                        "reference contract and retry with matching input."
                    )
            else:
                detail = (
                    "The artifact payload does not match its schema or semantics. "
                    "Check the reference contract and retry with matching input."
                )
            raise ArtifactValidationError(detail) from exc
        if len(set(parents)) != len(parents):
            raise ArtifactValidationError(
                "Artifact parents must be unique. Remove duplicate parent URIs, "
                "then retry."
            )
        try:
            return self.store.put(
                schema_uri=schema_uri,
                semantics_uri=semantics_uri,
                payload=normalized,
                parents=parents,
                summary=summary,
            )
        except ArtifactNotFoundError as exc:
            _LOGGER.warning("artifact parent is unavailable", exc_info=exc)
            raise ArtifactValidationError(
                "A parent artifact is unavailable. Check each parent URI, then retry."
            ) from exc
        except StorageLimitError as exc:
            _LOGGER.warning("artifact storage limit reached", exc_info=exc)
            raise ArtifactValidationError(
                "The artifact exceeds a configured storage limit. Reduce its payload, "
                "summary, or parent count, then retry."
            ) from exc
        except StorageError as exc:
            _LOGGER.warning("artifact persistence failed", exc_info=exc)
            raise ArtifactValidationError(
                "Jacobian could not save the artifact. Check the state directory and "
                "available disk space, then retry."
            ) from exc
