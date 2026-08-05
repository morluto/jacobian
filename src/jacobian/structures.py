"""Domain-dispatched canonicalization for finite search structures."""

from __future__ import annotations

import hashlib
import logging
import time

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.discovery import (
    PluginCanonicalizationResponse,
    StructureCanonicalizationResult,
)
from jacobian.contracts.plugins import CapabilityName
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class StructureService:
    """Compute untrusted canonical forms for search deduplication."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        executor: PluginExecutor,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.executor = executor

    def canonicalize(
        self,
        *,
        structure_uri: str,
        plugin_id: str,
        wall_seconds: int,
    ) -> StructureCanonicalizationResult:
        """Canonicalize one installed-domain object without asserting truth."""

        started = time.monotonic()
        if wall_seconds < 1:
            return self._failure(
                structure_uri,
                started,
                InputStatus.REJECTED,
                (
                    "The canonicalization time budget must be at least 1 second. "
                    "Set wall_seconds to 1 or more, then retry."
                ),
            )
        try:
            structure = self.store.get(structure_uri)
            manifest = self.plugins.get(plugin_id)
            if structure.manifest.schema_uri != manifest.candidate_schema_uri:
                raise ValueError(
                    "The structure uses the wrong schema for this plugin. Use a "
                    "structure from the same reference domain, then retry."
                )
            if structure.manifest.semantics_uri != manifest.semantics_uri:
                raise ValueError(
                    "The structure and plugin use different semantics. Choose the "
                    "plugin from the structure's reference domain, then retry."
                )
            self.schemas.validate(
                structure.manifest.schema_uri,
                structure.payload,
            )
            capability = self.plugins.resolve(
                plugin_id,
                CapabilityName.CANONICALIZER,
            )
            execution = self.executor.run(
                entrypoint=capability.descriptor.entrypoint,
                implementation_digest=capability.implementation_digest,
                request={
                    "request_version": "1",
                    "structure": structure.payload,
                },
                timeout_seconds=wall_seconds,
            )
            if execution.status != ExecutionStatus.COMPLETED:
                return self._operational_failure(
                    structure_uri,
                    started,
                    execution.status,
                    execution.detail or "canonicalizer failed",
                )
            response = PluginCanonicalizationResponse.model_validate(execution.output)
            self.schemas.validate(
                structure.manifest.schema_uri,
                response.canonical_payload,
            )
            canonical = self.store.put(
                schema_uri=structure.manifest.schema_uri,
                semantics_uri=structure.manifest.semantics_uri,
                payload=response.canonical_payload,
                summary="untrusted canonical structure",
            )
            canonical_key = (
                "sha256:"
                + hashlib.sha256(
                    canonicalize_json(
                        {
                            "canonical_object_digest": canonical.object_digest,
                            "canonicalizer_digest": capability.implementation_digest,
                        }
                    )
                ).hexdigest()
            )
            return StructureCanonicalizationResult(
                structure_uri=structure_uri,
                canonical_uri=canonical.artifact_uri,
                canonical_key=canonical_key,
                canonicalizer_digest=capability.implementation_digest,
                mapping=response.mapping,
                automorphism_group_order=response.automorphism_group_order,
                orbits=response.orbits,
                result=ResultEnvelope(
                    execution=Execution(
                        status=ExecutionStatus.COMPLETED,
                        runtime_ms=int((time.monotonic() - started) * 1000),
                    ),
                    input=InputValidation(status=InputStatus.ACCEPTED),
                    conclusion=Conclusion.NOT_APPLICABLE,
                    assurance=Assurance(
                        arithmetic=Arithmetic.SYMBOLIC,
                        method=Method.BOUNDED_SEARCH,
                        coverage=Coverage.NOT_APPLICABLE,
                        verification=Verification.UNVERIFIED,
                    ),
                    semantics_digest=self.store.get(
                        manifest.semantics_uri
                    ).manifest.object_digest,
                    candidate_digest=structure.manifest.object_digest,
                    evidence_uris=(canonical.artifact_uri,),
                ),
            )
        except (
            PluginRegistryError,
            SchemaRegistryError,
            StorageError,
            ValidationError,
            ValueError,
        ) as exc:
            return self._failure(
                structure_uri,
                started,
                InputStatus.REJECTED,
                _structure_failure_detail(exc),
            )

    @staticmethod
    def _failure(
        structure_uri: str,
        started: float,
        input_status: InputStatus,
        detail: str,
    ) -> StructureCanonicalizationResult:
        return StructureCanonicalizationResult(
            structure_uri=structure_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                ),
                input=InputValidation(
                    status=input_status,
                    errors=(detail,),
                ),
                conclusion=Conclusion.UNKNOWN,
                assurance=Assurance(
                    arithmetic=Arithmetic.SYMBOLIC,
                    method=Method.HEURISTIC,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
            ),
        )

    @staticmethod
    def _operational_failure(
        structure_uri: str,
        started: float,
        status: ExecutionStatus,
        detail: str,
    ) -> StructureCanonicalizationResult:
        return StructureCanonicalizationResult(
            structure_uri=structure_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=status,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                    detail=detail,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=Conclusion.UNKNOWN,
                assurance=Assurance(
                    arithmetic=Arithmetic.SYMBOLIC,
                    method=Method.BOUNDED_SEARCH,
                    coverage=Coverage.BOUNDED,
                    verification=Verification.UNVERIFIED,
                ),
            ),
        )


def _structure_failure_detail(exc: Exception) -> str:
    if isinstance(exc, ValueError) and not isinstance(
        exc,
        (PluginRegistryError, SchemaRegistryError, StorageError, ValidationError),
    ):
        return str(exc)
    _LOGGER.warning("structure canonicalization failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "The structure artifact is unavailable. Check its artifact URI, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The canonicalizer plugin is unavailable. Call math.find, "
            "choose an installed reference domain, and retry."
        )
    return (
        "The structure or canonicalizer response is invalid. Check the reference "
        "contract and retry."
    )
