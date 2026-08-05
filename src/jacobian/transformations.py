"""Untrusted representation proposals with explicit proof obligations."""

from __future__ import annotations

import hashlib
import logging
import time

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
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
from jacobian.contracts.transformations import (
    PluginTransformationResponse,
    TransformationApplyResult,
    TransformationBindings,
    TransformationClaim,
    TransformationEnvelope,
    TransformationRelation,
)
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class TransformationService:
    """Apply untrusted domain transformations and persist their obligations."""

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
        self.semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.transformation",
            version="1",
            definition={
                "description": (
                    "a proposed relation between two explicitly bound representations"
                )
            },
        )
        self.claim_schema_uri = schemas.register(
            name="jacobian.transformation-claim",
            version="1",
            schema=model_schema(TransformationClaim),
        )
        self.envelope_schema_uri = schemas.register(
            name="jacobian.transformation-envelope",
            version="1",
            schema=model_schema(TransformationEnvelope),
        )

    def apply(
        self,
        *,
        source_uri: str,
        plugin_id: str,
        target_schema_uri: str,
        target_semantics_uri: str,
        requested_relation: TransformationRelation | str,
        wall_seconds: int,
    ) -> TransformationApplyResult:
        """Run a transformer; the returned relation remains unverified."""

        started = time.monotonic()
        relation = TransformationRelation(requested_relation)
        try:
            if wall_seconds < 1:
                raise ValueError(
                    "The transformation time budget must be at least 1 second. Set "
                    "wall_seconds to 1 or more, then retry."
                )
            source = self.store.get(source_uri)
            manifest = self.plugins.get(plugin_id)
            if source.manifest.schema_uri != manifest.candidate_schema_uri:
                raise ValueError(
                    "The source artifact uses the wrong schema for this plugin. Use "
                    "an artifact from the same reference domain, then retry."
                )
            if source.manifest.semantics_uri != manifest.semantics_uri:
                raise ValueError(
                    "The source artifact and plugin use different semantics. Choose "
                    "the plugin from the source artifact's reference domain, then "
                    "retry."
                )
            self.schemas.validate(source.manifest.schema_uri, source.payload)
            self.store.get_descriptor(target_schema_uri, expected_kind="schema")
            self.store.get_descriptor(
                target_semantics_uri,
                expected_kind="semantics",
            )
            transformer = self.plugins.resolve(
                plugin_id,
                CapabilityName.TRANSFORMER,
            )
            execution = self.executor.run(
                entrypoint=transformer.descriptor.entrypoint,
                implementation_digest=transformer.implementation_digest,
                request={
                    "request_version": "1",
                    "source": source.payload,
                    "requested_relation": relation.value,
                    "target_schema_uri": target_schema_uri,
                    "target_semantics_uri": target_semantics_uri,
                },
                timeout_seconds=wall_seconds,
            )
            if execution.status != ExecutionStatus.COMPLETED:
                return self._operational_failure(
                    source_uri,
                    started,
                    execution.status,
                    execution.detail or "transformer failed",
                )
            response = PluginTransformationResponse.model_validate(execution.output)
            if response.relation != relation:
                raise ValueError(
                    "transformer relation differs from the requested relation"
                )
            self.schemas.validate(target_schema_uri, response.target_payload)
            target = self.store.put(
                schema_uri=target_schema_uri,
                semantics_uri=target_semantics_uri,
                payload=response.target_payload,
                parents=(source_uri,),
                summary="untrusted transformed representation",
            )
            source_semantics_digest = self.store.get(
                source.manifest.semantics_uri
            ).manifest.object_digest
            target_semantics_digest = self.store.get(
                target_semantics_uri
            ).manifest.object_digest
            bindings = TransformationBindings(
                source_digest=source.manifest.object_digest,
                source_schema_uri=source.manifest.schema_uri,
                source_semantics_digest=source_semantics_digest,
                target_digest=target.object_digest,
                target_schema_uri=target_schema_uri,
                target_semantics_digest=target_semantics_digest,
            )
            claim = self.store.put(
                schema_uri=self.claim_schema_uri,
                semantics_uri=self.semantics_uri,
                payload=TransformationClaim(
                    transform_format=response.transform_format,
                    format_version=response.format_version,
                    relation=relation,
                    bindings=bindings,
                ).model_dump(mode="json"),
                parents=(source_uri, target.artifact_uri),
                summary="transformation proof obligation",
            )
            obligation_digest = (
                "sha256:"
                + hashlib.sha256(canonicalize_json(response.obligation)).hexdigest()
            )
            envelope = TransformationEnvelope(
                claim_uri=claim.artifact_uri,
                source_uri=source_uri,
                target_uri=target.artifact_uri,
                transform_format=response.transform_format,
                format_version=response.format_version,
                relation=relation,
                bindings=bindings,
                transformer_digest=transformer.implementation_digest,
                obligation_digest=obligation_digest,
                obligation=response.obligation,
            )
            stored_envelope = self.store.put(
                schema_uri=self.envelope_schema_uri,
                semantics_uri=self.semantics_uri,
                payload=envelope.model_dump(mode="json"),
                parents=(
                    claim.artifact_uri,
                    source_uri,
                    target.artifact_uri,
                    plugin_id,
                ),
                summary="unverified representation transformation",
            )
            return TransformationApplyResult(
                source_uri=source_uri,
                target_uri=target.artifact_uri,
                claim_uri=claim.artifact_uri,
                transformation_uri=stored_envelope.artifact_uri,
                relation=relation,
                result=ResultEnvelope(
                    execution=Execution(
                        status=ExecutionStatus.COMPLETED,
                        runtime_ms=int((time.monotonic() - started) * 1000),
                    ),
                    input=InputValidation(status=InputStatus.ACCEPTED),
                    conclusion=Conclusion.TRUE,
                    assurance=Assurance(
                        arithmetic=Arithmetic.SYMBOLIC,
                        method=Method.HEURISTIC,
                        coverage=Coverage.NOT_APPLICABLE,
                        verification=Verification.UNVERIFIED,
                    ),
                    claim_digest=claim.object_digest,
                    semantics_digest=self.store.get(
                        self.semantics_uri
                    ).manifest.object_digest,
                    candidate_digest=source.manifest.object_digest,
                    evidence_uris=(stored_envelope.artifact_uri,),
                ),
            )
        except (
            PluginRegistryError,
            SchemaRegistryError,
            StorageError,
            ValidationError,
            ValueError,
        ) as exc:
            return self._rejected(
                source_uri,
                started,
                _transformation_failure_detail(exc),
            )

    @staticmethod
    def _rejected(
        source_uri: str,
        started: float,
        detail: str,
    ) -> TransformationApplyResult:
        return TransformationApplyResult(
            source_uri=source_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                ),
                input=InputValidation(
                    status=InputStatus.REJECTED,
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
        source_uri: str,
        started: float,
        status: ExecutionStatus,
        detail: str,
    ) -> TransformationApplyResult:
        return TransformationApplyResult(
            source_uri=source_uri,
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
                    method=Method.HEURISTIC,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
            ),
        )


def _transformation_failure_detail(exc: Exception) -> str:
    if isinstance(exc, ValueError) and not isinstance(
        exc,
        (PluginRegistryError, SchemaRegistryError, StorageError, ValidationError),
    ):
        return str(exc)
    _LOGGER.warning("representation transformation failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "A required source or target artifact is unavailable. Check the "
            "artifact URIs, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The transformer plugin is unavailable. Call math.find, "
            "choose an installed reference domain, and retry."
        )
    return (
        "The source artifact or transformer response is invalid. Check the "
        "reference contract and retry."
    )
