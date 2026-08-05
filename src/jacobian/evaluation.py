"""Batched, explicitly unverified candidate evaluation."""

from __future__ import annotations

import hashlib
import logging
import platform
import time
from collections.abc import Sequence

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.claims import ClaimValidationService
from jacobian.contracts.evaluation import (
    EvaluationBatchResult,
    EvaluationItem,
    EvaluationProfile,
    PluginEvaluationResponse,
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
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class EvaluationService:
    """Dispatch installed evaluators without granting certification authority."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        *,
        max_batch_size: int = 256,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.max_batch_size = max_batch_size

    def evaluate_batch(
        self,
        *,
        claim_uri: str,
        candidate_uris: tuple[str, ...] | list[str],
        plugin_id: str,
        profile: EvaluationProfile | str,
        seed: int,
        wall_seconds: float,
    ) -> EvaluationBatchResult:
        """Evaluate candidates without promoting any plugin result to verified."""

        selected_profile = EvaluationProfile(profile)
        candidates = tuple(candidate_uris)
        if not candidates:
            return self._rejected_batch(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                profile=selected_profile,
                seed=seed,
                error="candidate batch cannot be empty",
            )
        if len(candidates) > self.max_batch_size:
            return self._rejected_batch(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                profile=selected_profile,
                seed=seed,
                error="candidate batch exceeds the configured limit",
            )
        if wall_seconds < 1:
            return self._rejected_batch(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                profile=selected_profile,
                seed=seed,
                error=(
                    "The evaluation time budget must be at least 1 second. Set "
                    "wall_seconds to 1 or more, then retry."
                ),
            )

        validation = self.claims.validate(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
        )
        if not validation.valid:
            return self._rejected_batch(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                profile=selected_profile,
                seed=seed,
                error="; ".join(validation.input.errors),
            )
        try:
            claim = self.store.get(claim_uri)
            manifest = self.plugins.get(plugin_id)
            evaluator = self.plugins.resolve(
                plugin_id,
                CapabilityName.EVALUATOR,
            )
            semantics_digest = self.store.get(
                manifest.semantics_uri
            ).manifest.object_digest
        except (PluginRegistryError, StorageError) as exc:
            return self._rejected_batch(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                profile=selected_profile,
                seed=seed,
                error=_evaluation_failure_detail(exc),
            )

        started = time.monotonic()
        items: list[EvaluationItem] = []
        for candidate_uri in candidates:
            remaining_seconds = wall_seconds - (time.monotonic() - started)
            if remaining_seconds <= 0:
                items.append(
                    EvaluationItem(
                        candidate_uri=candidate_uri,
                        result=ResultEnvelope(
                            execution=Execution(
                                status=ExecutionStatus.TIMEOUT,
                                detail=(
                                    "The evaluation batch reached its time budget. "
                                    "Retry with fewer candidates or a larger "
                                    "wall_seconds budget."
                                ),
                            ),
                            input=InputValidation(status=InputStatus.ACCEPTED),
                            conclusion=Conclusion.UNKNOWN,
                            assurance=Assurance(
                                arithmetic=Arithmetic.SYMBOLIC,
                                method=Method.BOUNDED_SEARCH,
                                coverage=Coverage.BOUNDED,
                                verification=Verification.UNVERIFIED,
                            ),
                            claim_digest=claim.manifest.object_digest,
                            semantics_digest=semantics_digest,
                        ),
                        detail=(
                            "The evaluation batch reached its time budget. Retry "
                            "with fewer candidates or a larger wall_seconds budget."
                        ),
                    )
                )
                continue
            items.append(
                self._evaluate_one(
                    claim=claim,
                    candidate_uri=candidate_uri,
                    candidate_schema_uri=manifest.candidate_schema_uri,
                    semantics_uri=manifest.semantics_uri,
                    semantics_digest=semantics_digest,
                    entrypoint=evaluator.descriptor.entrypoint,
                    implementation_digest=evaluator.implementation_digest,
                    profile=selected_profile,
                    seed=seed,
                    wall_seconds=remaining_seconds,
                )
            )
        return EvaluationBatchResult(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=int((time.monotonic() - started) * 1000),
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            profile=selected_profile,
            seed=seed,
            evaluator_digest=evaluator.implementation_digest,
            environment_digest=_evaluation_environment_digest(
                evaluator.implementation_digest
            ),
            items=tuple(items),
        )

    def _evaluate_one(
        self,
        *,
        claim: StoredArtifact,
        candidate_uri: str,
        candidate_schema_uri: str,
        semantics_uri: str,
        semantics_digest: str,
        entrypoint: str,
        implementation_digest: str,
        profile: EvaluationProfile,
        seed: int,
        wall_seconds: float,
    ) -> EvaluationItem:
        try:
            candidate = self.store.get(candidate_uri)
            if candidate.manifest.schema_uri != candidate_schema_uri:
                raise ValueError(
                    "The candidate uses the wrong schema for this plugin. Use a "
                    "candidate returned by the same reference domain, then retry."
                )
            if candidate.manifest.semantics_uri != semantics_uri:
                raise ValueError(
                    "The candidate and plugin use different semantics. Choose the "
                    "plugin from the candidate's reference domain, then retry."
                )
            self.schemas.validate(candidate_schema_uri, candidate.payload)
            request = {
                "request_version": "1",
                "profile": profile.value,
                "seed": seed,
                "claim": claim.payload,
                "candidate": candidate.payload,
                "bindings": {
                    "claim_digest": claim.manifest.object_digest,
                    "candidate_digest": candidate.manifest.object_digest,
                    "semantics_digest": semantics_digest,
                },
            }
            execution = self.executor.run(
                entrypoint=entrypoint,
                implementation_digest=implementation_digest,
                request=request,
                timeout_seconds=wall_seconds,
            )
            if execution.status != ExecutionStatus.COMPLETED:
                return EvaluationItem(
                    candidate_uri=candidate_uri,
                    result=ResultEnvelope(
                        execution=Execution(
                            status=execution.status,
                            runtime_ms=execution.runtime_ms,
                            detail=execution.detail,
                        ),
                        input=InputValidation(status=InputStatus.ACCEPTED),
                        conclusion=Conclusion.UNKNOWN,
                        assurance=Assurance(
                            arithmetic=Arithmetic.SYMBOLIC,
                            method=Method.BOUNDED_SEARCH,
                            coverage=Coverage.BOUNDED,
                            verification=Verification.UNVERIFIED,
                        ),
                        claim_digest=claim.manifest.object_digest,
                        semantics_digest=semantics_digest,
                        candidate_digest=candidate.manifest.object_digest,
                    ),
                    detail=execution.detail or "",
                )
            response = PluginEvaluationResponse.model_validate(execution.output)
            result = ResultEnvelope(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=execution.runtime_ms,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=response.conclusion,
                assurance=Assurance(
                    arithmetic=response.arithmetic,
                    method=response.method,
                    coverage=response.coverage,
                    verification=Verification.UNVERIFIED,
                ),
                claim_digest=claim.manifest.object_digest,
                semantics_digest=semantics_digest,
                candidate_digest=candidate.manifest.object_digest,
            )
            return EvaluationItem(
                candidate_uri=candidate_uri,
                result=result,
                objectives=response.objectives,
                features=response.features,
                failure_classifications=response.failure_classifications,
                detail=response.detail,
            )
        except (
            StorageError,
            SchemaRegistryError,
            ValidationError,
            ValueError,
        ) as exc:
            detail = _evaluation_failure_detail(exc)
            return EvaluationItem(
                candidate_uri=candidate_uri,
                result=ResultEnvelope(
                    execution=Execution(status=ExecutionStatus.COMPLETED),
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
                detail=detail,
            )

    @staticmethod
    def _rejected_batch(
        *,
        claim_uri: str,
        plugin_id: str,
        profile: EvaluationProfile,
        seed: int,
        error: str,
    ) -> EvaluationBatchResult:
        return EvaluationBatchResult(
            execution=Execution(status=ExecutionStatus.COMPLETED),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(error,),
            ),
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            profile=profile,
            seed=seed,
        )


def require_complete_evaluation_batch(
    evaluation: EvaluationBatchResult,
    candidate_uris: Sequence[str],
) -> None:
    """Reject a batch that does not cover the requested candidates in order."""

    if (
        evaluation.input.status is not InputStatus.ACCEPTED
        or len(evaluation.items) != len(candidate_uris)
        or tuple(item.candidate_uri for item in evaluation.items)
        != tuple(candidate_uris)
    ):
        detail = (
            "; ".join(evaluation.input.errors)
            or "evaluation did not cover the selected candidates"
        )
        raise ValueError(detail)


def _evaluation_failure_detail(exc: Exception) -> str:
    if isinstance(exc, ValueError) and not isinstance(
        exc,
        (PluginRegistryError, SchemaRegistryError, StorageError, ValidationError),
    ):
        return str(exc)
    _LOGGER.warning("candidate evaluation failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "A required claim or candidate artifact is unavailable. Check the "
            "artifact URIs, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The evaluator plugin is unavailable. Call math.find, choose "
            "an installed reference domain, and retry."
        )
    return (
        "The claim, candidate, or evaluator response is invalid. Check the reference "
        "contract and retry with matching artifacts."
    )


def _evaluation_environment_digest(evaluator_digest: str) -> str:
    payload = canonicalize_json(
        {
            "environment_version": "1",
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "evaluator_digest": evaluator_digest,
        }
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()
