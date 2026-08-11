"""Greedy preservation-checked shrinking for pure-data artifacts."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.claims import ClaimValidationService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.checkers import EvidenceKind
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
from jacobian.contracts.shrinking import (
    Minimality,
    PluginReductionResponse,
    ShrinkResult,
    ShrinkStep,
    ShrinkTargetKind,
)
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.registry import CheckerRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _ShrinkState:
    """Mutable progress for one bounded shrinking operation."""

    kind: ShrinkTargetKind
    target_uri: str
    claim_uri: str
    preservation_checker_id: str
    claim: StoredArtifact
    initial: StoredArtifact
    current: StoredArtifact
    manifest: Any
    reducer_capability: Any
    expected_schema: str
    preservation_format: str
    semantics_digest: str
    requested_reducers: tuple[str, ...]
    requested_objectives: tuple[str, ...]
    evaluation_budget: int
    reducer_timeout_seconds: int
    proposal_validator: Callable[[str, Any, Any], None] | None
    evaluations: int = 0
    steps: list[ShrinkStep] = field(default_factory=list)
    operational_failure: tuple[ExecutionStatus, str] | None = None
    current_objectives: dict[str, Any] = field(default_factory=dict)
    expected_objective_values: tuple[Fraction, ...] | None = None
    checked_boundary_rejection: bool = False


class ShrinkService:
    """Reduce evidence while independently replaying every accepted step."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        verification: VerificationService,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.verification = verification

    def _reject_invalid_run_request(
        self,
        *,
        kind: ShrinkTargetKind,
        target_uri: str,
        evaluation_budget: int,
        requested_reducers: tuple[str, ...],
        requested_objectives: tuple[str, ...],
        claim_uri: str,
        plugin_id: str,
    ) -> ShrinkResult | None:
        if evaluation_budget < 1:
            return self._rejected(
                kind=kind,
                target_uri=target_uri,
                detail=(
                    "The shrinking evaluation budget must be at least 1. Increase "
                    "evaluation_budget, then retry."
                ),
            )
        if not requested_reducers:
            return self._rejected(
                kind=kind,
                target_uri=target_uri,
                detail=(
                    "No reducer was requested. Choose at least one reducer from the "
                    "reference contract, then retry."
                ),
            )
        if not requested_objectives:
            return self._rejected(
                kind=kind,
                target_uri=target_uri,
                detail=(
                    "No shrinking objective was requested. Provide at least one "
                    "ordered objective from the reference contract, then retry."
                ),
            )
        validation = self.claims.validate(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
        )
        if not validation.valid:
            return self._rejected(
                kind=kind,
                target_uri=target_uri,
                detail="; ".join(validation.input.errors),
            )
        return None

    def _load_shrink_context(
        self,
        *,
        kind: ShrinkTargetKind,
        target_uri: str,
        claim_uri: str,
        plugin_id: str,
        preservation_checker_id: str,
    ) -> tuple[Any, Any, Any, Any, str, str, str] | ShrinkResult:
        try:
            claim = self.store.get(claim_uri)
            initial = self.store.get(target_uri)
            manifest = self.plugins.get(plugin_id)
            reducer_capability = self.plugins.resolve(
                plugin_id,
                CapabilityName.REDUCER,
            )
            if kind == ShrinkTargetKind.CANDIDATE:
                expected_schema = manifest.candidate_schema_uri
            else:
                if initial.manifest.schema_uri not in manifest.witness_schema_uris:
                    raise ValueError(
                        "witness target schema is not declared by the plugin"
                    )
                expected_schema = initial.manifest.schema_uri
            if initial.manifest.schema_uri != expected_schema:
                raise ValueError("target schema does not match plugin target schema")
            if initial.manifest.semantics_uri != manifest.semantics_uri:
                raise ValueError("target semantics do not match plugin semantics")
            checker = self.verification.checker_registry.require_active(
                preservation_checker_id
            )
            if checker.evidence_kind != EvidenceKind.PRESERVATION:
                raise ValueError("selected checker is not a preservation checker")
            preservation_format = checker.format_id
            semantics_digest = self.store.get(
                manifest.semantics_uri
            ).manifest.object_digest
        except (
            StorageError,
            PluginRegistryError,
            CheckerRegistryError,
            ValueError,
        ) as exc:
            return self._rejected(
                kind=kind,
                target_uri=target_uri,
                detail=_shrink_failure_detail(exc),
            )
        return (
            claim,
            initial,
            manifest,
            reducer_capability,
            expected_schema,
            preservation_format,
            semantics_digest,
        )

    def run(
        self,
        *,
        target_kind: ShrinkTargetKind | str,
        target_uri: str,
        claim_uri: str,
        plugin_id: str,
        preservation_checker_id: str,
        reducers: tuple[str, ...] | list[str],
        objectives: tuple[str, ...] | list[str],
        evaluation_budget: int,
        reducer_timeout_seconds: int = 30,
        proposal_validator: Callable[[str, Any, Any], None] | None = None,
    ) -> ShrinkResult:
        """Run bounded shrinking and report the achieved minimality level."""

        kind = ShrinkTargetKind(target_kind)
        requested_reducers = tuple(reducers)
        requested_objectives = tuple(objectives)
        early = self._reject_invalid_run_request(
            kind=kind,
            target_uri=target_uri,
            evaluation_budget=evaluation_budget,
            requested_reducers=requested_reducers,
            requested_objectives=requested_objectives,
            claim_uri=claim_uri,
            plugin_id=plugin_id,
        )
        if early is not None:
            return early
        loaded = self._load_shrink_context(
            kind=kind,
            target_uri=target_uri,
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            preservation_checker_id=preservation_checker_id,
        )
        if isinstance(loaded, ShrinkResult):
            return loaded
        (
            claim,
            initial,
            manifest,
            reducer_capability,
            expected_schema,
            preservation_format,
            semantics_digest,
        ) = loaded

        state = _ShrinkState(
            kind=kind,
            target_uri=target_uri,
            claim_uri=claim_uri,
            preservation_checker_id=preservation_checker_id,
            claim=claim,
            initial=initial,
            current=initial,
            manifest=manifest,
            reducer_capability=reducer_capability,
            expected_schema=expected_schema,
            preservation_format=preservation_format,
            semantics_digest=semantics_digest,
            requested_reducers=requested_reducers,
            requested_objectives=requested_objectives,
            evaluation_budget=evaluation_budget,
            reducer_timeout_seconds=reducer_timeout_seconds,
            proposal_validator=proposal_validator,
        )
        while state.evaluations < evaluation_budget and self._run_round(state):
            pass
        return self._finalize(state, evaluation_budget)

    def _run_round(self, state: _ShrinkState) -> bool:
        """Run one reducer response and accept at most one proposal."""

        state.checked_boundary_rejection = False
        execution = self.executor.run(
            entrypoint=state.reducer_capability.descriptor.entrypoint,
            implementation_digest=state.reducer_capability.implementation_digest,
            request={
                "request_version": "1",
                "target_kind": state.kind.value,
                "target": state.current.payload,
                "claim": state.claim.payload,
                "reducers": list(state.requested_reducers),
                "objectives": list(state.requested_objectives),
                "bindings": {
                    "claim_digest": state.claim.manifest.object_digest,
                    "target_digest": state.current.manifest.object_digest,
                    "semantics_digest": state.semantics_digest,
                },
            },
            timeout_seconds=state.reducer_timeout_seconds,
        )
        if execution.status != ExecutionStatus.COMPLETED:
            state.operational_failure = (
                execution.status,
                execution.detail or "reducer execution failed",
            )
            return False
        try:
            response = PluginReductionResponse.model_validate(execution.output)
            declared_values = _ordered_objective_values(
                response.current_objectives,
                state.requested_objectives,
            )
            if (
                state.expected_objective_values is not None
                and declared_values != state.expected_objective_values
            ):
                raise ValueError(
                    "reducer changed the declared objectives for the current target"
                )
            state.current_objectives = dict(response.current_objectives)
        except ValidationError as exc:
            _LOGGER.warning("reducer returned an invalid response", exc_info=exc)
            state.operational_failure = (
                ExecutionStatus.ERROR,
                (
                    "The reducer returned an invalid response. Check the "
                    "reference contract and inspect the local plugin log."
                ),
            )
            return False
        except ValueError as exc:
            _LOGGER.warning("reducer objective validation failed", exc_info=exc)
            state.operational_failure = (
                ExecutionStatus.ERROR,
                _shrink_failure_detail(exc),
            )
            return False
        if not response.reductions:
            return False

        for proposal in response.reductions:
            if state.evaluations >= state.evaluation_budget:
                break
            state.evaluations += 1
            if self._evaluate_proposal(state, proposal, declared_values):
                return True
        return False

    def _evaluate_proposal(
        self,
        state: _ShrinkState,
        proposal: Any,
        declared_values: tuple[Fraction, ...],
    ) -> bool:
        if proposal.reducer not in state.requested_reducers:
            state.steps.append(
                ShrinkStep(
                    index=len(state.steps),
                    reducer=proposal.reducer,
                    from_uri=state.current.artifact_uri,
                    accepted=False,
                    execution_status=ExecutionStatus.COMPLETED,
                    input_status=InputStatus.REJECTED,
                    objectives=proposal.objectives,
                    detail="plugin used a reducer that was not requested",
                )
            )
            return False
        try:
            proposed_values = _ordered_objective_values(
                proposal.objectives,
                state.requested_objectives,
            )
            if proposed_values >= declared_values:
                raise ValueError(
                    "proposal does not strictly improve the ordered objectives"
                )
            proposed = self._materialize_proposal(
                reducer=proposal.reducer,
                payload=proposal.payload,
                current=state.current,
                expected_schema=state.expected_schema,
                semantics_uri=state.manifest.semantics_uri,
                proposal_validator=state.proposal_validator,
            )
            decision = self.verification.verify_preservation(
                claim_uri=state.claim_uri,
                original_uri=state.current.artifact_uri,
                reduced_uri=proposed.artifact_uri,
                checker_id=state.preservation_checker_id,
                preservation_format=state.preservation_format,
                reducer=proposal.reducer,
            )
            accepted = decision.assurance.verification == Verification.VERIFIED
            state.steps.append(
                ShrinkStep(
                    index=len(state.steps),
                    reducer=proposal.reducer,
                    from_uri=state.current.artifact_uri,
                    proposed_uri=proposed.artifact_uri,
                    accepted=accepted,
                    execution_status=decision.execution.status,
                    input_status=decision.input.status,
                    verification_record_uri=(
                        decision.verification_record_uri if accepted else None
                    ),
                    objectives=proposal.objectives,
                    detail=(
                        "preservation verified"
                        if accepted
                        else "; ".join(decision.input.errors)
                    ),
                )
            )
            if accepted:
                state.current = self.store.get(proposed.artifact_uri)
                state.current_objectives = dict(proposal.objectives)
                state.expected_objective_values = proposed_values
                state.checked_boundary_rejection = False
                return True
            state.checked_boundary_rejection = (
                decision.execution.status is ExecutionStatus.COMPLETED
                and decision.input.status is InputStatus.REJECTED
            )
        except (StorageError, SchemaRegistryError, ValueError) as exc:
            state.steps.append(
                ShrinkStep(
                    index=len(state.steps),
                    reducer=proposal.reducer,
                    from_uri=state.current.artifact_uri,
                    accepted=False,
                    execution_status=ExecutionStatus.ERROR,
                    input_status=InputStatus.REJECTED,
                    objectives=proposal.objectives,
                    detail=_shrink_failure_detail(exc),
                )
            )
        return False

    def _finalize(self, state: _ShrinkState, evaluation_budget: int) -> ShrinkResult:
        if state.operational_failure is not None:
            status, detail = state.operational_failure
            final_result = _unverified_result(
                status=status,
                detail=detail,
                claim_digest=state.claim.manifest.object_digest,
                semantics_digest=state.semantics_digest,
                candidate_digest=state.current.manifest.object_digest,
            )
        elif state.evaluations < evaluation_budget:
            state.evaluations += 1
            final_result = self.verification.verify_preservation(
                claim_uri=state.claim_uri,
                original_uri=state.current.artifact_uri,
                reduced_uri=state.current.artifact_uri,
                checker_id=state.preservation_checker_id,
                preservation_format=state.preservation_format,
                reducer="identity-final-verification",
            )
        else:
            final_result = _unverified_result(
                status=ExecutionStatus.COMPLETED,
                detail="budget ended before fresh final verification",
                claim_digest=state.claim.manifest.object_digest,
                semantics_digest=state.semantics_digest,
                candidate_digest=state.current.manifest.object_digest,
            )
        verified_final = final_result.assurance.verification == Verification.VERIFIED
        minimality = (
            Minimality.LOCAL
            if (
                state.current.artifact_uri != state.initial.artifact_uri
                and verified_final
                and state.checked_boundary_rejection
            )
            else Minimality.NONE
        )
        return ShrinkResult(
            execution=Execution(
                status=(
                    state.operational_failure[0]
                    if state.operational_failure is not None
                    else ExecutionStatus.COMPLETED
                ),
                detail=(
                    state.operational_failure[1]
                    if state.operational_failure is not None
                    else None
                ),
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            result=final_result,
            target_kind=state.kind,
            initial_target_uri=state.target_uri,
            final_target_uri=state.current.artifact_uri,
            minimality=minimality,
            evaluations=state.evaluations,
            steps=tuple(state.steps),
            objectives=state.current_objectives,
        )

    def _materialize_proposal(
        self,
        *,
        reducer: str,
        payload: Any,
        current: StoredArtifact,
        expected_schema: str,
        semantics_uri: str,
        proposal_validator: Callable[[str, Any, Any], None] | None,
    ) -> ArtifactPutResult:
        normalized = self.schemas.validate(expected_schema, payload)
        if proposal_validator is not None:
            proposal_validator(reducer, current.payload, normalized)
        proposed = self.store.put(
            schema_uri=expected_schema,
            semantics_uri=semantics_uri,
            payload=normalized,
            parents=(current.artifact_uri,),
            summary=f"shrink proposal: {reducer}",
        )
        if proposed.artifact_uri == current.artifact_uri:
            raise ValueError("proposal does not change the target")
        return proposed

    @staticmethod
    def _rejected(
        *,
        kind: ShrinkTargetKind,
        target_uri: str,
        detail: str,
    ) -> ShrinkResult:
        result = _unverified_result(
            status=ExecutionStatus.COMPLETED,
            detail=detail,
        )
        return ShrinkResult(
            execution=Execution(status=ExecutionStatus.COMPLETED),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(detail,),
            ),
            result=result,
            target_kind=kind,
            initial_target_uri=target_uri,
            final_target_uri=target_uri,
            minimality=Minimality.NONE,
            evaluations=0,
            objectives={},
        )


def _shrink_failure_detail(exc: Exception) -> str:
    if isinstance(exc, (ValueError, CheckerRegistryError)):
        return str(exc)
    _LOGGER.warning("shrinking step failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "A required shrinking artifact is unavailable. Check the artifact URIs "
            "and state directory, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The reducer plugin is unavailable. Call math.find, choose an "
            "installed reference domain, and retry."
        )
    return (
        "The reduction does not match the target schema. Check the reference "
        "contract and retry with a valid reduction."
    )


def _unverified_result(
    *,
    status: ExecutionStatus,
    detail: str,
    claim_digest: str | None = None,
    semantics_digest: str | None = None,
    candidate_digest: str | None = None,
) -> ResultEnvelope:
    return ResultEnvelope(
        execution=Execution(status=status, detail=detail),
        input=InputValidation(
            status=(
                InputStatus.ACCEPTED
                if status != ExecutionStatus.COMPLETED or not detail
                else InputStatus.REJECTED
            ),
            errors=(
                (detail,) if status == ExecutionStatus.COMPLETED and detail else ()
            ),
        ),
        conclusion=Conclusion.UNKNOWN,
        assurance=Assurance(
            arithmetic=Arithmetic.SYMBOLIC,
            method=Method.BOUNDED_SEARCH,
            coverage=Coverage.BOUNDED,
            verification=Verification.UNVERIFIED,
        ),
        claim_digest=claim_digest,
        semantics_digest=semantics_digest,
        candidate_digest=candidate_digest,
    )


_INTEGER_OBJECTIVE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


def _ordered_objective_values(
    objectives: dict[str, Any],
    ordering: tuple[str, ...],
) -> tuple[Fraction, ...]:
    if set(objectives) != set(ordering):
        raise ValueError(
            "objective map must contain exactly the requested objective names"
        )
    return tuple(_objective_value(objectives[name]) for name in ordering)


def _objective_value(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("boolean objective values are not ordered numbers")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str) and _INTEGER_OBJECTIVE.fullmatch(value):
        return Fraction(parse_canonical_integer(value))
    if isinstance(value, dict) and set(value) == {"num", "den"}:
        numerator = value.get("num")
        denominator = value.get("den")
        if (
            isinstance(numerator, str)
            and isinstance(denominator, str)
            and _INTEGER_OBJECTIVE.fullmatch(numerator)
            and _INTEGER_OBJECTIVE.fullmatch(denominator)
        ):
            result = Fraction(
                parse_canonical_integer(numerator),
                parse_canonical_integer(denominator),
            )
            if value == {
                "num": format_canonical_integer(result.numerator),
                "den": format_canonical_integer(result.denominator),
            }:
                return result
    raise ValueError("objective values must be canonical exact numbers")
