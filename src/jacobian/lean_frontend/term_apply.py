"""Bounded Lean term application to an immutable proof-state artifact.

``lean.term.apply`` applies one bounded Lean term to a replayable proof state
by elaborating ``exact <term>`` through the maintained Lean REPL tactic
protocol. It reuses :class:`LeanProofStateAdapter` so that term application and
tactic application share one clean-replay path, one immutable successor-state
artifact type, one resource-bound enforcement, and one fail-closed boundary.
Strategy stays out of the contract: the adapter only validates the term and
delegates; it never selects terms, ranks successors, or prescribes proof
strategy.
"""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.lean import LeanDiagnosticPhase, LeanDiagnosticSource
from jacobian.contracts.lean_exploration import LeanProofStateRequest
from jacobian.contracts.lean_term_apply import (
    LeanTermApplyOutput,
    LeanTermApplyRequest,
)
from jacobian.lean_frontend.exploration import _FORBIDDEN
from jacobian.lean_frontend.proof_state import LeanProofStateAdapter

_EXACT_PREFIX = "exact "


class LeanTermApplyAdapter:
    def __init__(
        self,
        proof_state_adapter: LeanProofStateAdapter,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self._proof_state = proof_state_adapter
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.term.apply",
            version="2",
            title="Elaborate one Lean proof term against a proof state",
            description=(
                "Apply one bounded Lean term to an immutable replayable proof "
                "state by elaborating `exact <term>` in a clean pinned Lean "
                "process. Returns typed goals and an immutable successor state, or "
                "term-elaboration diagnostics with term-relative source spans. "
                "Timeout and operational errors fail closed."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=LeanTermApplyRequest.model_json_schema(),
            output_schema=LeanTermApplyOutput.model_json_schema(),
            tags=(
                "lean",
                "term",
                "proof-term",
                "term-elaboration",
                "proof-state",
                "goals",
                "exploration",
                "proof-repair",
                "diagnostics",
                "type-mismatch",
                "source-span",
            ),
            accepted_input_kinds=(
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(proof_state_adapter.resources.state_schema_uri,),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="close_true_with_exact_true_intro",
                    description=(
                        "Apply the term `True.intro` to a replayable proof "
                        "state for True via `exact True.intro`; a completed "
                        "transition still requires lean.check."
                    ),
                    input=LeanTermApplyRequest.model_validate(
                        {
                            "environment": "CORE",
                            "statement": "True",
                            "term": "True.intro",
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanTermApplyRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_TERM_APPLY_REQUEST",
                    stage="request_validation",
                    message="The Lean term-application request is invalid.",
                    hint=(
                        "Provide one bounded Lean term expression without "
                        "newlines, `:=`, or forbidden commands."
                    ),
                )
            ) from exc
        if _FORBIDDEN.search(validated.term):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_TERM_APPLY_REQUEST",
                    stage="request_validation",
                    message=("The term contains a forbidden Lean command or syntax."),
                    hint=(
                        "Provide a term that does not use sorry, admit, "
                        "set_option, unsafe, or other forbidden commands."
                    ),
                )
            )
        delegated = self._proof_state.apply(
            LeanProofStateRequest(
                state_uri=validated.state_uri,
                environment=validated.environment,
                statement=validated.statement,
                proof_prefix=validated.proof_prefix,
                tactic=_EXACT_PREFIX + validated.term,
                max_goals=validated.max_goals,
                max_local_declarations=validated.max_local_declarations,
                max_rendered_bytes=validated.max_rendered_bytes,
            ),
            diagnostic_phase=LeanDiagnosticPhase.TERM_ELABORATION,
            diagnostic_source=LeanDiagnosticSource.TERM,
            diagnostic_column_offset=len(_EXACT_PREFIX),
        )
        try:
            output = LeanTermApplyOutput.model_validate(
                {
                    **delegated.output.model_dump(mode="python"),
                    "term_apply_uri": delegated.output.transition_uri,
                    "term_application": "LEAN_EXACT_ELABORATION",
                }
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_TERM_APPLY_OUTPUT_INVALID",
                    stage="term_application",
                    message=(
                        "The delegated proof-state transition did not produce "
                        "a valid term-application output."
                    ),
                    hint="Retry the term application or report this transition.",
                )
            ) from exc
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=delegated.execution,
            output=output.model_dump(mode="json"),
            artifact_uris=delegated.artifact_uris,
        )


def install_lean_term_apply_capability(
    proof_state_adapter: LeanProofStateAdapter,
    provider_runtime: CapabilityProviderRuntime,
) -> LeanTermApplyAdapter:
    return LeanTermApplyAdapter(proof_state_adapter, provider_runtime)


__all__ = ["LeanTermApplyAdapter", "install_lean_term_apply_capability"]
