"""Bounded SymPy producer for typed polynomial-expression normalization."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.polynomial_expressions import (
    SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION,
    PolynomialExpansionTermBudgetError,
    PolynomialExpressionNormalizeOutput,
    PolynomialExpressionNormalizeRequest,
)
from jacobian.contracts.polynomials import SparseRationalPolynomial
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.provider_runtime import SYMPY_VERSION
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)
from jacobian.schema_registry import model_schema
from jacobian.sympy_polynomial_protocol import (
    make_sympy_polynomial_worker_request,
    parse_sympy_polynomial_worker_response,
)
from jacobian.worker_environment import worker_environment

SYMPY_NORMALIZATION_STDOUT_LIMIT = 2_000_000
SYMPY_NORMALIZATION_STDERR_LIMIT = 64_000


@dataclass(frozen=True, slots=True)
class _SympyNormalizationRun:
    execution_status: ExecutionStatus
    runtime_ms: int
    normalized: SparseRationalPolynomial | None = None
    detail: str | None = None


def install_sympy_polynomial_normalization_capability(
    expressions: PolynomialExpressionArtifactService,
    runtime: CapabilityProviderRuntime,
) -> CapabilityAdapter:
    """Install the producer only for the exact supported SymPy profile."""

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.provider != "jacobian.sympy"
        or runtime.version != SYMPY_VERSION
        or runtime.configuration != SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION
    ):
        raise ValueError("the pinned SymPy polynomial-normalization runtime is absent")
    return SympyPolynomialExpressionNormalizeAdapter(
        expressions=expressions,
        runtime=runtime,
    )


class _SympyPolynomialNormalizationBackend:
    def __init__(self, runtime: CapabilityProviderRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        request: PolynomialExpressionNormalizeRequest,
    ) -> _SympyNormalizationRun:
        started = time.monotonic()
        if (
            sympy_polynomial_normalization_provider_runtime(refresh=True)
            != self.runtime
        ):
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed SymPy runtime no longer matches the capability "
                    "descriptor; no normalization evidence was retained."
                ),
            )
        worker_request = make_sympy_polynomial_worker_request(request.expression)
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=(
                    "-I",
                    "-m",
                    "jacobian.sympy_polynomial_worker",
                ),
                stdin_bytes=canonicalize_json(worker_request.model_dump(mode="json")),
                timeout_seconds=float(request.resource_budget.wall_seconds),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                stdout_limit_bytes=SYMPY_NORMALIZATION_STDOUT_LIMIT,
                stderr_limit_bytes=SYMPY_NORMALIZATION_STDERR_LIMIT,
            )
        )
        if completed.termination is ProcessTermination.START_FAILED:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                "The isolated SymPy normalization worker could not be started.",
            )
        operational = _operational_failure(started, completed)
        if operational is not None:
            return operational
        try:
            normalized = _parse_worker_output(
                completed.stdout,
                variables=request.expression.variables,
            )
        except (UnicodeDecodeError, ValueError, ValidationError):
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The SymPy worker returned output outside its exact bounded "
                    "protocol; no normalization evidence was retained."
                ),
            )
        if (
            sympy_polynomial_normalization_provider_runtime(refresh=True)
            != self.runtime
        ):
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed SymPy runtime changed during execution; no "
                    "normalization evidence was retained."
                ),
            )
        return _SympyNormalizationRun(
            execution_status=ExecutionStatus.COMPLETED,
            runtime_ms=_runtime_ms(started),
            normalized=normalized,
        )


class SympyPolynomialExpressionNormalizeAdapter:
    """Normalize one safe typed expression to canonical sparse coefficients."""

    def __init__(
        self,
        *,
        expressions: PolynomialExpressionArtifactService,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        self.expressions = expressions
        self.backend = _SympyPolynomialNormalizationBackend(runtime)
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.expression.normalize",
            version="1",
            title="Normalize a typed rational polynomial expression",
            description=(
                "Convert one bounded, versioned QQ-polynomial AST to canonical "
                "sparse coefficients without parsing or evaluating user strings. "
                "This is one concrete-expression outcome: finitely many "
                "normalizations do not verify an identity parameterized over all "
                "orders. Verify each full expression relation separately."
            ),
            provider="jacobian.sympy",
            provider_runtime=runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialExpressionNormalizeRequest),
            output_schema=model_schema(PolynomialExpressionNormalizeOutput),
            tags=(
                "polynomial",
                "symbolic",
                "normalization",
                "typed-expression",
                "exact-rational",
                "sympy",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="combine_like_terms",
                    description=(
                        "Normalize x + x to canonical sparse coefficients over QQ."
                    ),
                    mode=CapabilityMode.EXPLORE,
                    input=PolynomialExpressionNormalizeRequest.model_validate(
                        {
                            "expression": {
                                "variables": ["x"],
                                "expression": {
                                    "kind": "add",
                                    "operands": [
                                        {"kind": "variable", "name": "x"},
                                        {"kind": "variable", "name": "x"},
                                    ],
                                },
                            }
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
            validated = PolynomialExpressionNormalizeRequest.model_validate(
                request.input
            )
            expression_uri = self.expressions.put_expression(
                validated.expression
            ).artifact_uri
            resolved = self.expressions.resolve_expression(expression_uri)
        except (ValidationError, ValueError) as exc:
            budget_error = _expansion_budget_error(exc)
            if budget_error is not None:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="EXPANSION_TERM_BUDGET_EXCEEDED",
                        stage="bounded_normalization",
                        message=(
                            "The typed expression has a conservatively proven "
                            "expansion larger than the hard normalization term budget."
                        ),
                        path="expression.expression",
                        schema_uri=(
                            self.expressions.installation.expression_schema_uri
                        ),
                        expected=(
                            f"expanded term upper bound at most {budget_error.limit}"
                        ),
                        hint=(
                            "Do not increase the exponent or expression size using "
                            "the same full-expansion approach. Keep the expression "
                            "factored, use an operation that does not require full "
                            "sparse expansion, or split the calculation into smaller "
                            "exact expressions. Finite normalizations cannot prove "
                            "an all-orders claim."
                        ),
                        details={
                            "limit": budget_error.limit,
                            "estimated_expanded_terms_upper_bound": (
                                budget_error.estimated_expanded_terms_upper_bound
                            ),
                            "bound_kind": "CONSERVATIVE_UPPER_BOUND",
                            "requested_exponent": budget_error.requested_exponent,
                            "retryable_with_same_input": False,
                            "mathematical_scope": "ONE_CONCRETE_TYPED_EXPRESSION",
                            "supports_universal_claim": False,
                            "larger_same_family_full_expansions_expected_to_help": (
                                False
                            ),
                            "alternatives": [
                                "use a factored symbolic operation",
                                "split the expression before normalization",
                                "use a domain capability with bounded coefficient access",
                            ],
                            "normalization_uri": None,
                            "checker_input_available": False,
                        },
                    )
                ) from exc
            validation_errors = _validation_errors(exc)
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_TYPED_POLYNOMIAL_EXPRESSION",
                    stage="input_validation",
                    message=str(exc),
                    path="expression",
                    schema_uri=self.expressions.installation.expression_schema_uri,
                    expected=(
                        "one bounded v1 AST over an explicit ordered QQ polynomial "
                        "ring using rational, variable, add, multiply, negate, and "
                        "nonnegative power nodes"
                    ),
                    hint=(
                        "Declare every variable and use typed nodes; do not pass "
                        "formula strings or expression denominators."
                    ),
                    details={"validation_errors": validation_errors},
                )
            ) from exc

        run = self.backend.run(validated)
        normalization_uri: str | None = None
        if (
            run.execution_status is ExecutionStatus.COMPLETED
            and run.normalized is not None
        ):
            normalization_uri = self.expressions.put_normalization(
                expression_uri=expression_uri,
                normalized=run.normalized,
                producer=self.backend.runtime,
                resource_budget=validated.resource_budget,
            ).artifact_uri
        output = PolynomialExpressionNormalizeOutput(
            status=(
                "NORMALIZATION_PRODUCED"
                if normalization_uri is not None
                else "NO_NORMALIZATION_PRODUCED"
            ),
            expression_uri=expression_uri,
            normalization_uri=normalization_uri,
            normalized=run.normalized if normalization_uri is not None else None,
            verification_candidate_available=normalization_uri is not None,
            detail=(
                "Pinned SymPy produced canonical exact sparse QQ coefficients; "
                "equality with the typed source expression remains unverified."
                if normalization_uri is not None
                else (
                    run.detail
                    or "No normalization evidence was produced; no conclusion follows."
                )
            ),
        )
        relationships = (
            (
                CapabilityRelationship(
                    relation_id="polynomial.relation.expression-normalization-of",
                    source_artifact_uris=(normalization_uri,),
                    target_artifact_uris=(expression_uri,),
                ),
            )
            if normalization_uri is not None
            else ()
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=run.execution_status,
                runtime_ms=run.runtime_ms,
                detail=run.detail,
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the full declared typed QQ-polynomial expression",
                parameters={
                    "declared_scope": "FULL_EXPRESSION",
                    "variables": list(resolved.expression.variables),
                    "node_count": resolved.binding.node_count,
                    "depth": resolved.binding.depth,
                    "expanded_term_upper_bound": (
                        resolved.binding.expanded_term_upper_bound
                    ),
                    "coefficient_digit_budget": (
                        resolved.binding.coefficient_digit_budget
                    ),
                    "wall_seconds": validated.resource_budget.wall_seconds,
                },
                artifact_uri=expression_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if normalization_uri is not None
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "one full canonical sparse coefficient map was produced; "
                    "provider completion is not independent verification"
                    if normalization_uri is not None
                    else "the bounded provider attempt produced no normalization "
                    "evidence; no mathematical conclusion follows"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if run.execution_status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if run.execution_status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "the pinned exact SymPy provider produced bound canonical "
                    "coefficients, but provider success does not verify equivalence"
                    if normalization_uri is not None
                    else "provider execution did not complete; no mathematical "
                    "conclusion follows"
                ),
            ),
            artifact_uris=(
                (expression_uri, normalization_uri)
                if normalization_uri is not None
                else (expression_uri,)
            ),
            relationships=relationships,
        )


def _parse_worker_output(
    stdout: bytes,
    *,
    variables: tuple[str, ...],
) -> SparseRationalPolynomial:
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise ValueError("worker output is not exactly one line")
    return parse_sympy_polynomial_worker_response(
        loads_strict_json(stdout[:-1]),
        variables=variables,
    )


def _operational_failure(
    started: float,
    completed: ProcessResult,
) -> _SympyNormalizationRun | None:
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            started,
            ExecutionStatus.TIMEOUT,
            "The bounded SymPy normalization attempt timed out; no conclusion follows.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The SymPy normalization worker exceeded its output limit.",
        )
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The SymPy normalization worker failed; no evidence was retained.",
        )
    return None


def _failure(
    started: float,
    status: ExecutionStatus,
    detail: str,
) -> _SympyNormalizationRun:
    return _SympyNormalizationRun(
        execution_status=status,
        runtime_ms=_runtime_ms(started),
        detail=detail,
    )


def _expansion_budget_error(
    error: ValidationError | ValueError,
) -> PolynomialExpansionTermBudgetError | None:
    if isinstance(error, PolynomialExpansionTermBudgetError):
        return error
    if isinstance(error, ValidationError):
        for item in error.errors():
            context = item.get("ctx")
            nested = context.get("error") if isinstance(context, dict) else None
            if isinstance(nested, PolynomialExpansionTermBudgetError):
                return nested
    return None


def _validation_errors(error: ValidationError | ValueError) -> list[dict[str, Any]]:
    if isinstance(error, ValidationError):
        return [
            dict(item)
            for item in error.errors(include_url=False, include_context=False)
        ]
    return [
        {
            "type": type(error).__name__,
            "loc": ["expression"],
            "msg": str(error),
        }
    ]


def _runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
