"""Domain adapter for universal-algebra operations."""

from __future__ import annotations

from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    CongruenceResult,
    EquationCounterexample,
    EquationProfileRequest,
    EquationProfileResult,
    EvaluateRequest,
    EvaluateResult,
    QuotientRequest,
    QuotientResult,
    SubalgebraRequest,
    SubalgebraResult,
)
from jacobian.math.universal_algebra.operations import (
    congruence_check,
    equation_profile,
    evaluate_term,
    generated_subalgebra,
    quotient,
)

__all__ = [
    "compute_congruence",
    "compute_equation_profile",
    "compute_evaluate",
    "compute_generated_subalgebra",
    "compute_quotient",
]


def compute_evaluate(request: EvaluateRequest) -> EvaluateResult:
    assignment = dict(enumerate(request.assignment))
    value = evaluate_term(request.algebra, request.term, assignment)
    return EvaluateResult(value=value)


def compute_equation_profile(request: EquationProfileRequest) -> EquationProfileResult:
    result = equation_profile(
        request.algebra, request.left, request.right, request.variable_count
    )
    if result["status"] == "HOLDS":
        return EquationProfileResult(
            status="HOLDS",
            satisfying_count=result["satisfying_count"],  # type: ignore[arg-type]
        )
    return EquationProfileResult(
        status="FAILS",
        satisfying_count=result["satisfying_count"],  # type: ignore[arg-type]
        first_counterassignment=EquationCounterexample.model_validate(
            result["first_counterassignment"]
        ),
    )


def compute_generated_subalgebra(request: SubalgebraRequest) -> SubalgebraResult:
    result = generated_subalgebra(request.algebra, request.generators)
    return SubalgebraResult(
        generated_carrier=result["generated_carrier"],  # type: ignore[arg-type]
        rounds=result["rounds"],  # type: ignore[arg-type]
        is_closed=result["is_closed"],  # type: ignore[arg-type]
    )


def compute_congruence(request: CongruenceRequest) -> CongruenceResult:
    result = congruence_check(request.algebra, request.partition)
    return CongruenceResult(
        is_congruence=result["is_congruence"],  # type: ignore[arg-type]
        obstruction=result.get("obstruction"),  # type: ignore[arg-type]
    )


def compute_quotient(request: QuotientRequest) -> QuotientResult:
    algebra, quotient_map = quotient(request.algebra, request.partition)
    return QuotientResult(
        algebra=algebra,
        quotient_map=quotient_map,
    )
