"""Domain adapter for universal-algebra operations."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    CongruenceResult,
    EquationCounterexample,
    EquationProfileRequest,
    EquationProfileResult,
    EvaluateRequest,
    EvaluateResult,
    HomomorphismProfileRequest,
    HomomorphismProfileResult,
    QuotientRequest,
    SubalgebraRequest,
    SubalgebraResult,
)
from jacobian.math.universal_algebra.operations import (
    congruence_check,
    equation_profile,
    evaluate_term,
    generated_subalgebra,
    homomorphism_profile,
    quotient,
)
from jacobian.math.universal_algebra.values import FiniteAlgebraHomomorphism

__all__ = [
    "compute_congruence",
    "compute_equation_profile",
    "compute_evaluate",
    "compute_generated_subalgebra",
    "compute_homomorphism_profile",
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


def compute_homomorphism_profile(
    request: HomomorphismProfileRequest,
) -> HomomorphismProfileResult:
    return HomomorphismProfileResult._from_kernel(
        homomorphism_profile(request.carrier_map)
    )


def _verify_homomorphism_profile_result(result: HomomorphismProfileResult) -> bool:
    """Deliberately recheck one independently supplied homomorphism claim."""

    source = (
        result.homomorphism if result.status == "HOMOMORPHISM" else result.carrier_map
    )
    if source is None:
        return False
    try:
        request = HomomorphismProfileRequest.model_validate(
            {"carrier_map": source.model_dump(mode="python")}
        )
    except ValidationError:
        return False
    expected = HomomorphismProfileResult._from_kernel(
        homomorphism_profile(request.carrier_map)
    )
    return result == expected


def compute_congruence(request: CongruenceRequest) -> CongruenceResult:
    result = congruence_check(request.algebra, request.partition)
    return CongruenceResult(
        is_congruence=result["is_congruence"],  # type: ignore[arg-type]
        obstruction=result.get("obstruction"),  # type: ignore[arg-type]
    )


def compute_quotient(request: QuotientRequest) -> FiniteAlgebraHomomorphism:
    return quotient(request.algebra, request.partition)
