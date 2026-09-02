"""Exact bounded singular-locus kernel for projective plane curves."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from time import monotonic
from typing import Literal, NoReturn

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.math.geometry.algebraic_curves._singularity_models import (
    MAX_PROJECTIVE_SINGULAR_COMPONENTS,
    MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE,
    MAX_PROJECTIVE_SINGULAR_POINTS,
    PositiveDimensionalProjectivePlaneCurveSingularLocus,
    ProjectivePlaneCurveFirstJet,
    ProjectivePlaneCurveSingularityBudget,
    ProjectivePlaneCurveSingularityOutcome,
    ProjectivePlaneCurveSingularityProfile,
    ProjectivePlaneCurveSingularityRequest,
    ProjectivePlaneCurveSingularPointRecord,
    SmoothProjectivePlaneCurve,
    ZeroDimensionalProjectivePlaneCurveSingularLocus,
)
from jacobian.math.geometry.algebraic_curves._singularity_point_process import (
    PointConstructionLimitError,
    run_point_construction_worker,
)
from jacobian.math.geometry.algebraic_curves._singularity_point_worker import (
    ProjectiveSingularityPointSeed,
    ProjectiveSingularityPointWorkerRequest,
)
from jacobian.math.geometry.projective.values import AlgebraicProjectivePlanePoint
from jacobian.math.number_theory.number_fields.operations import embeddings
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldEmbeddingProfile,
    SimpleNumberFieldElement,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.ideals._models import IdealComputationBudget
from jacobian.math.polynomials.ideals._singular import (
    SingularIdealResult,
    SingularMinimalPrimesResult,
    run_singular_ideal_operation,
    run_singular_minimal_primes,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_MAX_NORMALIZED_SOURCE_COEFFICIENT_DIGITS = 8
_MAX_BACKEND_GENERATORS = 64
_MAX_BACKEND_TERMS = 1_024
_MAX_IDEAL_GENERATOR_DEGREE = 4
_MAX_POINT_COORDINATE_DIGITS = 256
_MAX_SHAPE_ATTEMPTS = comb(MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE, 2) + 1

FailureStage = Literal[
    "SATURATION",
    "PROJECTIVE_COMPONENTS",
    "CHART_ZERO_COMPONENTS",
    "CHART_ONE_COMPONENTS",
    "POINT_CONSTRUCTION",
    "RESULT_CONSTRUCTION",
]


class _SingularityAdmissionError(ValueError):
    """The normalized request exceeds the derived execution envelope."""


@dataclass(frozen=True, slots=True)
class _SingularityAdmission:
    """One derived execution proof reused by every kernel phase."""

    degree: int
    source_terms: int
    normalized_coefficient_digits: int
    jacobian_generator_bound: int
    jacobian_term_bound: int
    macaulay_matrix_dimension_bound: int
    macaulay_minor_component_digits: int
    ideal_generator_degree_bound: int
    quotient_degree_bound: int
    geometric_point_bound: int
    shape_attempt_bound: int
    has_repeated_component: bool


def _remaining(deadline: float) -> float:
    return deadline - monotonic()


def _deadline_for(request: ProjectivePlaneCurveSingularityRequest) -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else monotonic()
    owner_deadline = started + request.resource_budget.wall_seconds
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _normalized_source(polynomial: RationalPolynomial) -> sympy.Poly:
    source = rational_polynomial_to_sympy(polynomial)
    _denominator, integer_source = source.clear_denoms(convert=True)
    _content, primitive = integer_source.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    return sympy.Poly(primitive, *source.gens, domain=sympy.QQ)


def _admit_singularity(source: sympy.Poly) -> _SingularityAdmission:
    """Derive intermediate, quotient, point, and exact-result bounds once."""

    degree = int(source.total_degree())
    source_terms = len(source.terms())
    coefficient_height = max(abs(int(coefficient)) for coefficient in source.coeffs())
    coefficient_digits = len(str(coefficient_height))
    if coefficient_digits > _MAX_NORMALIZED_SOURCE_COEFFICIENT_DIGITS:
        raise _SingularityAdmissionError(
            "primitive source coefficients exceed the derived 8-digit elimination envelope"
        )

    # For ternary cubics, the relevant homogeneous Macaulay layer through
    # degree four has C(4+2,2)=15 monomials.  Hadamard bounds a 15-square
    # minor by (3H)^15 * 15^8; Landau-Mignotte factor growth for a degree-four
    # eliminant contributes at most (n+1)2^n.  This bounds both numerator and
    # denominator components used by the residue-field coordinates.
    macaulay_dimension = 15
    minor_bound = (3 * max(coefficient_height, 1)) ** macaulay_dimension * (
        macaulay_dimension**8
    )
    factor_bound = (
        (MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE + 1)
        * 2**MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE
        * minor_bound
    )
    minor_digits = len(str(factor_bound))
    if minor_digits > _MAX_POINT_COORDINATE_DIGITS:
        raise _SingularityAdmissionError(
            "the derived elimination minors exceed the 256-digit point-carrier bound"
        )

    quotient_degree = (degree - 1) ** 2
    admission = _SingularityAdmission(
        degree=degree,
        source_terms=source_terms,
        normalized_coefficient_digits=coefficient_digits,
        jacobian_generator_bound=4,
        jacobian_term_bound=4 * source_terms,
        macaulay_matrix_dimension_bound=macaulay_dimension,
        macaulay_minor_component_digits=minor_digits,
        ideal_generator_degree_bound=_MAX_IDEAL_GENERATOR_DEGREE,
        quotient_degree_bound=quotient_degree,
        geometric_point_bound=quotient_degree,
        shape_attempt_bound=comb(quotient_degree, 2) + 1,
        has_repeated_component=not source.is_sqf,
    )
    if (
        admission.quotient_degree_bound > MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE
        or admission.geometric_point_bound > MAX_PROJECTIVE_SINGULAR_POINTS
        or admission.shape_attempt_bound > _MAX_SHAPE_ATTEMPTS
    ):
        raise _SingularityAdmissionError(
            "the derived quotient, point, shape, or exact-result envelope is exceeded"
        )
    return admission


def _to_public_polynomial(
    polynomial: sympy.Poly,
    variables: tuple[str, ...],
) -> RationalPolynomial:
    return rational_polynomial_from_sympy(
        polynomial,
        variables,
        maximum_terms=_MAX_BACKEND_TERMS,
    )


def _variable_polynomial(
    variables: tuple[str, str, str], index: int
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"),
                    exponents=tuple(1 if slot == index else 0 for slot in range(3)),
                ),
            )
        ),
    )


def _ideal_budget(
    request: ProjectivePlaneCurveSingularityRequest,
) -> IdealComputationBudget:
    return IdealComputationBudget(
        wall_seconds=request.resource_budget.wall_seconds,
        maximum_output_generators=_MAX_BACKEND_GENERATORS,
        maximum_output_terms=_MAX_BACKEND_TERMS,
    )


def _is_unit_ideal(ideal: RationalPolynomialIdeal) -> bool:
    if len(ideal.generators) != 1:
        return False
    terms = ideal.generators[0].polynomial.terms
    return (
        len(terms) == 1
        and not any(terms[0].exponents)
        and terms[0].coefficient.as_fraction() != 0
    )


def _unit_ideal(axis: tuple[str, str, str]) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=axis,
        generators=(
            RationalPolynomial(
                variables=axis,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational.from_integer_ratio(1, 1),
                            exponents=(0, 0, 0),
                        ),
                    )
                ),
            ),
        ),
    )


def _ideal_projection_limit_failure(
    stage: FailureStage,
    ideals: tuple[RationalPolynomialIdeal, ...],
    admission: _SingularityAdmission,
    *,
    maximum_ideals: int,
) -> None:
    """Reject a decoded ideal family that contradicts the admitted plan."""

    if len(ideals) > maximum_ideals:
        _limit_failure(
            stage,
            "the exact ideal family exceeds the admitted component bound",
        )
    generator_count = sum(len(ideal.generators) for ideal in ideals)
    term_count = sum(
        len(generator.polynomial.terms)
        for ideal in ideals
        for generator in ideal.generators
    )
    if generator_count > _MAX_BACKEND_GENERATORS:
        _limit_failure(
            stage,
            "the exact ideal family exceeds the admitted generator bound",
        )
    if term_count > _MAX_BACKEND_TERMS:
        _limit_failure(
            stage,
            "the exact ideal family exceeds the admitted term bound",
        )
    for ideal in ideals:
        for generator in ideal.generators:
            for term in generator.polynomial.terms:
                if sum(term.exponents) > admission.ideal_generator_degree_bound:
                    _limit_failure(
                        stage,
                        "an exact ideal generator exceeds the admitted degree bound",
                    )
                if (
                    len(term.coefficient.num.lstrip("-"))
                    > admission.macaulay_minor_component_digits
                    or len(term.coefficient.den)
                    > admission.macaulay_minor_component_digits
                ):
                    _limit_failure(
                        stage,
                        "an exact ideal coefficient exceeds the admitted Macaulay bound",
                    )


def _failure(
    stage: FailureStage,
    backend: SingularIdealResult | SingularMinimalPrimesResult,
) -> NoReturn:
    detail = backend.detail or "the exact backend did not produce a complete result"
    if backend.outcome == "TIMEOUT":
        raise OperationExecutionTimeoutError(detail)
    if backend.outcome == "CANCELLED":
        raise OperationExecutionCancelledError(detail)
    raise RuntimeError(f"projective singularity backend failed at {stage}: {detail}")


def _local_failure(
    detail: str,
) -> NoReturn:
    raise RuntimeError(detail[:256])


def _limit_failure(
    stage: FailureStage,
    detail: str,
) -> NoReturn:
    raise RuntimeError(
        f"projective singularity limit exceeded at {stage}: {detail[:256]}"
    )


def _timeout_failure(
    stage: FailureStage,
) -> NoReturn:
    raise OperationExecutionTimeoutError(
        f"projective singularity deadline expired during {stage}"
    )


def _cancelled_failure(
    stage: FailureStage,
) -> NoReturn:
    raise OperationExecutionCancelledError(
        f"projective singularity request was cancelled during {stage}"
    )


def _specialize_ideal(
    ideal: RationalPolynomialIdeal,
    *,
    substitutions: dict[int, int],
    remaining_indices: tuple[int, ...],
) -> RationalPolynomialIdeal:
    remaining_variables = tuple(ideal.variables[index] for index in remaining_indices)
    generators: list[RationalPolynomial] = []
    for generator in ideal.generators:
        coefficients: dict[tuple[int, ...], Fraction] = {}
        for term in generator.polynomial.terms:
            if any(
                value == 0 and term.exponents[index] > 0
                for index, value in substitutions.items()
            ):
                continue
            exponents = tuple(term.exponents[index] for index in remaining_indices)
            coefficients[exponents] = (
                coefficients.get(exponents, Fraction(0))
                + term.coefficient.as_fraction()
            )
        terms = tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                exponents=exponents,
            )
            for exponents, coefficient in sorted(coefficients.items(), reverse=True)
            if coefficient
        )
        generators.append(
            RationalPolynomial(
                variables=remaining_variables,
                polynomial=SparseRationalPolynomial(terms=terms),
            )
        )
    return RationalPolynomialIdeal(
        variables=remaining_variables,
        generators=tuple(generators),
    )


def _chart_two_is_present(ideal: RationalPolynomialIdeal) -> bool:
    return all(
        sum(
            (
                term.coefficient.as_fraction()
                for term in generator.polynomial.terms
                if term.exponents[0] == 0 and term.exponents[1] == 0
            ),
            Fraction(0),
        )
        == 0
        for generator in ideal.generators
    )


def _records_from_worker_seeds(
    seeds: tuple[ProjectiveSingularityPointSeed, ...],
    *,
    axis: tuple[str, str, str],
    maximum_points: int,
) -> tuple[ProjectivePlaneCurveSingularPointRecord, ...]:
    if sum(seed.presentation.degree for seed in seeds) > maximum_points:
        raise RuntimeError("point worker exceeded the admitted geometric point bound")
    records: list[ProjectivePlaneCurveSingularPointRecord] = []
    profiles_by_presentation: dict[str, NumberFieldEmbeddingProfile] = {}
    for seed in seeds:
        zero = SimpleNumberFieldElement(
            presentation=seed.presentation,
            coefficients_ascending=tuple(
                CanonicalRational(num="0", den="1")
                for _ in range(seed.presentation.degree)
            ),
        )
        presentation_key = seed.presentation.model_dump_json()
        profile = profiles_by_presentation.get(presentation_key)
        if profile is None:
            profile = embeddings(seed.presentation)
            profiles_by_presentation[presentation_key] = profile
        records.extend(
            ProjectivePlaneCurveSingularPointRecord(
                point=AlgebraicProjectivePlanePoint(
                    axis=axis,
                    embedding=record.embedding,
                    coordinates=seed.coordinates,
                    chart_index=seed.chart_index,
                ),
                first_jet=ProjectivePlaneCurveFirstJet._from_kernel(
                    value=zero,
                    gradient=(zero, zero, zero),
                ),
            )
            for record in profile.records
        )
    return tuple(sorted(records, key=lambda record: record.point.model_dump_json()))


def _complete_zero_dimensional_points(
    saturation: RationalPolynomialIdeal,
    *,
    admission: _SingularityAdmission,
    axis: tuple[str, str, str],
    budget: IdealComputationBudget,
    deadline: float,
    maximum_points: int,
) -> tuple[ProjectivePlaneCurveSingularPointRecord, ...]:
    chart_zero = _specialize_ideal(
        saturation,
        substitutions={0: 1},
        remaining_indices=(1, 2),
    )
    chart_zero_primes = run_singular_minimal_primes(
        chart_zero,
        budget,
        wall_seconds=_remaining(deadline),
    )
    if chart_zero_primes.outcome != "COMPUTED":
        _failure("CHART_ZERO_COMPONENTS", chart_zero_primes)
    chart_zero_components = chart_zero_primes.components or ()
    _ideal_projection_limit_failure(
        "CHART_ZERO_COMPONENTS",
        chart_zero_components,
        admission,
        maximum_ideals=MAX_PROJECTIVE_SINGULAR_POINTS,
    )

    chart_one = _specialize_ideal(
        saturation,
        substitutions={0: 0, 1: 1},
        remaining_indices=(2,),
    )
    chart_one_primes = run_singular_minimal_primes(
        chart_one,
        budget,
        wall_seconds=_remaining(deadline),
    )
    if chart_one_primes.outcome != "COMPUTED":
        _failure("CHART_ONE_COMPONENTS", chart_one_primes)
    chart_one_components = chart_one_primes.components or ()
    _ideal_projection_limit_failure(
        "CHART_ONE_COMPONENTS",
        chart_one_components,
        admission,
        maximum_ideals=MAX_PROJECTIVE_SINGULAR_POINTS,
    )

    worker_request = ProjectiveSingularityPointWorkerRequest(
        variables=axis,
        chart_zero_components=chart_zero_components,
        chart_one_components=chart_one_components,
        chart_two_present=_chart_two_is_present(saturation),
    )
    worker_response = run_point_construction_worker(
        worker_request,
        deadline=deadline,
    )
    return _records_from_worker_seeds(
        worker_response.seeds,
        axis=axis,
        maximum_points=maximum_points,
    )


def _profile(
    *,
    source: RationalPolynomial,
    partials: tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial],
    outcome: ProjectivePlaneCurveSingularityOutcome,
) -> ProjectivePlaneCurveSingularityProfile:
    return ProjectivePlaneCurveSingularityProfile._from_kernel(
        source=source,
        axis=(source.variables[0], source.variables[1], source.variables[2]),
        partials=partials,
        outcome=outcome,
    )


def _positive_dimensional_outcome(
    saturation: RationalPolynomialIdeal,
    *,
    admission: _SingularityAdmission,
    budget: IdealComputationBudget,
    deadline: float,
) -> ProjectivePlaneCurveSingularityOutcome:
    components_backend = run_singular_minimal_primes(
        saturation,
        budget,
        wall_seconds=_remaining(deadline),
    )
    if components_backend.outcome != "COMPUTED":
        _failure("PROJECTIVE_COMPONENTS", components_backend)
    components = components_backend.components or ()
    _ideal_projection_limit_failure(
        "PROJECTIVE_COMPONENTS",
        components,
        admission,
        maximum_ideals=MAX_PROJECTIVE_SINGULAR_COMPONENTS,
    )
    if not components:
        _local_failure("a repeated component produced no rational minimal component")
    return PositiveDimensionalProjectivePlaneCurveSingularLocus._from_kernel(
        ideal=saturation,
        components=components,
    )


def _bounded_result_profile(
    *,
    source: RationalPolynomial,
    partials: tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial],
    outcome: ProjectivePlaneCurveSingularityOutcome,
    admission: _SingularityAdmission,
    deadline: float,
) -> ProjectivePlaneCurveSingularityProfile:
    if _remaining(deadline) <= 0:
        return _profile(
            source=source,
            partials=partials,
            outcome=_timeout_failure("RESULT_CONSTRUCTION"),
        )
    profile = _profile(source=source, partials=partials, outcome=outcome)
    if _remaining(deadline) <= 0:
        return _profile(
            source=source,
            partials=partials,
            outcome=_timeout_failure("RESULT_CONSTRUCTION"),
        )
    return profile


def _singularity_profile_request(
    request: ProjectivePlaneCurveSingularityRequest,
) -> ProjectivePlaneCurveSingularityProfile:
    deadline = _deadline_for(request)

    try:
        source_backend = _normalized_source(request.polynomial)
        admission = _admit_singularity(source_backend)
    except _SingularityAdmissionError as exc:
        # This is deterministic semantic admission, not a backend failure.
        from jacobian.catalog.models import OperationDomainValidationError

        raise OperationDomainValidationError(
            location=("polynomial",),
            code="projective_plane_curve.normalized_height_bound",
            message=str(exc),
        ) from exc

    axis = (
        request.polynomial.variables[0],
        request.polynomial.variables[1],
        request.polynomial.variables[2],
    )
    source = _to_public_polynomial(source_backend, axis)
    partials_backend = (
        source_backend.diff(source_backend.gens[0]),
        source_backend.diff(source_backend.gens[1]),
        source_backend.diff(source_backend.gens[2]),
    )
    partials = (
        _to_public_polynomial(partials_backend[0], axis),
        _to_public_polynomial(partials_backend[1], axis),
        _to_public_polynomial(partials_backend[2], axis),
    )
    if admission.degree == 1:
        return _bounded_result_profile(
            source=source,
            partials=partials,
            outcome=SmoothProjectivePlaneCurve._from_kernel(_unit_ideal(axis)),
            admission=admission,
            deadline=deadline,
        )
    if _remaining(deadline) <= 0:
        return _profile(
            source=source,
            partials=partials,
            outcome=_timeout_failure("SATURATION"),
        )
    budget = _ideal_budget(request)
    jacobian_ideal = RationalPolynomialIdeal(
        variables=axis,
        generators=(source, *partials),
    )
    irrelevant_ideal = RationalPolynomialIdeal(
        variables=axis,
        generators=tuple(_variable_polynomial(axis, index) for index in range(3)),
    )
    saturation_backend = run_singular_ideal_operation(
        "saturation",
        jacobian_ideal,
        irrelevant_ideal,
        budget,
        wall_seconds=_remaining(deadline),
    )
    if saturation_backend.outcome != "COMPUTED" or saturation_backend.ideal is None:
        _failure("SATURATION", saturation_backend)
    saturation = saturation_backend.ideal
    _ideal_projection_limit_failure(
        "SATURATION",
        (saturation,),
        admission,
        maximum_ideals=1,
    )

    if admission.has_repeated_component:
        outcome: ProjectivePlaneCurveSingularityOutcome = _positive_dimensional_outcome(
            saturation,
            admission=admission,
            budget=budget,
            deadline=deadline,
        )
    elif _is_unit_ideal(saturation):
        outcome = SmoothProjectivePlaneCurve._from_kernel(saturation)
    else:
        try:
            points = _complete_zero_dimensional_points(
                saturation,
                admission=admission,
                axis=axis,
                budget=budget,
                deadline=deadline,
                maximum_points=admission.geometric_point_bound,
            )
        except OperationExecutionTimeoutError:
            outcome = _timeout_failure("POINT_CONSTRUCTION")
        except OperationExecutionCancelledError:
            outcome = _cancelled_failure("POINT_CONSTRUCTION")
        except PointConstructionLimitError as exc:
            outcome = _limit_failure("POINT_CONSTRUCTION", str(exc))
        except (ValueError, TypeError, ArithmeticError, RuntimeError):
            outcome = _local_failure(
                "exact point construction rejected malformed backend algebra"
            )
        else:
            outcome = ZeroDimensionalProjectivePlaneCurveSingularLocus._from_kernel(
                ideal=saturation,
                points=points,
            )

    return _bounded_result_profile(
        source=source,
        partials=partials,
        outcome=outcome,
        admission=admission,
        deadline=deadline,
    )


def singularity_profile(
    polynomial: RationalPolynomial,
    resource_budget: ProjectivePlaneCurveSingularityBudget | None = None,
) -> ProjectivePlaneCurveSingularityProfile:
    """Compute the complete geometric singular locus of one admitted curve."""

    started = monotonic()
    request = ProjectivePlaneCurveSingularityRequest(
        polynomial=polynomial,
        resource_budget=resource_budget or ProjectivePlaneCurveSingularityBudget(),
    )
    if current_request_execution() is not None:
        return _singularity_profile_request(request)
    with request_execution(started):
        return _singularity_profile_request(request)


__all__ = ["singularity_profile"]
