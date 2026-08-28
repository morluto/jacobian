"""Wire adapters for public combinatorics-on-words operations."""

from __future__ import annotations

from collections.abc import Callable

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.languages.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
    SubstitutionDependencyGraphRequest,
    SubstitutionDependencyGraphResult,
    SubstitutionFixedPointPrefixRequest,
    SubstitutionFixedPointPrefixResult,
    SubstitutionPrimitivityProfileRequest,
    SubstitutionPrimitivityProfileResult,
)
from jacobian.math.logic.languages.words.operations import (
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    periods,
    substitution_dependency_graph,
    substitution_primitivity_profile,
)


def _admit[T](
    admission: Callable[[], T],
    *,
    location: tuple[str | int, ...],
    code: str,
) -> T:
    try:
        return admission()
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(exc),
        ) from exc


def compute_factors_length(request: FactorsLengthRequest) -> FactorsLengthResult:
    analysis = _admit(
        lambda: factors_of_length(request.word, request.factor_length),
        location=("factor_length",),
        code="word.factor_length_out_of_range",
    )
    return FactorsLengthResult._from_kernel(
        request, factors=analysis.factors, occurrences=analysis.occurrences
    )


def compute_periods(request: PeriodsRequest) -> PeriodsResult:
    analysis = periods(request.word)
    return PeriodsResult._from_kernel(
        request,
        periods=analysis.periods,
        least_period=analysis.least_period,
        is_primitive=analysis.primitive,
    )


def compute_incidence_matrix(
    request: IncidenceMatrixRequest,
) -> IncidenceMatrixResult:
    return IncidenceMatrixResult._from_kernel(
        request, incidence_matrix(request.morphism)
    )


def compute_substitution_dependency_graph(
    request: SubstitutionDependencyGraphRequest,
) -> SubstitutionDependencyGraphResult:
    graph = _admit(
        lambda: substitution_dependency_graph(request.substitution),
        location=("substitution",),
        code="words.substitution_dependency_graph_not_admitted",
    )
    return SubstitutionDependencyGraphResult._from_kernel(request, graph)


def compute_substitution_primitivity_profile(
    request: SubstitutionPrimitivityProfileRequest,
) -> SubstitutionPrimitivityProfileResult:
    analysis = _admit(
        lambda: substitution_primitivity_profile(request.dependency_graph),
        location=("dependency_graph",),
        code="words.substitution_primitivity_not_admitted",
    )
    return SubstitutionPrimitivityProfileResult._from_kernel(
        request,
        strongly_connected_components=analysis.strongly_connected_components,
        irreducible=analysis.irreducible,
        aperiodic=analysis.aperiodic,
        primitive=analysis.primitive,
        least_positive_power=analysis.least_positive_power,
        exponent_upper_bound=analysis.exponent_upper_bound,
        obstruction=analysis.obstruction,
    )


def compute_substitution_fixed_point_prefix(
    request: SubstitutionFixedPointPrefixRequest,
) -> SubstitutionFixedPointPrefixResult:
    analysis = _admit(
        lambda: fixed_point_prefix(request.source, request.prefix_length),
        location=("source", "prefix_length"),
        code="words.fixed_point_prefix_not_admitted",
    )
    return SubstitutionFixedPointPrefixResult._from_kernel(
        request,
        prefix=analysis.prefix,
        least_iterate_depth=analysis.least_iterate_depth,
        retained_prefix_lengths=analysis.retained_prefix_lengths,
    )


__all__ = [
    "compute_factors_length",
    "compute_incidence_matrix",
    "compute_periods",
    "compute_substitution_dependency_graph",
    "compute_substitution_fixed_point_prefix",
    "compute_substitution_primitivity_profile",
]
