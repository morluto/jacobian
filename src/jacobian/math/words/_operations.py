"""Wire adapters for public combinatorics-on-words operations."""

from __future__ import annotations

from jacobian.math.words._models import (
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
from jacobian.math.words.operations import (
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    periods,
    substitution_dependency_graph,
    substitution_primitivity_profile,
)


def compute_factors_length(request: FactorsLengthRequest) -> FactorsLengthResult:
    analysis = factors_of_length(request.word, request.factor_length)
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
    return SubstitutionDependencyGraphResult._from_kernel(
        request, substitution_dependency_graph(request.substitution)
    )


def compute_substitution_primitivity_profile(
    request: SubstitutionPrimitivityProfileRequest,
) -> SubstitutionPrimitivityProfileResult:
    analysis = substitution_primitivity_profile(request.dependency_graph)
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
    analysis = fixed_point_prefix(request.source, request.prefix_length)
    return SubstitutionFixedPointPrefixResult._from_kernel(
        request,
        prefix=analysis.prefix,
        least_iterate_depth=analysis.least_iterate_depth,
        retained_prefix_lengths=analysis.retained_prefix_lengths,
    )


def verify_factors_length_result(result: FactorsLengthResult) -> bool:
    """Check one independently supplied complete factor enumeration."""

    expected = factors_of_length(result.word, result.factor_length)
    return (
        result.factors,
        result.occurrences,
        result.multiplicities,
        result.first_occurrence,
        result.distinct_count,
    ) == (
        expected.factors,
        expected.occurrences,
        tuple(len(indices) for indices in expected.occurrences),
        tuple(indices[0] for indices in expected.occurrences),
        len(expected.factors),
    )


def verify_periods_result(result: PeriodsResult) -> bool:
    """Check one independently supplied complete period profile."""

    expected = periods(result.word)
    return (result.periods, result.least_period, result.is_primitive) == (
        expected.periods,
        expected.least_period,
        expected.primitive,
    )


def verify_incidence_matrix_result(result: IncidenceMatrixResult) -> bool:
    """Check one independently supplied incidence matrix."""

    return result.matrix == incidence_matrix(result.morphism)


def verify_substitution_dependency_graph_result(
    result: SubstitutionDependencyGraphResult,
) -> bool:
    """Check one independently supplied complete dependency graph."""

    return result.graph == substitution_dependency_graph(result.substitution)


def verify_substitution_primitivity_profile_result(
    result: SubstitutionPrimitivityProfileResult,
) -> bool:
    """Check one independently supplied bounded primitivity profile."""

    expected = substitution_primitivity_profile(result.dependency_graph)
    return (
        result.strongly_connected_components,
        result.irreducible,
        result.aperiodic,
        result.primitive,
        result.least_positive_power,
        result.exponent_upper_bound,
        result.obstruction,
    ) == (
        expected.strongly_connected_components,
        expected.irreducible,
        expected.aperiodic,
        expected.primitive,
        expected.least_positive_power,
        expected.exponent_upper_bound,
        expected.obstruction,
    )


def verify_substitution_fixed_point_prefix_result(
    result: SubstitutionFixedPointPrefixResult,
) -> bool:
    """Check one independently supplied prefix within the admitted replay envelope."""

    analysis = fixed_point_prefix(result.source, result.prefix_length)
    return (
        result.prefix,
        result.least_iterate_depth,
        result.retained_prefix_lengths,
    ) == (
        analysis.prefix,
        analysis.least_iterate_depth,
        analysis.retained_prefix_lengths,
    )


__all__ = [
    "compute_factors_length",
    "compute_incidence_matrix",
    "compute_periods",
    "compute_substitution_dependency_graph",
    "compute_substitution_fixed_point_prefix",
    "compute_substitution_primitivity_profile",
    "verify_factors_length_result",
    "verify_incidence_matrix_result",
    "verify_periods_result",
    "verify_substitution_dependency_graph_result",
    "verify_substitution_fixed_point_prefix_result",
    "verify_substitution_primitivity_profile_result",
]
