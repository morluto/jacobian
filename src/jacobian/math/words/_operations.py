"""Wire adapters for public combinatorics-on-words operations."""

from __future__ import annotations

from jacobian.math.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
)
from jacobian.math.words.operations import factors_of_length, incidence_matrix, periods


def compute_factors_length(request: FactorsLengthRequest) -> FactorsLengthResult:
    analysis = factors_of_length(request.word, request.factor_length)
    occurrences = analysis.occurrences
    return FactorsLengthResult(
        **request.model_dump(),
        factors=analysis.factors,
        occurrences=occurrences,
        multiplicities=tuple(len(indices) for indices in occurrences),
        first_occurrence=tuple(indices[0] for indices in occurrences),
        distinct_count=len(analysis.factors),
    )


def compute_periods(request: PeriodsRequest) -> PeriodsResult:
    analysis = periods(request.word)
    return PeriodsResult(
        **request.model_dump(),
        periods=analysis.periods,
        least_period=analysis.least_period,
        is_primitive=analysis.primitive,
    )


def compute_incidence_matrix(
    request: IncidenceMatrixRequest,
) -> IncidenceMatrixResult:
    return IncidenceMatrixResult(
        **request.model_dump(),
        matrix=incidence_matrix(request.morphism),
    )


__all__ = ["compute_factors_length", "compute_incidence_matrix", "compute_periods"]
