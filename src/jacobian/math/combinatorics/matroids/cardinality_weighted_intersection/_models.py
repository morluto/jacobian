"""Typed values for lexicographic matroid intersection optimization."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids._models import (
    MAX_GROUND_SIZE,
    MAX_WEIGHT_DIGITS,
    MatroidWeightedIntersectionOptimizationResult,
    _validation_error,
)


class MatroidCardinalityWeightedIntersectionResult(StrictModel):
    """Maximum-cardinality common independent set, then maximum original weight.

    ``optimized.weight_function`` stores the cardinality-shifted objective.
    Subtracting ``cardinality_bonus`` from each entry recovers the caller's
    original objective without duplicating its ground axis in the result.
    """

    optimized: MatroidWeightedIntersectionOptimizationResult
    cardinality_bonus: StrictInt = Field(gt=0)
    cardinality: StrictInt = Field(ge=0, le=MAX_GROUND_SIZE)
    total_weight: StrictInt

    @model_validator(mode="after")
    def require_lexicographic_contract(self) -> Self:
        witness = self.optimized
        n = witness.first.ground_size
        if self.cardinality != len(witness.common_independent):
            raise _validation_error(
                "cardinality_weighted.cardinality",
                "cardinality must equal the selected common-set size",
            )

        original_weights = tuple(
            value - self.cardinality_bonus for value in witness.weight_function.values
        )
        if any(abs(value) >= 10**MAX_WEIGHT_DIGITS for value in original_weights):
            raise _validation_error(
                "cardinality_weighted.weights",
                "recovered original weights exceed the objective bound",
            )
        maximum_absolute_weight = max(
            (abs(value) for value in original_weights), default=0
        )
        if self.cardinality_bonus <= 2 * n * maximum_absolute_weight:
            raise _validation_error(
                "cardinality_weighted.bonus",
                "cardinality bonus does not dominate every weight change",
            )
        expected_total = sum(
            original_weights[index] for index in witness.common_independent
        )
        if self.total_weight != expected_total:
            raise _validation_error(
                "cardinality_weighted.objective",
                "total_weight must equal the original selected weight",
            )
        return self
