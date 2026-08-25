"""Typed wire contracts for exact combinatorics-on-words operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.words.operations import (
    _require_fixed_point_prefix_budget,
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    periods,
    substitution_dependency_graph,
    substitution_primitivity_profile,
)
from jacobian.math.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyGraph,
    WordMorphism,
    _require_dependency_occurrence_bound,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"word.{reason}", message)


class FactorsLengthRequest(StrictModel):
    """Enumerate all distinct factors of one valid length."""

    word: FiniteWord
    factor_length: int = Field(ge=0)

    @model_validator(mode="after")
    def require_bounded_factor_length(self) -> Self:
        if self.factor_length > len(self.word.letters):
            raise _validation_error(
                "factor_length_exceeds_word",
                "factor_length must not exceed the word length",
            )
        return self


class FactorsLengthResult(FactorsLengthRequest):
    """Complete factor enumeration, ordered by first occurrence."""

    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]
    multiplicities: tuple[int, ...]
    first_occurrence: tuple[int, ...]
    distinct_count: int = Field(ge=0)
    complete: Literal[True] = True
    scope: Literal["ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"] = (
        "ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"
    )
    method: Literal["EXACT_SLIDING_WINDOW_ENUMERATION"] = (
        "EXACT_SLIDING_WINDOW_ENUMERATION"
    )

    @model_validator(mode="after")
    def bind_exact_factor_enumeration(self) -> Self:
        expected = factors_of_length(self.word, self.factor_length)
        expected_occurrences = expected.occurrences
        if (
            self.factors != expected.factors
            or self.occurrences != expected_occurrences
            or self.multiplicities
            != tuple(len(indices) for indices in expected_occurrences)
            or self.first_occurrence
            != tuple(indices[0] for indices in expected_occurrences)
            or self.distinct_count != len(expected.factors)
        ):
            raise _validation_error(
                "factor_result_unbound",
                "factor result is not bound to the requested word",
            )
        return self


class PeriodsRequest(StrictModel):
    """Compute all overlap periods of a finite word."""

    word: FiniteWord


class PeriodsResult(PeriodsRequest):
    """Complete overlap-period profile and proper-power primitivity."""

    periods: tuple[int, ...]
    least_period: int = Field(ge=0)
    is_primitive: bool
    complete: Literal[True] = True
    method: Literal["EXACT_OVERLAP_COMPARISON"] = "EXACT_OVERLAP_COMPARISON"
    primitive_convention: Literal["NOT_A_NONTRIVIAL_INTEGER_POWER"] = (
        "NOT_A_NONTRIVIAL_INTEGER_POWER"
    )
    empty_word_convention: Literal["NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"] = (
        "NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"
    )

    @model_validator(mode="after")
    def bind_exact_period_profile(self) -> Self:
        expected = periods(self.word)
        if (
            self.periods != expected.periods
            or self.least_period != expected.least_period
            or self.is_primitive != expected.primitive
        ):
            raise _validation_error(
                "period_result_unbound",
                "period result is not bound to the requested word",
            )
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix of a finite word morphism."""

    morphism: WordMorphism


class IncidenceMatrixResult(IncidenceMatrixRequest):
    """Exact target-by-source incidence matrix."""

    matrix: tuple[tuple[int, ...], ...]
    complete: Literal[True] = True
    method: Literal["EXACT_SYMBOL_COUNTING"] = "EXACT_SYMBOL_COUNTING"
    orientation: Literal["ROWS_TARGET_COLUMNS_SOURCE"] = "ROWS_TARGET_COLUMNS_SOURCE"

    @model_validator(mode="after")
    def bind_exact_incidence_matrix(self) -> Self:
        if self.matrix != incidence_matrix(self.morphism):
            raise _validation_error(
                "incidence_matrix_unbound",
                "incidence matrix is not bound to the requested morphism",
            )
        return self


class SubstitutionDependencyGraphRequest(StrictModel):
    """Construct the exact dependency graph of one bounded substitution."""

    substitution: Substitution

    @model_validator(mode="after")
    def require_bounded_occurrence_output(self) -> Self:
        try:
            _require_dependency_occurrence_bound(self.substitution)
        except ValueError as error:
            raise _validation_error(
                "dependency_occurrence_bound", str(error)
            ) from error
        return self


class SubstitutionDependencyGraphResult(SubstitutionDependencyGraphRequest):
    """Exact source-bound letter graph, including every occurrence position."""

    graph: SubstitutionDependencyGraph
    complete: Literal[True] = True
    method: Literal["EXACT_IMAGE_OCCURRENCE_ENUMERATION"] = (
        "EXACT_IMAGE_OCCURRENCE_ENUMERATION"
    )
    edge_convention: Literal["SOURCE_TO_OCCURRING_TARGET"] = (
        "SOURCE_TO_OCCURRING_TARGET"
    )

    @model_validator(mode="after")
    def bind_exact_dependency_graph(self) -> Self:
        if self.graph != substitution_dependency_graph(self.substitution):
            raise _validation_error(
                "dependency_graph_unbound",
                "dependency graph result is not bound to the substitution",
            )
        return self


class SubstitutionPrimitivityProfileRequest(StrictModel):
    """Decide primitivity from a canonical substitution dependency graph."""

    dependency_graph: SubstitutionDependencyGraph

    @model_validator(mode="after")
    def require_bounded_dependency_source(self) -> Self:
        try:
            _require_dependency_occurrence_bound(self.dependency_graph.substitution)
        except ValueError as error:
            raise _validation_error(
                "dependency_occurrence_bound", str(error)
            ) from error
        return self


class SubstitutionPrimitivityProfileResult(SubstitutionPrimitivityProfileRequest):
    """Complete Boolean-power primitivity profile with graph obstruction."""

    strongly_connected_components: tuple[tuple[str, ...], ...]
    irreducible: bool
    aperiodic: bool | None
    primitive: bool
    least_positive_power: int | None = Field(default=None, ge=1)
    exponent_upper_bound: int = Field(ge=1)
    obstruction: Literal[
        "NONE", "REDUCIBLE_DEPENDENCY_GRAPH", "PERIODIC_DEPENDENCY_GRAPH"
    ]
    complete: Literal[True] = True
    method: Literal["BOOLEAN_POWERS_THROUGH_WIELANDT_BOUND"] = (
        "BOOLEAN_POWERS_THROUGH_WIELANDT_BOUND"
    )

    @model_validator(mode="after")
    def bind_exact_primitivity_profile(self) -> Self:
        expected = substitution_primitivity_profile(self.dependency_graph)
        if (
            self.strongly_connected_components != expected.strongly_connected_components
            or self.irreducible != expected.irreducible
            or self.aperiodic != expected.aperiodic
            or self.primitive != expected.primitive
            or self.least_positive_power != expected.least_positive_power
            or self.exponent_upper_bound != expected.exponent_upper_bound
            or self.obstruction != expected.obstruction
        ):
            raise _validation_error(
                "primitivity_result_unbound",
                "primitivity result is not bound to the dependency graph",
            )
        return self


class SubstitutionFixedPointPrefixRequest(StrictModel):
    """Request one bounded prefix of a certified prolongable substitution."""

    source: ProlongableSubstitution
    prefix_length: int = Field(ge=0, le=MAX_MORPHISM_OUTPUT_LENGTH)

    @model_validator(mode="after")
    def require_bounded_source_work_and_result(self) -> Self:
        try:
            _require_fixed_point_prefix_budget(self.source, self.prefix_length)
        except ValueError as error:
            raise _validation_error("fixed_point_budget", str(error)) from error
        return self


class SubstitutionFixedPointPrefixResult(SubstitutionFixedPointPrefixRequest):
    """Exact fixed-point prefix from the least sufficient iterate."""

    prefix: FiniteWord
    least_iterate_depth: int = Field(ge=0, le=MAX_MORPHISM_OUTPUT_LENGTH)
    retained_prefix_lengths: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_MORPHISM_OUTPUT_LENGTH
    )
    complete: Literal[True] = True
    scope: Literal["FIRST_REQUESTED_LETTERS_OF_ONE_SIDED_FIXED_POINT"] = (
        "FIRST_REQUESTED_LETTERS_OF_ONE_SIDED_FIXED_POINT"
    )
    method: Literal["LEAST_TRUNCATED_SUBSTITUTION_ITERATE"] = (
        "LEAST_TRUNCATED_SUBSTITUTION_ITERATE"
    )

    @model_validator(mode="after")
    def bind_exact_fixed_point_prefix(self) -> Self:
        expected = fixed_point_prefix(self.source, self.prefix_length)
        if (
            self.prefix != expected.prefix
            or self.least_iterate_depth != expected.least_iterate_depth
            or self.retained_prefix_lengths != expected.retained_prefix_lengths
        ):
            raise _validation_error(
                "fixed_point_prefix_unbound",
                "fixed-point prefix is not bound to the request",
            )
        return self


__all__ = [
    "FactorsLengthRequest",
    "FactorsLengthResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "PeriodsRequest",
    "PeriodsResult",
    "SubstitutionDependencyGraphRequest",
    "SubstitutionDependencyGraphResult",
    "SubstitutionFixedPointPrefixRequest",
    "SubstitutionFixedPointPrefixResult",
    "SubstitutionPrimitivityProfileRequest",
    "SubstitutionPrimitivityProfileResult",
]
