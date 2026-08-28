"""Typed wire contracts for exact combinatorics-on-words operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.languages.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyGraph,
    WordMorphism,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"word.{reason}", message)


class FactorsLengthRequest(StrictModel):
    """Enumerate all distinct factors of one valid length."""

    word: FiniteWord
    factor_length: int = Field(ge=0)


class FactorsLengthResult(FactorsLengthRequest):
    """Complete factor enumeration, ordered by first occurrence."""

    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]
    multiplicities: tuple[int, ...]
    first_occurrence: tuple[int, ...]
    distinct_count: int = Field(ge=0)
    scope: Literal["ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"] = (
        "ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"
    )

    @model_validator(mode="after")
    def require_structural_factor_enumeration(self) -> Self:
        if not (
            len(self.factors)
            == len(self.occurrences)
            == len(self.multiplicities)
            == len(self.first_occurrence)
            == self.distinct_count
        ):
            raise _validation_error(
                "factor_result_shape",
                "factor result fields must have one entry per factor",
            )
        for factor, positions, multiplicity, first in zip(
            self.factors,
            self.occurrences,
            self.multiplicities,
            self.first_occurrence,
            strict=True,
        ):
            if len(factor) != self.factor_length or any(
                letter not in self.word.alphabet for letter in factor
            ):
                raise _validation_error(
                    "factor_result_factor",
                    "each factor must be a requested-length word",
                )
            if not positions or positions != tuple(sorted(set(positions))):
                raise _validation_error(
                    "factor_result_positions",
                    "each factor must retain nonempty increasing occurrence positions",
                )
            if any(
                position < 0 or position + self.factor_length > len(self.word.letters)
                for position in positions
            ):
                raise _validation_error(
                    "factor_result_positions",
                    "factor occurrence position is outside the word",
                )
            if multiplicity != len(positions) or first != positions[0]:
                raise _validation_error(
                    "factor_result_summary",
                    "factor summaries must agree with occurrences",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FactorsLengthRequest,
        *,
        factors: tuple[tuple[str, ...], ...],
        occurrences: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            word=request.word,
            factor_length=request.factor_length,
            factors=factors,
            occurrences=occurrences,
            multiplicities=tuple(len(indices) for indices in occurrences),
            first_occurrence=tuple(indices[0] for indices in occurrences),
            distinct_count=len(factors),
        )


class PeriodsRequest(StrictModel):
    """Compute all overlap periods of a finite word."""

    word: FiniteWord


class PeriodsResult(PeriodsRequest):
    """Complete overlap-period profile and proper-power primitivity."""

    periods: tuple[int, ...]
    least_period: int = Field(ge=0)
    is_primitive: bool
    primitive_convention: Literal["NOT_A_NONTRIVIAL_INTEGER_POWER"] = (
        "NOT_A_NONTRIVIAL_INTEGER_POWER"
    )
    empty_word_convention: Literal["NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"] = (
        "NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"
    )

    @model_validator(mode="after")
    def require_structural_period_profile(self) -> Self:
        if self.periods != tuple(sorted(set(self.periods))) or any(
            period <= 0 or period > len(self.word.letters) for period in self.periods
        ):
            raise _validation_error(
                "period_result_periods",
                "periods must be increasing positive word offsets",
            )
        if (self.periods and self.least_period != self.periods[0]) or (
            not self.periods and self.least_period != 0
        ):
            raise _validation_error(
                "period_result_least",
                "least_period must be the first reported period, or zero",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PeriodsRequest,
        *,
        periods: tuple[int, ...],
        least_period: int,
        is_primitive: bool,
    ) -> Self:
        return cls.model_construct(
            word=request.word,
            periods=periods,
            least_period=least_period,
            is_primitive=is_primitive,
        )


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix of a finite word morphism."""

    morphism: WordMorphism


class IncidenceMatrixResult(IncidenceMatrixRequest):
    """Exact target-by-source incidence matrix."""

    matrix: tuple[tuple[int, ...], ...]
    orientation: Literal["ROWS_TARGET_COLUMNS_SOURCE"] = "ROWS_TARGET_COLUMNS_SOURCE"

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if len(self.matrix) != len(self.morphism.target_alphabet) or any(
            len(row) != len(self.morphism.source_alphabet)
            or any(entry < 0 for entry in row)
            for row in self.matrix
        ):
            raise _validation_error(
                "incidence_matrix_shape",
                "matrix must be nonnegative target-by-source counts",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: IncidenceMatrixRequest, matrix: tuple[tuple[int, ...], ...]
    ) -> Self:
        return cls.model_construct(morphism=request.morphism, matrix=matrix)


class SubstitutionDependencyGraphRequest(StrictModel):
    """Construct the exact dependency graph of one bounded substitution."""

    substitution: Substitution


class SubstitutionDependencyGraphResult(SubstitutionDependencyGraphRequest):
    """Exact source-bound letter graph, including every occurrence position."""

    graph: SubstitutionDependencyGraph
    edge_convention: Literal["SOURCE_TO_OCCURRING_TARGET"] = (
        "SOURCE_TO_OCCURRING_TARGET"
    )

    @classmethod
    def _from_kernel(
        cls,
        request: SubstitutionDependencyGraphRequest,
        graph: SubstitutionDependencyGraph,
    ) -> Self:
        return cls.model_construct(substitution=request.substitution, graph=graph)


class SubstitutionPrimitivityProfileRequest(StrictModel):
    """Decide primitivity from a canonical substitution dependency graph."""

    dependency_graph: SubstitutionDependencyGraph


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

    @model_validator(mode="after")
    def require_structural_primitivity_profile(self) -> Self:
        alphabet = self.dependency_graph.substitution.morphism.source_alphabet
        flattened = tuple(
            symbol
            for component in self.strongly_connected_components
            for symbol in component
        )
        if (
            not self.strongly_connected_components
            or any(not component for component in self.strongly_connected_components)
            or set(flattened) != set(alphabet)
            or len(flattened) != len(set(flattened))
        ):
            raise _validation_error(
                "primitivity_components",
                "components must partition the substitution alphabet",
            )
        if self.irreducible != (len(self.strongly_connected_components) == 1):
            raise _validation_error(
                "primitivity_irreducible", "irreducible must match component count"
            )
        if self.primitive:
            if (
                self.aperiodic is not True
                or self.least_positive_power is None
                or self.obstruction != "NONE"
            ):
                raise _validation_error(
                    "primitivity_positive",
                    "a primitive profile needs a positive aperiodic witness",
                )
        elif self.least_positive_power is not None or self.obstruction == "NONE":
            raise _validation_error(
                "primitivity_negative",
                "a nonprimitive profile cannot claim a positive power",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubstitutionPrimitivityProfileRequest,
        *,
        strongly_connected_components: tuple[tuple[str, ...], ...],
        irreducible: bool,
        aperiodic: bool | None,
        primitive: bool,
        least_positive_power: int | None,
        exponent_upper_bound: int,
        obstruction: Literal[
            "NONE", "REDUCIBLE_DEPENDENCY_GRAPH", "PERIODIC_DEPENDENCY_GRAPH"
        ],
    ) -> Self:
        return cls.model_construct(
            dependency_graph=request.dependency_graph,
            strongly_connected_components=strongly_connected_components,
            irreducible=irreducible,
            aperiodic=aperiodic,
            primitive=primitive,
            least_positive_power=least_positive_power,
            exponent_upper_bound=exponent_upper_bound,
            obstruction=obstruction,
        )


class SubstitutionFixedPointPrefixRequest(StrictModel):
    """Request one bounded prefix of a certified prolongable substitution."""

    source: ProlongableSubstitution
    prefix_length: int = Field(ge=0, le=MAX_MORPHISM_OUTPUT_LENGTH)


class SubstitutionFixedPointPrefixResult(SubstitutionFixedPointPrefixRequest):
    """Exact fixed-point prefix from the least sufficient iterate."""

    prefix: FiniteWord
    least_iterate_depth: int = Field(ge=0, le=MAX_MORPHISM_OUTPUT_LENGTH)
    retained_prefix_lengths: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_MORPHISM_OUTPUT_LENGTH
    )
    scope: Literal["FIRST_REQUESTED_LETTERS_OF_ONE_SIDED_FIXED_POINT"] = (
        "FIRST_REQUESTED_LETTERS_OF_ONE_SIDED_FIXED_POINT"
    )

    @model_validator(mode="after")
    def require_structural_fixed_point_prefix(self) -> Self:
        if (
            self.prefix.alphabet != self.source.substitution.morphism.target_alphabet
            or len(self.prefix.letters) != self.prefix_length
        ):
            raise _validation_error(
                "fixed_point_prefix_shape",
                "prefix must have the requested target-alphabet length",
            )
        if (
            self.retained_prefix_lengths[0] != min(1, self.prefix_length)
            or self.retained_prefix_lengths[-1] != self.prefix_length
            or any(
                left > right
                for left, right in zip(
                    self.retained_prefix_lengths,
                    self.retained_prefix_lengths[1:],
                    strict=False,
                )
            )
            or self.least_iterate_depth != len(self.retained_prefix_lengths) - 1
        ):
            raise _validation_error(
                "fixed_point_prefix_ledger",
                "retained prefix lengths must describe the iterate ledger",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubstitutionFixedPointPrefixRequest,
        *,
        prefix: FiniteWord,
        least_iterate_depth: int,
        retained_prefix_lengths: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            source=request.source,
            prefix_length=request.prefix_length,
            prefix=prefix,
            least_iterate_depth=least_iterate_depth,
            retained_prefix_lengths=retained_prefix_lengths,
        )


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
