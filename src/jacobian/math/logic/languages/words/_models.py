"""Typed wire contracts for exact combinatorics-on-words operations."""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    MAX_WORD_LENGTH,
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyGraph,
    WordMorphism,
)
from jacobian.math.matrices.values import IntegerMatrix


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"word.{reason}", message)


def _json_string_size(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _json_array_size(item_sizes: list[int]) -> int:
    return 2 + sum(item_sizes) + max(len(item_sizes) - 1, 0)


def require_word_family_output(
    word: FiniteWord, family_name: Literal["prefixes", "suffixes"]
) -> None:
    """Admit the complete source-bound family before materializing its cells."""

    symbol_sizes = {symbol: _json_string_size(symbol) for symbol in word.alphabet}
    alphabet_size = _json_array_size([symbol_sizes[symbol] for symbol in word.alphabet])
    letter_sizes = [symbol_sizes[letter] for letter in word.letters]
    source_letters_size = _json_array_size(letter_sizes)
    source_word_size = (
        len(b'{"alphabet":')
        + alphabet_size
        + len(b',"letters":')
        + source_letters_size
        + 1
    )
    family_count = len(letter_sizes) + 1
    if family_name == "prefixes":
        family_letter_size = sum(
            (len(letter_sizes) - index) * size
            for index, size in enumerate(letter_sizes)
        )
    else:
        family_letter_size = sum(
            (index + 1) * size for index, size in enumerate(letter_sizes)
        )
    family_letters_size = (
        2 * family_count
        + family_letter_size
        + len(letter_sizes) * max(len(letter_sizes) - 1, 0) // 2
    )
    family_member_static_size = (
        len(b'{"alphabet":') + alphabet_size + len(b',"letters":') + 1
    )
    family_members_size = (
        family_count * family_member_static_size
        + family_letters_size
        + max(family_count - 1, 0)
    )
    estimated_output_bytes = (
        len(b'{"word":')
        + source_word_size
        + len(f',"{family_name}":['.encode())
        + family_members_size
        + 2
    )
    limit = CanonicalLimits().max_output_bytes
    if estimated_output_bytes > limit:
        raise OperationResourceAdmissionError(
            location=("word", family_name),
            code="word.family_output_bytes",
            message=(
                f"{family_name} output requires approximately {estimated_output_bytes} "
                f"UTF-8 bytes, exceeding the {limit}-byte output bound"
            ),
        )


class WordFamilyRequest(StrictModel):
    """Request all prefixes or suffixes of one finite word."""

    word: FiniteWord


class WordPrefixesResult(WordFamilyRequest):
    """Complete prefix family, including the empty prefix."""

    prefixes: tuple[FiniteWord, ...] = Field(max_length=MAX_WORD_LENGTH + 1)

    @model_validator(mode="after")
    def require_prefix_axis(self) -> Self:
        letters = self.word.letters
        if len(self.prefixes) != len(letters) + 1 or any(
            prefix.alphabet != self.word.alphabet or len(prefix.letters) != index
            for index, prefix in enumerate(self.prefixes)
        ):
            raise _validation_error(
                "prefix_family_shape",
                "prefixes must be the complete ordered prefix family",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: WordFamilyRequest, prefixes: tuple[FiniteWord, ...]
    ) -> Self:
        return cls.model_construct(word=request.word, prefixes=prefixes)


class WordSuffixesResult(WordFamilyRequest):
    """Complete suffix family, including the empty suffix."""

    suffixes: tuple[FiniteWord, ...] = Field(max_length=MAX_WORD_LENGTH + 1)

    @model_validator(mode="after")
    def require_suffix_axis(self) -> Self:
        letters = self.word.letters
        if len(self.suffixes) != len(letters) + 1 or any(
            suffix.alphabet != self.word.alphabet
            or len(suffix.letters) != len(letters) - index
            for index, suffix in enumerate(self.suffixes)
        ):
            raise _validation_error(
                "suffix_family_shape",
                "suffixes must be the complete ordered suffix family",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: WordFamilyRequest, suffixes: tuple[FiniteWord, ...]
    ) -> Self:
        return cls.model_construct(word=request.word, suffixes=suffixes)


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
        for factor, positions in zip(
            self.factors,
            self.occurrences,
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

    matrix: IntegerMatrix
    orientation: Literal["ROWS_TARGET_COLUMNS_SOURCE"] = "ROWS_TARGET_COLUMNS_SOURCE"

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if (
            self.matrix.row_count != len(self.morphism.target_alphabet)
            or self.matrix.column_count != len(self.morphism.source_alphabet)
            or any(int(entry) < 0 for row in self.matrix.entries for entry in row)
        ):
            raise _validation_error(
                "incidence_matrix_shape",
                "matrix must be nonnegative target-by-source counts",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: IncidenceMatrixRequest, matrix: IntegerMatrix
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
    "WordFamilyRequest",
    "WordPrefixesResult",
    "WordSuffixesResult",
]
