"""Typed wire contracts for exact combinatorics-on-words operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    MAX_WORD_FAMILY_CELLS,
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyGraph,
    WordMorphism,
)
from jacobian.math.matrices.values import IntegerMatrix


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"word.{reason}", message)


def require_word_family_allocation(
    word: FiniteWord, family_name: Literal["prefixes", "suffixes"]
) -> None:
    """Admit the complete source-bound family before materializing its cells."""

    word_length = len(word.letters)
    family_count = word_length + 1
    family_cells = (
        family_count * len(word.alphabet) + word_length * (word_length + 1) // 2
    )
    if family_cells > MAX_WORD_FAMILY_CELLS:
        raise OperationResourceAdmissionError(
            location=("word", family_name),
            code="word.family_cells",
            message=(
                f"{family_name} materialization requires {family_cells} cells, "
                f"exceeding the {MAX_WORD_FAMILY_CELLS}-cell allocation bound"
            ),
        )


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


class MorphismApplyRequest(StrictModel):
    """Apply one morphism to a finite word over its source alphabet."""

    morphism: WordMorphism
    word: FiniteWord


class MorphismApplyResult(MorphismApplyRequest):
    """Exact image word over the morphism target alphabet."""

    image: FiniteWord

    @model_validator(mode="after")
    def require_image_axis(self) -> Self:
        if self.word.alphabet != self.morphism.source_alphabet:
            raise _validation_error(
                "morphism_apply_axis_mismatch",
                "word alphabet must equal the morphism source alphabet",
            )
        if self.image.alphabet != self.morphism.target_alphabet:
            raise _validation_error(
                "morphism_apply_image_axis",
                "image alphabet must equal the morphism target alphabet",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: MorphismApplyRequest, image: FiniteWord) -> Self:
        return cls.model_construct(
            morphism=request.morphism, word=request.word, image=image
        )


class MorphismComposeRequest(StrictModel):
    """Compose two axis-compatible morphisms."""

    first: WordMorphism
    second: WordMorphism


class MorphismComposeResult(MorphismComposeRequest):
    """Exact composite with explicit source/target axes."""

    composite: WordMorphism

    @model_validator(mode="after")
    def require_composite_axis(self) -> Self:
        if self.first.target_alphabet != self.second.source_alphabet:
            raise _validation_error(
                "morphism_compose_axis_mismatch",
                "first target alphabet must equal second source alphabet",
            )
        if (
            self.composite.source_alphabet != self.first.source_alphabet
            or self.composite.target_alphabet != self.second.target_alphabet
        ):
            raise _validation_error(
                "morphism_composite_axis",
                "composite must carry the outer source/target alphabets",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: MorphismComposeRequest, composite: WordMorphism
    ) -> Self:
        return cls.model_construct(
            first=request.first, second=request.second, composite=composite
        )


class MorphismPowerRequest(StrictModel):
    """Iterate one endomorphism a bounded number of times."""

    morphism: WordMorphism
    exponent: int = Field(ge=0, le=16)


class MorphismPowerResult(MorphismPowerRequest):
    """Exact morphism power with explicit axes."""

    power: WordMorphism

    @model_validator(mode="after")
    def require_power_axis(self) -> Self:
        if self.morphism.source_alphabet != self.morphism.target_alphabet:
            raise _validation_error(
                "morphism_power_not_endomorphism",
                "morphism power requires identical source and target alphabets",
            )
        if (
            self.power.source_alphabet != self.morphism.source_alphabet
            or self.power.target_alphabet != self.morphism.target_alphabet
        ):
            raise _validation_error(
                "morphism_power_axis",
                "power must preserve the endomorphism alphabet",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: MorphismPowerRequest, power: WordMorphism) -> Self:
        return cls.model_construct(
            morphism=request.morphism, exponent=request.exponent, power=power
        )


class MorphismIterateRequest(StrictModel):
    """Apply one endomorphism to a word a bounded number of times."""

    morphism: WordMorphism
    word: FiniteWord
    steps: int = Field(ge=0, le=16)


class MorphismIterateResult(MorphismIterateRequest):
    """Exact stepped image over the endomorphism alphabet."""

    image: FiniteWord

    @model_validator(mode="after")
    def require_iterate_axis(self) -> Self:
        if self.morphism.source_alphabet != self.morphism.target_alphabet:
            raise _validation_error(
                "morphism_iterate_not_endomorphism",
                "iteration requires identical source and target alphabets",
            )
        if self.word.alphabet != self.morphism.source_alphabet:
            raise _validation_error(
                "morphism_iterate_axis_mismatch",
                "word alphabet must equal the morphism source alphabet",
            )
        if self.image.alphabet != self.morphism.target_alphabet:
            raise _validation_error(
                "morphism_iterate_image_axis",
                "image alphabet must equal the morphism target alphabet",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: MorphismIterateRequest, image: FiniteWord) -> Self:
        return cls.model_construct(
            morphism=request.morphism,
            word=request.word,
            steps=request.steps,
            image=image,
        )


class MorphismImageLengthsRequest(StrictModel):
    """Return image lengths bound to the morphism source axis."""

    morphism: WordMorphism


class MorphismImageLengthsResult(MorphismImageLengthsRequest):
    """One image length per source symbol, in source order."""

    lengths: tuple[int, ...]
    total_length: int = Field(ge=0)
    max_length: int = Field(ge=0)

    @model_validator(mode="after")
    def require_lengths_axis(self) -> Self:
        if len(self.lengths) != len(self.morphism.source_alphabet):
            raise _validation_error(
                "morphism_lengths_axis_mismatch",
                "lengths must have one entry per source symbol",
            )
        if tuple(self.lengths) != tuple(
            len(image) for image in self.morphism.images
        ):
            raise _validation_error(
                "morphism_lengths_mismatch",
                "lengths must equal the retained morphism image lengths",
            )
        if self.total_length != sum(self.lengths) or (
            self.lengths and self.max_length != max(self.lengths)
        ):
            raise _validation_error(
                "morphism_lengths_summary_mismatch",
                "total/max summaries must match the retained lengths",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: MorphismImageLengthsRequest, lengths: tuple[int, ...]
    ) -> Self:
        return cls.model_construct(
            morphism=request.morphism,
            lengths=lengths,
            total_length=sum(lengths),
            max_length=max(lengths, default=0),
        )


class FactorComplexityRequest(StrictModel):
    """Complexity prefix p(0)..p(max) of one supplied finite word."""

    word: FiniteWord
    max_order: int = Field(ge=0)


class FactorComplexityResult(FactorComplexityRequest):
    """Complete factor families with identity, scoped to the supplied word."""

    complexity: tuple[int, ...]
    families: tuple[tuple[tuple[str, ...], ...], ...]
    scope: Literal["FACTORS_OF_SUPPLIED_PREFIX_ONLY"] = (
        "FACTORS_OF_SUPPLIED_PREFIX_ONLY"
    )

    @model_validator(mode="after")
    def require_complexity_shape(self) -> Self:
        if len(self.complexity) != self.max_order + 1:
            raise _validation_error(
                "complexity_count_mismatch",
                "complexity must have one entry per order through max_order",
            )
        if len(self.families) != self.max_order + 1:
            raise _validation_error(
                "complexity_family_mismatch",
                "families must have one entry per order through max_order",
            )
        for order, (count, family) in enumerate(
            zip(self.complexity, self.families, strict=True)
        ):
            if count != len(family):
                raise _validation_error(
                    "complexity_family_count",
                    "each complexity entry must equal its family size",
                )
            if any(len(factor) != order for factor in family):
                raise _validation_error(
                    "complexity_family_length",
                    "every family member must have the declared order length",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FactorComplexityRequest,
        *,
        complexity: tuple[int, ...],
        families: tuple[tuple[tuple[str, ...], ...], ...],
    ) -> Self:
        return cls.model_construct(
            word=request.word,
            max_order=request.max_order,
            complexity=complexity,
            families=families,
        )


class RauzyGraphEdge(StrictModel):
    source: tuple[str, ...]
    label: tuple[str, ...]
    target: tuple[str, ...]
    occurrences: tuple[int, ...]


class RauzyGraphRequest(StrictModel):
    """Rauzy graph of one supplied word at one declared order."""

    word: FiniteWord
    order: int = Field(ge=0)


class RauzyGraphResult(RauzyGraphRequest):
    """Labeled vertices/edges scoped to the supplied word."""

    vertices: tuple[tuple[str, ...], ...]
    edges: tuple[RauzyGraphEdge, ...]
    scope: Literal["RAUZY_OF_SUPPLIED_PREFIX_ONLY"] = (
        "RAUZY_OF_SUPPLIED_PREFIX_ONLY"
    )

    @model_validator(mode="after")
    def require_rauzy_shape(self) -> Self:
        for vertex in self.vertices:
            if len(vertex) != self.order or any(
                letter not in self.word.alphabet for letter in vertex
            ):
                raise _validation_error(
                    "rauzy_vertex_shape",
                    "vertices must be order-length words over the alphabet",
                )
        for edge in self.edges:
            if len(edge.label) != self.order + 1:
                raise _validation_error(
                    "rauzy_edge_label",
                    "edge labels must have length order plus one",
                )
            if edge.source != edge.label[:-1] or edge.target != edge.label[1:]:
                raise _validation_error(
                    "rauzy_edge_endpoints",
                    "edge endpoints must be the label prefix and suffix",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RauzyGraphRequest,
        *,
        vertices: tuple[tuple[str, ...], ...],
        edges: tuple[RauzyGraphEdge, ...],
    ) -> Self:
        return cls.model_construct(
            word=request.word, order=request.order, vertices=vertices, edges=edges
        )


class SubstitutionFactorComplexityRequest(StrictModel):
    """Factor complexity of a fixed-point prefix with completion data."""

    source: ProlongableSubstitution
    prefix_length: int = Field(ge=0, le=MAX_MORPHISM_OUTPUT_LENGTH)
    max_order: int = Field(ge=0)


class SubstitutionFactorComplexityResult(SubstitutionFactorComplexityRequest):
    prefix: FiniteWord
    least_iterate_depth: int = Field(ge=0)
    complexity: tuple[int, ...]
    families: tuple[tuple[tuple[str, ...], ...], ...]
    scope: Literal["FACTORS_OF_RETAINED_PREFIX_ONLY"] = (
        "FACTORS_OF_RETAINED_PREFIX_ONLY"
    )

    @classmethod
    def _from_kernel(
        cls,
        request: SubstitutionFactorComplexityRequest,
        *,
        prefix: FiniteWord,
        least_iterate_depth: int,
        complexity: tuple[int, ...],
        families: tuple[tuple[tuple[str, ...], ...], ...],
    ) -> Self:
        return cls.model_construct(
            source=request.source,
            prefix_length=request.prefix_length,
            max_order=request.max_order,
            prefix=prefix,
            least_iterate_depth=least_iterate_depth,
            complexity=complexity,
            families=families,
        )


__all__ = [
    "FactorComplexityRequest",
    "FactorComplexityResult",
    "FactorsLengthRequest",
    "FactorsLengthResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "MorphismApplyRequest",
    "MorphismApplyResult",
    "MorphismComposeRequest",
    "MorphismComposeResult",
    "MorphismImageLengthsRequest",
    "MorphismImageLengthsResult",
    "MorphismIterateRequest",
    "MorphismIterateResult",
    "MorphismPowerRequest",
    "MorphismPowerResult",
    "PeriodsRequest",
    "PeriodsResult",
    "RauzyGraphRequest",
    "RauzyGraphResult",
    "SubstitutionDependencyGraphRequest",
    "SubstitutionDependencyGraphResult",
    "SubstitutionFactorComplexityRequest",
    "SubstitutionFactorComplexityResult",
    "SubstitutionFixedPointPrefixRequest",
    "SubstitutionFixedPointPrefixResult",
    "SubstitutionPrimitivityProfileRequest",
    "SubstitutionPrimitivityProfileResult",
]
