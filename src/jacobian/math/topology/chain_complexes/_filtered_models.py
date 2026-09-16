"""Wire contracts for the associated graded of a filtered chain complex."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

MAX_FILTER_LEVELS = 8
MAX_FILTER_AMBIENT_DIMENSION = 32
MAX_FILTER_VECTORS_PER_GROUP = 64

Vector = tuple[str, ...]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"filtered_chain_complex.{reason}", message)


class FilteredSubspace(StrictModel):
    """One filtration subspace bound to its ambient chain group.

    ``vectors`` spans F_p C_n in ambient C_n coordinates; the empty tuple is
    the zero subspace. Entries use the canonical coefficient grammar of the
    retained complex: integers without leading zeros, reduced QQ fractions,
    and GF(p) residues in [0, p).
    """

    vectors: tuple[Vector, ...] = Field(
        max_length=MAX_FILTER_VECTORS_PER_GROUP,
        description=(
            "Spanning vectors of one filtration subspace in ambient chain "
            "coordinates; the empty tuple is the zero subspace."
        ),
    )


class FiltrationLevel(StrictModel):
    """One filtration level across every chain degree of the complex."""

    subspaces: tuple[FilteredSubspace, ...] = Field(
        description=(
            "One subspace per chain degree in increasing degree order, "
            "aligned with the retained complex basis sizes."
        )
    )


class FilteredChainComplexRequest(StrictModel):
    """A finite bounded increasing filtration of a based chain complex.

    Level 0 is the bottom of the filtration and the final level must be
    exhaustive; F_{-1} is definitionally zero so Gr_0 = F_0. The kernel
    establishes nesting, exhaustiveness, and differential compatibility.
    """

    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(
        min_length=1,
        max_length=MAX_FILTER_LEVELS,
        description=(
            "Increasing filtration levels from bottom to top; the final "
            "level must span every chain group."
        ),
    )

    @model_validator(mode="after")
    def require_structural_filtration(self) -> Self:
        from jacobian.math.topology.chain_complexes.values import (
            _require_rational_entry_grammar,
        )

        sizes = self.complex.basis_sizes
        if any(size > MAX_FILTER_AMBIENT_DIMENSION for size in sizes):
            raise _validation_error(
                "filtration_ambient_dimension_exceeded",
                "filtered chain complexes admit at most "
                f"{MAX_FILTER_AMBIENT_DIMENSION} basis vectors per chain group",
            )
        for level_index, level in enumerate(self.filtration):
            if len(level.subspaces) != len(sizes):
                raise _validation_error(
                    "filtration_degree_coverage_invalid",
                    f"filtration level {level_index} must carry one subspace "
                    "per chain degree of the retained complex",
                )
            for degree_index, subspace in enumerate(level.subspaces):
                for vector in subspace.vectors:
                    if len(vector) != sizes[degree_index]:
                        raise _validation_error(
                            "filtration_vector_axis_invalid",
                            f"level {level_index} degree {degree_index} vectors "
                            "must use ambient chain coordinates",
                        )
                    for entry in vector:
                        _require_rational_entry_grammar(
                            self.complex.coefficient_ring,
                            entry,
                            prime=self.complex.prime,
                        )
        return self


class GradedSquareLedgerEntry(StrictModel):
    """One replayed d_0^2 = 0 product inside a single filtration level."""

    filtration_level: int = Field(ge=0)
    degree: int
    product_rows: int = Field(ge=0)
    product_columns: int = Field(ge=0)
    nonzero_entries: Literal[0] = 0


class AssociatedGradedResult(StrictModel):
    """The associated graded complex Gr_p C_n with induced differentials.

    ``graded_differentials[p][k]`` is the dense induced map
    Gr_p C_{n+1} -> Gr_p C_n (rows = lower graded dimension, columns =
    upper), aligned with the retained complex degree interval.
    """

    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    graded_dimensions: tuple[tuple[int, ...], ...] = Field(min_length=1)
    quotient_representatives: tuple[tuple[tuple[Vector, ...], ...], ...] = Field(
        min_length=1
    )
    graded_differentials: tuple[tuple[tuple[Vector, ...], ...], ...] = ()
    differential_squared_zero: tuple[GradedSquareLedgerEntry, ...] = ()

    @model_validator(mode="after")
    def require_structural_grading(self) -> Self:
        sizes = self.complex.basis_sizes
        degrees = len(sizes)
        if len(self.graded_dimensions) != len(self.filtration):
            raise _validation_error(
                "graded_dimension_level_mismatch",
                "graded dimensions must cover every filtration level",
            )
        if len(self.quotient_representatives) != len(self.filtration):
            raise _validation_error(
                "representative_level_mismatch",
                "quotient representatives must cover every filtration level",
            )
        if len(self.graded_differentials) != len(self.filtration):
            raise _validation_error(
                "graded_differential_level_mismatch",
                "graded differentials must cover every filtration level",
            )
        for level, (dims, reps, diffs) in enumerate(
            zip(
                self.graded_dimensions,
                self.quotient_representatives,
                self.graded_differentials,
                strict=True,
            )
        ):
            if (
                len(dims) != degrees
                or any(dim < 0 for dim in dims)
                or any(dim > size for dim, size in zip(dims, sizes, strict=True))
            ):
                raise _validation_error(
                    "graded_dimension_degree_mismatch",
                    f"level {level} graded dimensions must cover every degree",
                )
            if len(reps) != degrees:
                raise _validation_error(
                    "representative_degree_mismatch",
                    f"level {level} representatives must cover every degree",
                )
            for degree, vectors in enumerate(reps):
                if len(vectors) != dims[degree] or any(
                    len(vector) != sizes[degree] for vector in vectors
                ):
                    raise _validation_error(
                        "representative_axis_invalid",
                        f"level {level} degree {degree} representatives must "
                        "match the graded dimension in ambient coordinates",
                    )
            if len(diffs) != max(0, degrees - 1):
                raise _validation_error(
                    "graded_differential_count_mismatch",
                    f"level {level} must carry one differential per degree pair",
                )
            for index, matrix in enumerate(diffs):
                if len(matrix) != dims[index] or any(
                    len(row) != dims[index + 1] for row in matrix
                ):
                    raise _validation_error(
                        "graded_differential_shape_mismatch",
                        f"level {level} differential {index} must have shape "
                        "graded_dim[n] x graded_dim[n+1]",
                    )
            expected_ledger = max(0, degrees - 2) if degrees > 1 else 0
            level_ledger = tuple(
                entry
                for entry in self.differential_squared_zero
                if entry.filtration_level == level
            )
            if len(level_ledger) != expected_ledger:
                raise _validation_error(
                    "graded_square_ledger_incomplete",
                    f"level {level} square-zero ledger must cover every "
                    "adjacent differential pair",
                )
        for degree in range(degrees):
            if sum(dims[degree] for dims in self.graded_dimensions) != sizes[degree]:
                raise _validation_error(
                    "graded_dimension_sum_mismatch",
                    "graded dimensions must sum to the retained chain rank "
                    "in every degree",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_FILTER_AMBIENT_DIMENSION",
    "MAX_FILTER_LEVELS",
    "MAX_FILTER_VECTORS_PER_GROUP",
    "AssociatedGradedResult",
    "FilteredChainComplexRequest",
    "FilteredSubspace",
    "FiltrationLevel",
    "GradedSquareLedgerEntry",
]
