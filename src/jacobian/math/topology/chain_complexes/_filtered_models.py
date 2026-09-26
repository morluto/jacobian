"""Wire contracts for the associated graded of a filtered chain complex."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.chain_complexes.values import (
    ChainCoefficient,
    ChainComplexValue,
)

MAX_FILTER_LEVELS = 8
MAX_FILTER_AMBIENT_DIMENSION = 32
MAX_FILTER_VECTORS_PER_GROUP = 64
MAX_SPECTRAL_PAGE = 4

Vector = tuple[ChainCoefficient, ...]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"filtered_chain_complex.{reason}", message)


class FilteredSubspace(StrictModel):
    """One filtration subspace bound to its ambient chain group.

    ``vectors`` spans F_p C_n in ambient C_n coordinates; the empty tuple is
    the zero subspace. Entries are native integers or Fractions interpreted
    in the coefficient ring of the retained complex.
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
            _require_coefficient_scalar,
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
                        _require_coefficient_scalar(
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


class SpectralPageStatus(StrEnum):
    """Convergence accounting for one computed spectral-sequence page."""

    STABILIZED = "STABILIZED"
    ACTIVE = "ACTIVE"
    TRUNCATED = "TRUNCATED"


class SpectralPageRequest(StrictModel):
    """Compute the E^r page of a bounded filtered chain complex.

    Page 0 is the associated graded itself; page ``r >= 1`` is the
    bigraded quotient ``Z^r / D^r`` with the induced ``d^r``
    differentials. Pages above ``MAX_SPECTRAL_PAGE`` are refused at the
    schema boundary.
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
    page: int = Field(
        ge=0,
        le=MAX_SPECTRAL_PAGE,
        description=(
            "Requested page r; 0 returns the associated graded with its "
            "d^0 differentials and later pages carry d^r of bidegree "
            "(-r, r - 1)."
        ),
    )


class SpectralDifferential(StrictModel):
    """One dense d^r differential between two page bidegrees.

    ``rows`` is the target bidegree dimension and ``columns`` the source
    bidegree dimension; ``entries`` is row-major in the canonical
    coefficient grammar of the retained complex.
    """

    source_level: int = Field(ge=0)
    source_degree: int
    target_level: int = Field(ge=0)
    target_degree: int
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    entries: tuple[Vector, ...] = ()


class SpectralSquareLedgerEntry(StrictModel):
    """One replayed d^r d^r = 0 product over a composable record pair."""

    source_level: int = Field(ge=0)
    source_degree: int
    middle_level: int = Field(ge=0)
    middle_degree: int
    product_rows: int = Field(ge=0)
    product_columns: int = Field(ge=0)
    nonzero_entries: Literal[0] = 0


class SpectralPageResult(StrictModel):
    """The E^r page of a bounded filtered chain complex.

    ``page_dimensions[p][k]`` is ``dim E^r_{p, n - p}`` where ``k``
    indexes the retained complex degree interval and ``n`` is the actual
    chain degree. ``differentials`` carries every ``d^r`` map of
    bidegree ``(-r, r - 1)`` inside the window, ``next_page_dimensions``
    previews ``dim E^{r+1}`` as the homology of ``(E^r, d^r)``, and
    ``page_status`` reports stabilization accounting only: ``STABILIZED``
    means every longer differential provably vanishes so ``E^r`` is the
    limit page, ``TRUNCATED`` means the page budget ceiling was reached
    before stabilization could be decided, and ``ACTIVE`` means a later
    page within budget may still differ. Truncation is never a
    mathematical conclusion about the limit page.
    """

    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    page: int = Field(ge=0)
    max_page: int = Field(ge=0)
    level_count: int = Field(ge=1)
    degree_min: int
    degree_max: int
    page_dimensions: tuple[tuple[int, ...], ...] = Field(min_length=1)
    page_representatives: tuple[tuple[tuple[Vector, ...], ...], ...] = Field(
        min_length=1
    )
    differentials: tuple[SpectralDifferential, ...] = ()
    differential_squared_zero: tuple[SpectralSquareLedgerEntry, ...] = ()
    next_page_dimensions: tuple[tuple[int, ...], ...] = Field(min_length=1)
    page_status: SpectralPageStatus

    @model_validator(mode="after")
    def require_structural_page(self) -> Self:  # noqa: C901
        sizes = self.complex.basis_sizes
        degrees = len(sizes)
        levels = len(self.filtration)
        if self.page > MAX_SPECTRAL_PAGE:
            raise _validation_error(
                "spectral_page_above_budget",
                f"spectral pages admit at most page {MAX_SPECTRAL_PAGE}",
            )
        if self.max_page != MAX_SPECTRAL_PAGE:
            raise _validation_error(
                "spectral_window_mismatch",
                "the result window must state the admitted page budget",
            )
        if self.level_count != levels:
            raise _validation_error(
                "spectral_level_count_mismatch",
                "the level count must match the retained filtration",
            )
        if (self.degree_min, self.degree_max) != (
            self.complex.degree_min,
            self.complex.degree_max,
        ):
            raise _validation_error(
                "spectral_degree_interval_mismatch",
                "the degree window must match the retained complex",
            )
        for label, grid in (
            ("page dimensions", self.page_dimensions),
            ("next page dimensions", self.next_page_dimensions),
        ):
            if len(grid) != levels or any(len(row) != degrees for row in grid):
                raise _validation_error(
                    "spectral_dimension_window_mismatch",
                    f"{label} must cover every filtration level and chain degree",
                )
            for row in grid:
                if any(dim < 0 for dim in row):
                    raise _validation_error(
                        "spectral_dimension_negative",
                        f"{label} must be nonnegative",
                    )
        if len(self.page_representatives) != levels:
            raise _validation_error(
                "spectral_representative_level_mismatch",
                "page representatives must cover every filtration level",
            )
        for level, (dims, reps) in enumerate(
            zip(self.page_dimensions, self.page_representatives, strict=True)
        ):
            if len(reps) != degrees:
                raise _validation_error(
                    "spectral_representative_degree_mismatch",
                    f"level {level} representatives must cover every degree",
                )
            for degree, vectors in enumerate(reps):
                if len(vectors) != dims[degree] or any(
                    len(vector) != sizes[degree] for vector in vectors
                ):
                    raise _validation_error(
                        "spectral_representative_axis_invalid",
                        f"level {level} degree {degree} representatives must "
                        "match the page dimension in ambient coordinates",
                    )
        expected_sources = {
            (level, self.complex.degree_min + index)
            for level in range(levels)
            for index in range(degrees)
            if index > 0 and level - self.page >= 0
        }
        actual_sources = {
            (record.source_level, record.source_degree) for record in self.differentials
        }
        if actual_sources != expected_sources:
            raise _validation_error(
                "spectral_differential_window_mismatch",
                "differentials must cover exactly the in-window d^r sources",
            )
        for record in self.differentials:
            source_index = record.source_degree - self.complex.degree_min
            target_index = record.target_degree - self.complex.degree_min
            if record.target_level != record.source_level - self.page:
                raise _validation_error(
                    "spectral_differential_bidegree_invalid",
                    "every d^r differential shifts the filtration level by -r",
                )
            if target_index != source_index - 1:
                raise _validation_error(
                    "spectral_differential_degree_invalid",
                    "every d^r differential shifts the total degree by -1",
                )
            expected_rows = self.page_dimensions[record.target_level][target_index]
            expected_columns = self.page_dimensions[record.source_level][source_index]
            if record.rows != expected_rows or record.columns != expected_columns:
                raise _validation_error(
                    "spectral_differential_shape_mismatch",
                    "differential axes must match the page dimensions",
                )
            if len(record.entries) != record.rows or any(
                len(row) != record.columns for row in record.entries
            ):
                raise _validation_error(
                    "spectral_differential_entries_mismatch",
                    "differential entries must fill the declared axes",
                )
        by_source = {
            (record.source_level, record.source_degree): record
            for record in self.differentials
        }
        expected_ledger = {
            (record.source_level, record.source_degree)
            for record in self.differentials
            if (record.target_level, record.target_degree) in by_source
        }
        actual_ledger = {
            (entry.source_level, entry.source_degree)
            for entry in self.differential_squared_zero
        }
        if actual_ledger != expected_ledger:
            raise _validation_error(
                "spectral_square_ledger_incomplete",
                "the square-zero ledger must cover every composable "
                "differential pair exactly once",
            )
        for entry in self.differential_squared_zero:
            first = by_source[(entry.source_level, entry.source_degree)]
            second = by_source[(first.target_level, first.target_degree)]
            if (entry.middle_level, entry.middle_degree) != (
                first.target_level,
                first.target_degree,
            ):
                raise _validation_error(
                    "spectral_square_middle_mismatch",
                    "each ledger entry must name the shared middle bidegree",
                )
            if entry.product_rows != second.rows or (
                entry.product_columns != first.columns
            ):
                raise _validation_error(
                    "spectral_square_product_shape_mismatch",
                    "each ledger product shape must match its outer axes",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_FILTER_AMBIENT_DIMENSION",
    "MAX_FILTER_LEVELS",
    "MAX_FILTER_VECTORS_PER_GROUP",
    "MAX_SPECTRAL_PAGE",
    "AssociatedGradedResult",
    "FilteredChainComplexRequest",
    "FilteredSubspace",
    "FiltrationLevel",
    "GradedSquareLedgerEntry",
    "SpectralDifferential",
    "SpectralPageRequest",
    "SpectralPageResult",
    "SpectralPageStatus",
    "SpectralSquareLedgerEntry",
]
