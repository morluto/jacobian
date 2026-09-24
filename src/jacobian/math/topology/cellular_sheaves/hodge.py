"""Exact Hodge Laplacians for cellular sheaves with standard stalk metrics."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._kernel import (
    _admit_field,
    _cochain_nullspace,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_HODGE_OUTPUT_CHARS,
    MAX_SHEAF_TOTAL_STALK_RANK,
    FiniteCellularSheaf,
    SheafHodgeRequest,
    SheafHodgeResult,
    SheafScalar,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.operations import sheaf_cohomology

_MAX_HODGE_MATRIX_CELLS = 65_536
_MAX_HODGE_CUBIC_WORK = 4_000_000


def _admit_hodge_work(sheaf: FiniteCellularSheaf, dimensions: tuple[int, ...]) -> None:
    if sum(dimensions) > MAX_SHEAF_TOTAL_STALK_RANK:
        raise OperationResourceAdmissionError(
            location=("sheaf",),
            code="topology.cellular_sheaf.hodge.total_stalk_rank",
            message="the cellular cochain complex exceeds its admitted coordinate bound",
        )
    matrix_cells = sum(size * size for size in dimensions)
    cubic_work = sum(
        size
        * size
        * (
            (dimensions[degree - 1] if degree > 0 else 0)
            + (dimensions[degree + 1] if degree + 1 < len(dimensions) else 0)
        )
        for degree, size in enumerate(dimensions)
    )
    if matrix_cells > _MAX_HODGE_MATRIX_CELLS or cubic_work > _MAX_HODGE_CUBIC_WORK:
        raise OperationResourceAdmissionError(
            location=("sheaf",),
            code="topology.cellular_sheaf.hodge.work_bound",
            message="Hodge matrices exceed the admitted exact matrix-work envelope",
        )
    _admit_hodge_scalars_and_output(sheaf, dimensions, matrix_cells)


def _admit_hodge_scalars_and_output(
    sheaf: FiniteCellularSheaf,
    dimensions: tuple[int, ...],
    matrix_cells: int,
) -> None:
    source_chars = len(sheaf.model_dump_json())
    max_scalar_digits = 1
    for restriction in (*sheaf.cover_restrictions, *sheaf.derived_restrictions):
        for row in restriction.entries:
            for entry in row:
                if not isinstance(entry, CanonicalRational):
                    raise OperationDomainValidationError(
                        location=("sheaf", "cover_restrictions"),
                        code="topology.cellular_sheaf.hodge.scalar_invalid",
                        message="rational restriction entries must be CanonicalRational values",
                    )
                digits = sheaf_scalar_digits(entry)
                if digits > MAX_SHEAF_ENTRY_DIGITS:
                    raise OperationResourceAdmissionError(
                        location=("sheaf", "cover_restrictions"),
                        code="topology.cellular_sheaf.hodge.scalar_digits",
                        message="restriction scalar exceeds the admitted exact-height bound",
                    )
                max_scalar_digits = max(max_scalar_digits, digits)

    max_contractions = max(
        (
            (dimensions[k - 1] if k > 0 else 0)
            + (dimensions[k + 1] if k + 1 < len(dimensions) else 0)
            for k in range(len(dimensions))
        ),
        default=0,
    )
    laplacian_entry_digits = (
        2 * max_scalar_digits * max_contractions + len(str(max_contractions + 1)) + 4
    )
    output_chars = source_chars + sum(
        2 * len(stalk.basis) ** 2 for stalk in sheaf.stalks
    )
    for size in dimensions:
        # Exact elimination entries are quotients of minors. Bound both
        # rational components using Hadamard on the admitted Hodge entries.
        result_digits = max(1, size * (laplacian_entry_digits + 2 * size + 4))
        output_chars += sheaf_scalar_json_bound(size * size, result_digits)
    if output_chars > MAX_SHEAF_HODGE_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=("sheaf",),
            code="topology.cellular_sheaf.hodge.output_chars",
            message="the conservative exact Hodge output bound exceeds its character envelope",
        )


def _laplacian_parts(
    degree: int,
    size: int,
    differentials: list[list[list[Fraction]]],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    up = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    down = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    if degree > 0:
        for column in zip(*differentials[degree - 1], strict=False):
            for i, left in enumerate(column):
                for j, right in enumerate(column):
                    up[i][j] += left * right
    if degree < len(differentials):
        for row in differentials[degree]:
            for i, left in enumerate(row):
                for j, right in enumerate(row):
                    down[i][j] += left * right
    return up, down


def hodge_laplacians(sheaf: FiniteCellularSheaf) -> SheafHodgeResult:
    """Return exact up/down Hodge Laplacians for identity Gram forms.

    Standard coordinate inner products are explicitly recorded as identity
    Gram matrices on every stalk. Their orthogonal direct sums give the
    cochain metrics used in ``delta* delta + delta delta*``.
    """
    field = _admit_field(sheaf.coefficient_field, sheaf.prime)
    if field.prime is not None:
        raise OperationDomainValidationError(
            location=("sheaf", "coefficient_field"),
            code="topology.cellular_sheaf.hodge.characteristic_zero_required",
            message="exact Hodge Laplacians require the rational coefficient field",
        )

    dimensions = tuple(
        sum(len(stalk.basis) for stalk in sheaf.stalks if len(stalk.simplex) == d + 1)
        for d in range(sheaf.complex.dimension + 1)
    )
    _admit_hodge_work(sheaf, dimensions)

    cohomology = sheaf_cohomology(sheaf)
    differentials = [
        [[Fraction(entry.num, entry.den) for entry in row] for row in matrix]
        for matrix in cohomology.coboundary_matrices
    ]
    up_laplacians: list[tuple[tuple[SheafScalar, ...], ...]] = []
    down_laplacians: list[tuple[tuple[SheafScalar, ...], ...]] = []
    laplacians: list[tuple[tuple[SheafScalar, ...], ...]] = []
    harmonic_bases: list[tuple[tuple[SheafScalar, ...], ...]] = []
    for degree, size in enumerate(dimensions):
        up, down = _laplacian_parts(degree, size, differentials)
        matrix = [[up[i][j] + down[i][j] for j in range(size)] for i in range(size)]
        # Validate the mathematical consequence with exact arithmetic: the
        # kernel of the Hodge Laplacian has the sheaf-cohomology dimension.
        harmonic = _cochain_nullspace(field, matrix, size)
        expected = cohomology.groups[degree].betti_number
        if len(harmonic) != expected:
            raise OperationResourceAdmissionError(
                location=("sheaf",),
                code="topology.cellular_sheaf.hodge_harmonic_dimension_mismatch",
                message="exact harmonic dimension disagrees with sheaf cohomology",
            )
        up_laplacians.append(
            tuple(tuple(field.typed(value) for value in row) for row in up)
        )
        down_laplacians.append(
            tuple(tuple(field.typed(value) for value in row) for row in down)
        )
        laplacians.append(
            tuple(tuple(field.typed(value) for value in row) for row in matrix)
        )
        harmonic_bases.append(
            tuple(tuple(field.typed(value) for value in vector) for vector in harmonic)
        )

    return SheafHodgeResult._from_kernel(
        sheaf=sheaf,
        cochain_bases=cohomology.cochain_bases,
        stalk_gram_matrices=tuple(
            tuple(
                tuple(
                    field.typed(field.one() if i == j else field.zero())
                    for j in range(len(stalk.basis))
                )
                for i in range(len(stalk.basis))
            )
            for stalk in sheaf.stalks
        ),
        up_laplacians=tuple(up_laplacians),
        down_laplacians=tuple(down_laplacians),
        laplacians=tuple(laplacians),
        harmonic_bases=tuple(harmonic_bases),
    )


def compute_hodge(request: SheafHodgeRequest) -> SheafHodgeResult:
    return hodge_laplacians(request.sheaf)
