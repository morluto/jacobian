"""Exact cubical complex operations."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.operations import (
    construct_chain_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_OPERATION_MATRIX_CELLS,
    CoefficientRing,
    require_prime_field_admission,
)
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CELLS,
    MAX_CUBICAL_CHAIN_GROUP,
    MAX_FACE_CELLS,
    CubicalCell,
    CubicalCellBasis,
    CubicalChainCoefficient,
    CubicalChainComplexResult,
    CubicalComplex,
    CubicalSquareLedgerEntry,
    FaceClosureResult,
    FVector,
    FVectorResult,
)

_CHAIN_RING = {
    CubicalChainCoefficient.INTEGER: CoefficientRing.INTEGER,
    CubicalChainCoefficient.PRIME_FIELD: CoefficientRing.PRIME_FIELD,
}


def _face_cells(cells: tuple[CubicalCell, ...]) -> tuple[CubicalCell, ...]:
    """Materialize the canonical face closure once during operation admission."""
    all_cells: set[tuple[tuple[int, int], ...]] = set()

    def add_faces(intervals: tuple[tuple[int, int], ...]) -> None:
        if intervals in all_cells:
            return
        if len(all_cells) >= MAX_FACE_CELLS:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.face_output_budget",
                message="cubical face closure exceeds the cell output bound",
            )
        all_cells.add(intervals)
        for i, (a, b) in enumerate(intervals):
            if b > a:
                for endpoint in (a, b):
                    face = list(intervals)
                    face[i] = (endpoint, endpoint)
                    add_faces(tuple(face))

    for cell in cells:
        add_faces(cell.intervals)
    return tuple(CubicalCell(intervals=intervals) for intervals in sorted(all_cells))


def _canonical_complex(
    cells: tuple[CubicalCell, ...],
) -> tuple[CubicalComplex, tuple[CubicalCell, ...]]:
    if not cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message="at least one cell is required",
        )
    ambient_dimension = len(cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message="all cells must use one ambient coordinate axis",
        )
    if len(cells) > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.source_cell_budget",
            message="cubical source exceeds the cell input bound",
        )
    source_cells = tuple(sorted(set(cells), key=lambda cell: cell.intervals))
    closed_cells = _face_cells(source_cells)
    return (
        CubicalComplex(
            ambient_dimension=ambient_dimension,
            cells=closed_cells,
        ),
        source_cells,
    )


def _counts(complex_: CubicalComplex) -> FVector:
    by_dimension = [0] * (complex_.ambient_dimension + 1)
    for cell in complex_.cells:
        by_dimension[cell.dimension] += 1
    return FVector(
        dimension_axis=tuple(range(complex_.ambient_dimension + 1)),
        counts=tuple(by_dimension),
    )


def f_vector(cells: tuple[CubicalCell, ...]) -> FVectorResult:
    """Compute the f-vector and Euler characteristic of a cubical complex.

    The f-vector counts all faces (including the supplied maximal cells) by
    dimension.  A single square [0,1]x[0,1] has 4 vertices, 4 edges, 1 square,
    so its f-vector is (4, 4, 1).
    """
    complex_, source_cells = _canonical_complex(cells)
    vector = _counts(complex_)
    euler = sum((-1) ** d * count for d, count in enumerate(vector.counts))
    return FVectorResult(
        complex=complex_,
        source_cells=source_cells,
        f_vector=vector,
        euler_characteristic=euler,
    )


def face_closure(cells: tuple[CubicalCell, ...]) -> FaceClosureResult:
    """Compute the full face closure of a set of cells."""
    complex_, source_cells = _canonical_complex(cells)
    cells_by_dimension = _counts(complex_)

    return FaceClosureResult(
        complex=complex_,
        source_cells=source_cells,
        original_cells=len(source_cells),
        total_cells=len(complex_.cells),
        cells_by_dimension=cells_by_dimension,
    )


def verify_f_vector(claim: FVectorResult) -> bool:
    """Verify f-vector and Euler claims against retained source cells."""
    try:
        return f_vector(claim.source_cells) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_face_closure(claim: FaceClosureResult) -> bool:
    """Verify the canonical face closure and count summary."""
    try:
        return face_closure(claim.source_cells) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def _cells_by_dimension(
    complex_: CubicalComplex,
) -> tuple[tuple[CubicalCell, ...], ...]:
    """Partition the canonical face-closed cells by dimension.

    Face closure guarantees at least one cell in every dimension from zero
    through the top cell dimension, so the resulting groups form a
    contiguous degree axis.
    """
    top = max(cell.dimension for cell in complex_.cells)
    groups: list[list[CubicalCell]] = [[] for _ in range(top + 1)]
    for cell in complex_.cells:
        groups[cell.dimension].append(cell)
    return tuple(tuple(group) for group in groups)


def _boundary_terms(cell: CubicalCell) -> tuple[tuple[CubicalCell, int], ...]:
    """Return the signed codimension-one face terms of one elementary cube.

    With nondegenerate axes indexed in increasing ambient order, the oriented
    cellular boundary is ``sum_j (-1)^(j-1) (Q_j^+ - Q_j^-)`` where the first
    nondegenerate axis uses the ``+`` sign.  Each returned coefficient is
    relative to the face's own canonical orientation.
    """
    intervals = cell.intervals
    nondegenerate = [axis for axis, (a, b) in enumerate(intervals) if b > a]
    terms: list[tuple[CubicalCell, int]] = []
    for position, axis in enumerate(nondegenerate):
        lower, upper = intervals[axis]
        upper_sign = 1 if position % 2 == 0 else -1
        upper_intervals = list(intervals)
        upper_intervals[axis] = (upper, upper)
        lower_intervals = list(intervals)
        lower_intervals[axis] = (lower, lower)
        terms.append((CubicalCell(intervals=tuple(upper_intervals)), upper_sign))
        terms.append((CubicalCell(intervals=tuple(lower_intervals)), -upper_sign))
    return tuple(terms)


def _boundary_matrices(
    groups: tuple[tuple[CubicalCell, ...], ...],
    *,
    prime: int | None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Assemble the dense cubical boundary matrices in canonical basis order."""
    matrices: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(1, len(groups)):
        source = groups[degree]
        target = groups[degree - 1]
        row_for = {cell: index for index, cell in enumerate(target)}
        dense = [[0] * len(source) for _ in target]
        for column, cell in enumerate(source):
            for face, coefficient in _boundary_terms(cell):
                dense[row_for[face]][column] += coefficient
        matrices.append(
            tuple(
                tuple(value % prime if prime is not None else value for value in row)
                for row in dense
            )
        )
    return tuple(matrices)


def chain_complex(
    cells: tuple[CubicalCell, ...],
    coefficient_ring: CubicalChainCoefficient = CubicalChainCoefficient.INTEGER,
    prime: int | None = None,
) -> CubicalChainComplexResult:
    """Build the exact based cubical chain complex of a face-closed complex.

    The elementary cubes are closed under every cubical face once during
    admission, partitioned into canonical per-dimension bases, and assembled
    into the oriented cubical boundary ``d Q = sum_j (-1)^(j-1) (Q_j^+ -
    Q_j^-)`` over ``ZZ`` or ``GF(p)``.  The differentials are delegated to the
    shared exact based chain-complex kernel, which replays ``d^2 = 0`` before
    the result is returned.
    """
    complex_, _source_cells = _canonical_complex(cells)
    groups = _cells_by_dimension(complex_)
    if any(len(group) > MAX_CUBICAL_CHAIN_GROUP for group in groups):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.chain_group_budget",
            message=(
                "a cubical chain group exceeds the "
                f"{MAX_CUBICAL_CHAIN_GROUP}-cell per-degree bound"
            ),
        )
    aggregate_cells = sum(
        len(groups[degree]) * len(groups[degree + 1])
        for degree in range(len(groups) - 1)
    )
    if aggregate_cells > MAX_OPERATION_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.chain_cell_budget",
            message=(
                "the cubical boundary matrices exceed the "
                f"{MAX_OPERATION_MATRIX_CELLS}-cell aggregate bound"
            ),
        )
    ring = _CHAIN_RING[coefficient_ring]
    modulus = prime if ring is CoefficientRing.PRIME_FIELD else None
    try:
        require_prime_field_admission(ring, prime)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("coefficient_ring", "prime"),
            code="cubical_complex.chain_coefficient_invalid",
            message=str(exc),
        ) from exc
    basis_sizes = tuple(len(group) for group in groups)
    matrices = _boundary_matrices(groups, prime=modulus)
    value = construct_chain_complex(
        basis_sizes,
        matrices,
        coefficient_ring=ring,
        prime=modulus,
    )
    ledger = tuple(
        CubicalSquareLedgerEntry(
            upper_dimension=degree,
            product_rows=len(groups[degree - 2]) if degree >= 2 else 0,
            product_columns=len(groups[degree]),
        )
        for degree in range(1, len(groups))
    )
    return CubicalChainComplexResult._from_kernel(
        complex=complex_,
        coefficient_ring=coefficient_ring,
        prime=prime,
        cell_bases=tuple(
            CubicalCellBasis(dimension=degree, cells=groups[degree])
            for degree in range(len(groups))
        ),
        value=value,
        differential_squared_zero=ledger,
    )


__all__ = [
    "chain_complex",
    "f_vector",
    "face_closure",
    "verify_f_vector",
    "verify_face_closure",
]
