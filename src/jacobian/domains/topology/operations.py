"""Exact finite simplicial-complex and prime-field homology operations."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.contracts.topology import (
    BoundarySquareLedgerEntry,
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    FacesInDimension,
    FiniteSimplicialComplex,
    HomologyConvention,
    HomologyGroupResult,
    ModularVector,
    SimplexBasis,
    SimplicialComplexMaterializationResult,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
    SparseBoundaryMatrix,
    SparseMatrixEntry,
    face_closure,
    simplicial_complex_digest,
)
from jacobian.domains._examples import example
from jacobian.operations import ComputedOperation, ComputedSuccess


def _materialized_complex(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
) -> FiniteSimplicialComplex:
    canonical_vertices = tuple(sorted(vertices))
    canonical_facets = tuple(sorted(tuple(sorted(facet)) for facet in facets))
    closure = face_closure(canonical_facets)
    faces_by_dimension = tuple(
        FacesInDimension(dimension=dimension, faces=faces)
        for dimension, faces in enumerate(closure)
    )
    f_vector = tuple(len(faces) for faces in closure)
    closure_size = sum(f_vector)
    dimension = len(closure) - 1
    digest = simplicial_complex_digest(
        vertices=canonical_vertices,
        maximal_simplices=canonical_facets,
        faces_by_dimension=faces_by_dimension,
        dimension=dimension,
        f_vector=f_vector,
        closure_size=closure_size,
    )
    return FiniteSimplicialComplex(
        vertices=canonical_vertices,
        maximal_simplices=canonical_facets,
        faces_by_dimension=faces_by_dimension,
        dimension=dimension,
        f_vector=f_vector,
        closure_size=closure_size,
        complex_digest=digest,
    )


def _materialize(
    request: SimplicialComplexRequest,
) -> ComputedSuccess[SimplicialComplexMaterializationResult]:
    return ComputedSuccess(
        SimplicialComplexMaterializationResult(
            complex=_materialized_complex(request.vertices, request.facets)
        )
    )


def _boundary_matrix(
    complex_: FiniteSimplicialComplex,
    dimension: int,
    *,
    coefficient_ring: ChainCoefficientRing,
    prime: int | None,
) -> SparseBoundaryMatrix:
    source = complex_.faces_by_dimension[dimension].faces
    if dimension == 0:
        return SparseBoundaryMatrix(
            source_dimension=0,
            target_dimension=-1,
            rows=0,
            columns=len(source),
        )
    target = complex_.faces_by_dimension[dimension - 1].faces
    row_for_face = {face: index for index, face in enumerate(target)}
    entries: list[SparseMatrixEntry] = []
    for column, simplex in enumerate(source):
        for removed in range(len(simplex)):
            face = simplex[:removed] + simplex[removed + 1 :]
            value = 1 if removed % 2 == 0 else -1
            if coefficient_ring is ChainCoefficientRing.PRIME_FIELD:
                assert prime is not None
                value %= prime
            entries.append(
                SparseMatrixEntry(
                    row=row_for_face[face],
                    column=column,
                    value=value,
                )
            )
    return SparseBoundaryMatrix(
        source_dimension=dimension,
        target_dimension=dimension - 1,
        rows=len(target),
        columns=len(source),
        entries=tuple(sorted(entries, key=lambda item: (item.row, item.column))),
    )


def _augmentation(
    vertex_count: int,
) -> SparseBoundaryMatrix:
    return SparseBoundaryMatrix(
        source_dimension=0,
        target_dimension=-1,
        rows=1,
        columns=vertex_count,
        entries=tuple(
            SparseMatrixEntry(row=0, column=column, value=1)
            for column in range(vertex_count)
        ),
    )


def _dense(
    matrix: SparseBoundaryMatrix,
    *,
    modulus: int | None,
) -> list[list[int]]:
    dense = [[0] * matrix.columns for _ in range(matrix.rows)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] = (
            entry.value if modulus is None else entry.value % modulus
        )
    return dense


def _product_is_zero(
    left: SparseBoundaryMatrix,
    right: SparseBoundaryMatrix,
    *,
    modulus: int | None,
) -> bool:
    if left.columns != right.rows:
        raise ValueError("boundary matrices are not composable")
    left_dense = _dense(left, modulus=modulus)
    right_dense = _dense(right, modulus=modulus)
    for row in range(left.rows):
        for column in range(right.columns):
            total = sum(
                left_dense[row][middle] * right_dense[middle][column]
                for middle in range(left.columns)
            )
            if modulus is not None:
                total %= modulus
            if total != 0:
                return False
    return True


def _chain_result(request: ChainComplexRequest) -> ChainComplexResult:
    complex_ = request.complex
    bases = tuple(
        SimplexBasis(dimension=item.dimension, simplices=item.faces)
        for item in complex_.faces_by_dimension
    )
    boundaries = tuple(
        _boundary_matrix(
            complex_,
            dimension,
            coefficient_ring=request.coefficient_ring,
            prime=request.prime,
        )
        for dimension in range(complex_.dimension + 1)
    )
    augmentation = (
        _augmentation(len(complex_.vertices))
        if request.convention is HomologyConvention.REDUCED
        else None
    )
    modulus = (
        request.prime
        if request.coefficient_ring is ChainCoefficientRing.PRIME_FIELD
        else None
    )
    ledger: list[BoundarySquareLedgerEntry] = []
    for upper_dimension in range(1, complex_.dimension + 1):
        lower = (
            augmentation
            if upper_dimension == 1 and augmentation is not None
            else boundaries[upper_dimension - 1]
        )
        assert lower is not None
        upper = boundaries[upper_dimension]
        if not _product_is_zero(lower, upper, modulus=modulus):
            raise ValueError("constructed simplicial boundary does not square to zero")
        ledger.append(
            BoundarySquareLedgerEntry(
                upper_dimension=upper_dimension,
                product_rows=lower.rows,
                product_columns=upper.columns,
            )
        )
    return ChainComplexResult(
        complex_digest=complex_.complex_digest,
        coefficient_ring=request.coefficient_ring,
        prime=request.prime,
        convention=request.convention,
        simplex_bases=bases,
        boundary_matrices=boundaries,
        augmentation=augmentation,
        boundary_squared_zero=tuple(ledger),
    )


def _chain(
    request: ChainComplexRequest,
) -> ComputedSuccess[ChainComplexResult]:
    return ComputedSuccess(_chain_result(request))


def _rref(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> tuple[list[list[int]], tuple[int, ...]]:
    rows = [[value % prime for value in row] for row in matrix]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] % prime),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        inverse = pow(rows[pivot_row][column], -1, prime)
        rows[pivot_row] = [value * inverse % prime for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            factor = rows[row][column] % prime
            if factor:
                rows[row] = [
                    (value - factor * pivot) % prime
                    for value, pivot in zip(
                        rows[row],
                        rows[pivot_row],
                        strict=True,
                    )
                ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows, tuple(pivots)


def _rank(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> int:
    return len(_rref(matrix, columns=columns, prime=prime)[1])


def _nullspace(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    reduced, pivots = _rref(matrix, columns=columns, prime=prime)
    free_columns = tuple(column for column in range(columns) if column not in pivots)
    basis: list[tuple[int, ...]] = []
    for free in free_columns:
        vector = [0] * columns
        vector[free] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free] % prime
        basis.append(tuple(vector))
    return tuple(basis)


def _vector_rank(vectors: Sequence[Sequence[int]], *, prime: int) -> int:
    if not vectors:
        return 0
    rows = [[vector[row] for vector in vectors] for row in range(len(vectors[0]))]
    return _rank(rows, columns=len(vectors), prime=prime)


def _column_basis(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    if columns == 0:
        return ()
    row_count = len(matrix)
    selected: list[tuple[int, ...]] = []
    rank = 0
    for column in range(columns):
        vector = tuple(matrix[row][column] % prime for row in range(row_count))
        candidate_rank = _vector_rank((*selected, vector), prime=prime)
        if candidate_rank > rank:
            selected.append(vector)
            rank = candidate_rank
    return tuple(selected)


def _quotient_basis(
    cycles: Sequence[Sequence[int]],
    boundaries: Sequence[Sequence[int]],
    *,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    selected = [tuple(vector) for vector in boundaries]
    rank = _vector_rank(selected, prime=prime)
    quotient: list[tuple[int, ...]] = []
    for cycle in cycles:
        vector = tuple(cycle)
        candidate_rank = _vector_rank((*selected, vector), prime=prime)
        if candidate_rank > rank:
            selected.append(vector)
            quotient.append(vector)
            rank = candidate_rank
    return tuple(quotient)


def _homology(
    request: SimplicialHomologyRequest,
) -> ComputedSuccess[SimplicialHomologyResult]:
    chain = _chain_result(
        ChainComplexRequest(
            complex=request.complex,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=request.prime,
            convention=request.convention,
        )
    )
    boundaries = tuple(
        _dense(matrix, modulus=request.prime) for matrix in chain.boundary_matrices
    )
    augmentation = (
        None
        if chain.augmentation is None
        else _dense(chain.augmentation, modulus=request.prime)
    )
    groups: list[HomologyGroupResult] = []
    for dimension, basis in enumerate(chain.simplex_bases):
        chain_dimension = len(basis.simplices)
        outgoing = (
            augmentation
            if dimension == 0 and augmentation is not None
            else boundaries[dimension]
        )
        assert outgoing is not None
        cycles = _nullspace(
            outgoing,
            columns=chain_dimension,
            prime=request.prime,
        )
        outgoing_rank = _rank(
            outgoing,
            columns=chain_dimension,
            prime=request.prime,
        )
        if dimension < request.complex.dimension:
            incoming = boundaries[dimension + 1]
            incoming_columns = len(chain.simplex_bases[dimension + 1].simplices)
        else:
            incoming = [[] for _ in range(chain_dimension)]
            incoming_columns = 0
        boundary_basis = _column_basis(
            incoming,
            columns=incoming_columns,
            prime=request.prime,
        )
        homology_basis = _quotient_basis(
            cycles,
            boundary_basis,
            prime=request.prime,
        )
        quotient_span_rank = _vector_rank(
            (*boundary_basis, *homology_basis),
            prime=request.prime,
        )
        groups.append(
            HomologyGroupResult(
                dimension=dimension,
                chain_dimension=chain_dimension,
                outgoing_boundary_rank=outgoing_rank,
                cycle_dimension=len(cycles),
                incoming_boundary_rank=len(boundary_basis),
                betti_number=len(homology_basis),
                cycle_basis=tuple(
                    ModularVector(coefficients=vector) for vector in cycles
                ),
                boundary_basis=tuple(
                    ModularVector(coefficients=vector) for vector in boundary_basis
                ),
                homology_basis=tuple(
                    ModularVector(coefficients=vector) for vector in homology_basis
                ),
                quotient_span_rank=quotient_span_rank,
            )
        )
    return ComputedSuccess(
        SimplicialHomologyResult(
            complex_digest=request.complex.complex_digest,
            prime=request.prime,
            convention=request.convention,
            dimension_range=(0, request.complex.dimension),
            groups=tuple(groups),
        )
    )


_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

TOPOLOGY_CAPABILITIES = (
    ComputedOperation(
        capability_id="topology.simplicial_complex.materialize",
        title="Materialize a finite simplicial complex",
        description=(
            "Validate bounded maximal facets, close them under every non-empty "
            "face, and return canonical oriented simplex bases and the exact "
            "f-vector."
        ),
        request_model=SimplicialComplexRequest,
        result_model=SimplicialComplexMaterializationResult,
        implementation=_materialize,
        relation_id="topology.simplicial_complex.materialization.relation",
        tags=(
            "topology",
            "simplicial-complex",
            "facets",
            "face-closure",
            "f-vector",
            "exact",
        ),
        invocation_examples=(
            example(
                "triangle_boundary",
                "Materialize the three-edge simplicial model of a circle.",
                _CIRCLE,
            ),
        ),
    ),
    ComputedOperation(
        capability_id="topology.simplicial_complex.chain_complex.compute",
        title="Compute an oriented simplicial chain complex",
        description=(
            "Construct every oriented sparse boundary matrix for one canonical "
            "finite simplicial complex over the integers or a bounded prime field."
        ),
        request_model=ChainComplexRequest,
        result_model=ChainComplexResult,
        implementation=_chain,
        relation_id="topology.simplicial_complex.chain_complex.relation",
        tags=(
            "topology",
            "simplicial-complex",
            "chain-complex",
            "boundary-matrix",
            "exact",
        ),
    ),
    ComputedOperation(
        capability_id="topology.simplicial_homology.compute",
        title="Compute finite-field simplicial homology",
        description=(
            "Compute every Betti number and inspectable cycle, boundary, and "
            "quotient basis of a bounded finite simplicial complex over F_p."
        ),
        request_model=SimplicialHomologyRequest,
        result_model=SimplicialHomologyResult,
        implementation=_homology,
        relation_id="topology.simplicial_homology.relation",
        tags=(
            "topology",
            "simplicial-homology",
            "betti-number",
            "cycle-basis",
            "prime-field",
            "exact",
        ),
    ),
)

__all__ = ["TOPOLOGY_CAPABILITIES"]
