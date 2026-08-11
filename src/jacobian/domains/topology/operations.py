"""Exact finite simplicial-complex and prime-field homology operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.polys.matrices import DomainMatrix

from jacobian.contracts.certified_snf import CertifiedIntegerMatrix
from jacobian.contracts.topology import (
    BoundarySquareLedgerEntry,
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    FacesInDimension,
    FiniteSimplicialComplex,
    HomologyConvention,
    HomologyGroupResult,
    IntegralFreeGenerator,
    IntegralHomologyGroupResult,
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    IntegralTorsionGenerator,
    IntegralVector,
    ModularVector,
    SimplexBasis,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
    SparseBoundaryMatrix,
    SparseMatrixEntry,
    face_closure,
    simplicial_complex_digest,
)
from jacobian.domains._certified_snf import (
    certificate_from_reduction,
    inverse_unimodular,
    matrix_columns,
    matrix_multiply,
    matrix_vector_multiply,
    smith_reduce,
)
from jacobian.domains._examples import example
from jacobian.operations import ComputedOperation, ComputedSuccess


def _canonical_complex(
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


def _canonicalize(
    request: SimplicialComplexRequest,
) -> ComputedSuccess[SimplicialComplexCanonicalizationResult]:
    return ComputedSuccess(
        SimplicialComplexCanonicalizationResult(
            complex=_canonical_complex(request.vertices, request.facets)
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
                if prime is None:
                    raise ValueError(
                        "prime field coefficient ring requires a prime modulus"
                    )
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
        if lower is None:
            raise ValueError("boundary for lower dimension is unexpectedly None")
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


def _domain_matrix(
    matrix: Sequence[Sequence[int]],
    *,
    rows: int,
    columns: int,
    prime: int,
) -> DomainMatrix:
    """Build a SymPy ``DomainMatrix`` over ``GF(prime)`` with residues in ``[0, p)``."""

    import sympy
    from sympy.polys.matrices import DomainMatrix

    field = sympy.GF(prime)
    entries = [[int(value) % prime for value in row[:columns]] for row in matrix[:rows]]
    return DomainMatrix(entries, (rows, columns), field)


def _rref(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> tuple[list[list[int]], tuple[int, ...]]:
    row_count = len(matrix)
    # Explicitly handle empty shapes: DomainMatrix requires consistent row
    # lists, so a 0xm matrix is an empty row list and an nx0 matrix is n
    # empty row lists.
    if row_count == 0 or columns == 0:
        return [[0] * columns for _ in range(row_count)], ()
    domain = _domain_matrix(matrix, rows=row_count, columns=columns, prime=prime)
    reduced_domain, pivot_columns = domain.rref()
    reduced_matrix = reduced_domain.to_Matrix()
    rows_out = [
        [int(reduced_matrix[row, column]) % prime for column in range(columns)]
        for row in range(row_count)
    ]
    return rows_out, tuple(int(pivot) for pivot in pivot_columns)


def _rank(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> int:
    row_count = len(matrix)
    if row_count == 0 or columns == 0:
        return 0
    domain = _domain_matrix(matrix, rows=row_count, columns=columns, prime=prime)
    return int(domain.rank())


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


class _IncrementalVectorBasis:
    def __init__(self, *, dimension: int, prime: int) -> None:
        self._prime = prime
        self._rows: dict[int, list[int]] = {}
        self._dimension = dimension

    def add(self, vector: Sequence[int]) -> bool:
        reduced = [value % self._prime for value in vector]
        if len(reduced) != self._dimension:
            raise ValueError("basis vector has the wrong dimension")
        for existing_pivot, row in self._rows.items():
            factor = reduced[existing_pivot]
            if factor:
                reduced = [
                    (value - factor * basis_value) % self._prime
                    for value, basis_value in zip(reduced, row, strict=True)
                ]
        new_pivot = next(
            (index for index, value in enumerate(reduced) if value),
            None,
        )
        if new_pivot is None:
            return False
        inverse = pow(reduced[new_pivot], -1, self._prime)
        reduced = [value * inverse % self._prime for value in reduced]
        for existing_pivot, row in tuple(self._rows.items()):
            factor = row[new_pivot]
            if factor:
                self._rows[existing_pivot] = [
                    (value - factor * basis_value) % self._prime
                    for value, basis_value in zip(row, reduced, strict=True)
                ]
        self._rows[new_pivot] = reduced
        self._rows = dict(sorted(self._rows.items()))
        return True


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
    basis = _IncrementalVectorBasis(dimension=row_count, prime=prime)
    for column in range(columns):
        vector = tuple(matrix[row][column] % prime for row in range(row_count))
        if basis.add(vector):
            selected.append(vector)
    return tuple(selected)


def _quotient_basis(
    cycles: Sequence[Sequence[int]],
    boundaries: Sequence[Sequence[int]],
    *,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    dimension = len(cycles[0]) if cycles else (len(boundaries[0]) if boundaries else 0)
    basis = _IncrementalVectorBasis(dimension=dimension, prime=prime)
    for boundary in boundaries:
        basis.add(boundary)
    quotient: list[tuple[int, ...]] = []
    for cycle in cycles:
        vector = tuple(cycle)
        if basis.add(vector):
            quotient.append(vector)
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
        if outgoing is None:
            raise ValueError("boundary for dimension is unexpectedly None")
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


def _integer_matrix(
    entries: list[list[int]],
    *,
    rows: int,
    columns: int,
) -> CertifiedIntegerMatrix:
    return CertifiedIntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(tuple(str(value) for value in row) for row in entries),
    )


def _integral_vector(values: list[int]) -> IntegralVector:
    return IntegralVector(coefficients=tuple(str(value) for value in values))


def _integral_homology(
    request: IntegralSimplicialHomologyRequest,
) -> ComputedSuccess[IntegralSimplicialHomologyResult]:
    chain = _chain_result(
        ChainComplexRequest(
            complex=request.complex,
            coefficient_ring=ChainCoefficientRing.INTEGER,
            convention=request.convention,
        )
    )
    boundaries = tuple(
        _dense(matrix, modulus=None) for matrix in chain.boundary_matrices
    )
    groups: list[IntegralHomologyGroupResult] = []
    for dimension, basis in enumerate(chain.simplex_bases):
        chain_dimension = len(basis.simplices)
        if dimension == 0 and chain.augmentation is not None:
            outgoing = _dense(chain.augmentation, modulus=None)
            outgoing_rows = 1
        else:
            outgoing = boundaries[dimension]
            outgoing_rows = chain.boundary_matrices[dimension].rows
        outgoing_reduction = smith_reduce(
            outgoing,
            row_count=outgoing_rows,
            column_count=chain_dimension,
        )
        outgoing_rank = outgoing_reduction.rank
        cycle_rank = chain_dimension - outgoing_rank
        cycle_basis = matrix_columns(
            outgoing_reduction.right,
            start=outgoing_rank,
        )
        right_inverse = inverse_unimodular(outgoing_reduction.right)

        if dimension < request.complex.dimension:
            incoming = boundaries[dimension + 1]
            incoming_chain_dimension = len(chain.simplex_bases[dimension + 1].simplices)
        else:
            incoming_chain_dimension = 0
            incoming = [[] for _ in range(chain_dimension)]
        all_cycle_coordinates = matrix_multiply(
            right_inverse,
            incoming,
            right_columns_if_empty=incoming_chain_dimension,
        )
        if any(
            value != 0 for row in all_cycle_coordinates[:outgoing_rank] for value in row
        ):
            raise ArithmeticError("incoming boundary is not in the outgoing kernel")
        incoming_coordinates = all_cycle_coordinates[outgoing_rank:]
        incoming_reduction = smith_reduce(
            incoming_coordinates,
            row_count=cycle_rank,
            column_count=incoming_chain_dimension,
        )
        incoming_rank = incoming_reduction.rank
        incoming_left_inverse = inverse_unimodular(incoming_reduction.left)

        free_generators: list[IntegralFreeGenerator] = []
        for index in range(incoming_rank, cycle_rank):
            coordinate = [
                incoming_left_inverse[row][index] for row in range(cycle_rank)
            ]
            free_generators.append(
                IntegralFreeGenerator(
                    cycle=_integral_vector(
                        matrix_vector_multiply(cycle_basis, coordinate)
                    ),
                    cycle_coordinates=_integral_vector(coordinate),
                )
            )

        torsion_generators: list[IntegralTorsionGenerator] = []
        for index, factor in enumerate(incoming_reduction.invariant_factors):
            if factor == 1:
                continue
            coordinate = [
                incoming_left_inverse[row][index] for row in range(cycle_rank)
            ]
            cycle = matrix_vector_multiply(cycle_basis, coordinate)
            bounding_chain = [
                incoming_reduction.right[row][index]
                for row in range(incoming_chain_dimension)
            ]
            if matrix_vector_multiply(incoming, bounding_chain) != [
                factor * value for value in cycle
            ]:
                raise ArithmeticError("torsion bounding-chain relation is invalid")
            torsion_generators.append(
                IntegralTorsionGenerator(
                    order=str(factor),
                    cycle=_integral_vector(cycle),
                    cycle_coordinates=_integral_vector(coordinate),
                    bounding_chain=_integral_vector(bounding_chain),
                )
            )

        groups.append(
            IntegralHomologyGroupResult(
                dimension=dimension,
                chain_dimension=chain_dimension,
                incoming_chain_dimension=incoming_chain_dimension,
                outgoing_boundary_rank=outgoing_rank,
                cycle_rank=cycle_rank,
                incoming_boundary_rank=incoming_rank,
                betti_number=cycle_rank - incoming_rank,
                torsion_coefficients=tuple(
                    str(factor)
                    for factor in incoming_reduction.invariant_factors
                    if factor > 1
                ),
                free_generators=tuple(free_generators),
                torsion_generators=tuple(torsion_generators),
                outgoing_smith_certificate=certificate_from_reduction(
                    outgoing_reduction
                ),
                boundary_in_cycle_coordinates=_integer_matrix(
                    incoming_coordinates,
                    rows=cycle_rank,
                    columns=incoming_chain_dimension,
                ),
                incoming_smith_certificate=certificate_from_reduction(
                    incoming_reduction
                ),
            )
        )
    return ComputedSuccess(
        IntegralSimplicialHomologyResult(
            complex_digest=request.complex.complex_digest,
            convention=request.convention,
            dimension_range=(0, request.complex.dimension),
            groups=tuple(groups),
        )
    )


_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

type TopologyOperation = (
    ComputedOperation[SimplicialComplexRequest, SimplicialComplexCanonicalizationResult]
    | ComputedOperation[ChainComplexRequest, ChainComplexResult]
    | ComputedOperation[SimplicialHomologyRequest, SimplicialHomologyResult]
    | ComputedOperation[
        IntegralSimplicialHomologyRequest, IntegralSimplicialHomologyResult
    ]
)


TOPOLOGY_CAPABILITIES: tuple[TopologyOperation, ...] = (
    ComputedOperation(
        capability_id="topology.simplicial_complex.canonicalize",
        title="Canonicalize a finite simplicial complex",
        description=(
            "Validate bounded maximal facets, close them under every non-empty "
            "face, and return canonical oriented simplex bases and the exact "
            "f-vector."
        ),
        request_model=SimplicialComplexRequest,
        result_model=SimplicialComplexCanonicalizationResult,
        implementation=_canonicalize,
        relation_id="topology.simplicial_complex.canonicalization.relation",
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
                "Canonicalize the three-edge simplicial model of a circle.",
                _CIRCLE,
            ),
        ),
        version="4",
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
        version="4",
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
        version="4",
    ),
    ComputedOperation(
        capability_id="topology.simplicial_homology.integral.compute",
        title="Compute transformation-certified integral simplicial homology",
        description=(
            "Compute the free rank, torsion invariant factors, and simplex-basis "
            "cycle generators of every integral homology group, with explicit "
            "Smith transformations and bounding chains."
        ),
        request_model=IntegralSimplicialHomologyRequest,
        result_model=IntegralSimplicialHomologyResult,
        implementation=_integral_homology,
        relation_id="topology.simplicial_homology.integral.relation",
        tags=(
            "topology",
            "simplicial-homology",
            "integer-homology",
            "torsion",
            "betti-number",
            "cycle-generator",
            "smith-normal-form",
            "certificate",
            "exact",
        ),
        invocation_examples=(
            example(
                "integral_circle_homology",
                "Compute H_0 and H_1 over the integers for a triangle boundary.",
                {
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "maximal_simplices": [
                            ["a", "b"],
                            ["a", "c"],
                            ["b", "c"],
                        ],
                        "faces_by_dimension": [
                            {
                                "dimension": 0,
                                "faces": [["a"], ["b"], ["c"]],
                            },
                            {
                                "dimension": 1,
                                "faces": [
                                    ["a", "b"],
                                    ["a", "c"],
                                    ["b", "c"],
                                ],
                            },
                        ],
                        "dimension": 1,
                        "f_vector": [3, 3],
                        "closure_size": 6,
                        "complex_digest": (
                            "sha256:"
                            "6f797991bac967e2a8e572707df487061655df0f094c"
                            "bde0f52f82c5401fc043"
                        ),
                    }
                },
            ),
        ),
        version="4",
    ),
)

__all__ = ["TOPOLOGY_CAPABILITIES"]
