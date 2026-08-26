"""Exact finite simplicial-complex and prime-field homology operations."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math import prime_field_linear_algebra as prime_field
from jacobian.math.chain_complexes.values import ChainComplexValue
from jacobian.math.matrices.certified_snf.operations import (
    certificate_from_reduction,
    inverse_unimodular,
    matrix_columns,
    matrix_multiply,
    matrix_vector_multiply,
    smith_reduce,
)
from jacobian.math.matrices.certified_snf.values import CertifiedIntegerMatrix
from jacobian.math.topology._barycentric import barycentric_subdivision
from jacobian.math.topology._chain_conversion import (
    canonical_chain_complex_value_from_parts,
)
from jacobian.math.topology._homology import (
    HomologyGroupResult,
    IntegralFreeGenerator,
    IntegralHomologyGroupResult,
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    IntegralTorsionGenerator,
    IntegralVector,
    ModularVector,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
)
from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    BarycentricSubdivisionResult,
    BoundarySquareLedgerEntry,
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    FiniteSimplicialComplex,
    HomologyConvention,
    ShellingCheckRequest,
    ShellingCheckResult,
    SimplexBasis,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
    SparseBoundaryMatrix,
    SparseMatrixEntry,
    _all_faces,
    canonical_complex,
)
from jacobian.math.topology._pseudomanifold import (
    PseudomanifoldRequest,
    PseudomanifoldResult,
    pseudomanifold_decision,
)
from jacobian.math.topology._shelling import evaluate_shelling


def _canonical_complex(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
) -> FiniteSimplicialComplex:
    """Backward-compatible private spelling for the neutral value factory."""

    return canonical_complex(vertices, facets)


def _canonicalize(
    request: SimplicialComplexRequest,
) -> SimplicialComplexCanonicalizationResult:
    return SimplicialComplexCanonicalizationResult(
        complex=_canonical_complex(request.vertices, request.facets)
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
        canonical_value=canonical_chain_complex_value_from_parts(
            request.coefficient_ring,
            request.convention,
            request.prime,
            bases,
            boundaries,
        ),
    )


def _chain(
    request: ChainComplexRequest,
) -> ChainComplexResult:
    return _chain_result(request)


def _canonical_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """The canonical chain-complex value carried by one chain result."""
    value = canonical_chain_complex_value_from_parts(
        result.coefficient_ring,
        result.convention,
        result.prime,
        result.simplex_bases,
        result.boundary_matrices,
    )
    if value is None:
        raise ValueError(
            "only unreduced prime-field simplicial chain complexes convert "
            "to a canonical chain-complex value"
        )
    return value


def _prime_matrix(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> prime_field.PrimeFieldMatrix:
    return prime_field.PrimeFieldMatrix(
        prime=prime,
        entries=tuple(tuple(value for value in row[:columns]) for row in matrix),
        columns=columns,
    )


def _vector_rank(vectors: Sequence[Sequence[int]], *, prime: int) -> int:
    if not vectors:
        return 0
    rows = [[vector[row] for vector in vectors] for row in range(len(vectors[0]))]
    return prime_field.rank(_prime_matrix(rows, columns=len(vectors), prime=prime))


def _homology(
    request: SimplicialHomologyRequest,
) -> SimplicialHomologyResult:
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
        outgoing_matrix = _prime_matrix(
            outgoing, columns=chain_dimension, prime=request.prime
        )
        cycles = prime_field.nullspace(outgoing_matrix)
        outgoing_rank = prime_field.rank(outgoing_matrix)
        if dimension < request.complex.dimension:
            incoming = boundaries[dimension + 1]
            incoming_columns = len(chain.simplex_bases[dimension + 1].simplices)
        else:
            incoming = [[] for _ in range(chain_dimension)]
            incoming_columns = 0
        boundary_basis = prime_field.column_basis(
            _prime_matrix(
                incoming,
                columns=incoming_columns,
                prime=request.prime,
            )
        )
        homology_basis = prime_field.quotient_basis(
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
    return SimplicialHomologyResult(
        complex_digest=request.complex.complex_digest,
        prime=request.prime,
        convention=request.convention,
        dimension_range=(0, request.complex.dimension),
        groups=tuple(groups),
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
) -> IntegralSimplicialHomologyResult:
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
    return IntegralSimplicialHomologyResult(
        complex_digest=request.complex.complex_digest,
        convention=request.convention,
        dimension_range=(0, request.complex.dimension),
        groups=tuple(groups),
    )


_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

_CANONICAL_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
        {"dimension": 1, "faces": [["a", "b"], ["a", "c"], ["b", "c"]]},
    ],
    "dimension": 1,
    "f_vector": [3, 3],
    "closure_size": 6,
    "complex_digest": (
        "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647"
    ),
}

type TopologyOperation = (
    MathTool[SimplicialComplexRequest, SimplicialComplexCanonicalizationResult]
    | MathTool[ChainComplexRequest, ChainComplexResult]
    | MathTool[SimplicialHomologyRequest, SimplicialHomologyResult]
    | MathTool[IntegralSimplicialHomologyRequest, IntegralSimplicialHomologyResult]
)


TOPOLOGY_OPERATIONS: tuple[TopologyOperation, ...] = (
    MathTool(
        operation_id="topology.simplicial_complex.canonicalize",
        title="Canonicalize a finite simplicial complex",
        description=(
            "Validate bounded maximal facets, close them under every non-empty "
            "face, and return canonical oriented simplex bases and the exact "
            "f-vector."
        ),
        request_type=SimplicialComplexRequest,
        result_type=SimplicialComplexCanonicalizationResult,
        run=_canonicalize,
        tags=(
            "topology",
            "simplicial-complex",
            "facets",
            "face-closure",
            "f-vector",
            "exact",
        ),
        examples=(
            example(
                "triangle_boundary",
                "Canonicalize the three-edge simplicial model of a circle.",
                _CIRCLE,
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.chain_complex.compute",
        title="Compute an oriented simplicial chain complex",
        description=(
            "Construct every oriented sparse boundary matrix for one canonical "
            "finite simplicial complex over the integers or a bounded prime field."
        ),
        request_type=ChainComplexRequest,
        result_type=ChainComplexResult,
        run=_chain,
        tags=(
            "topology",
            "simplicial-complex",
            "chain-complex",
            "boundary-matrix",
            "exact",
        ),
        examples=(
            example(
                "circle_integer_chain_complex",
                "Construct the oriented integer boundary matrices of a circle.",
                {
                    "complex": _CANONICAL_CIRCLE,
                    "coefficient_ring": "INTEGER",
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_homology.compute",
        title="Compute finite-field simplicial homology",
        description=(
            "Compute every Betti number and inspectable cycle, boundary, and "
            "quotient basis of a bounded finite simplicial complex over F_p."
        ),
        request_type=SimplicialHomologyRequest,
        result_type=SimplicialHomologyResult,
        run=_homology,
        tags=(
            "topology",
            "simplicial-homology",
            "betti-number",
            "cycle-basis",
            "prime-field",
            "exact",
        ),
        examples=(
            example(
                "circle_homology_mod_two",
                "Compute H_0 and H_1 over F_2 for a triangle boundary.",
                {
                    "complex": _CANONICAL_CIRCLE,
                    "prime": 2,
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_homology.integral.compute",
        title="Compute transformation-certified integral simplicial homology",
        description=(
            "Compute the free rank, torsion invariant factors, and simplex-basis "
            "cycle generators of every integral homology group, with explicit "
            "Smith transformations and bounding chains. Each chain group is "
            "bounded by the certified Smith-certificate dimension."
        ),
        request_type=IntegralSimplicialHomologyRequest,
        result_type=IntegralSimplicialHomologyResult,
        run=_integral_homology,
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
        examples=(
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
                            "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647"
                        ),
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOPOLOGY_OPERATIONS"]


def compute_barycentric_subdivision(
    request: BarycentricSubdivisionRequest,
) -> BarycentricSubdivisionResult:
    """Compute the barycentric subdivision of a simplicial complex."""

    sorted_faces = sorted(
        _all_faces(request.complex.facets), key=lambda face: (len(face), face)
    )
    subdivision = barycentric_subdivision(sorted_faces)
    facets = tuple(sorted(tuple(sorted(facet)) for facet in subdivision.facets))
    return BarycentricSubdivisionResult(
        original_vertices=request.complex.vertices,
        original_dimension=max(len(facet) - 1 for facet in request.complex.facets),
        subdivision_vertices=subdivision.vertices,
        subdivision_facets=subdivision.facets,
        num_new_vertices=len(subdivision.vertices),
        complex=request.complex,
        subdivision_complex=(
            canonical_complex(tuple(sorted(subdivision.vertices)), facets)
            if facets
            else None
        ),
        subdivision_vertex_faces=subdivision.vertex_faces,
    )


def compute_pseudomanifold_decision(
    request: PseudomanifoldRequest,
) -> PseudomanifoldResult:
    """Decide whether a complex is a pseudomanifold."""

    decision = pseudomanifold_decision(request.complex.facets)
    return PseudomanifoldResult(
        complex=request.complex,
        is_pseudomanifold=decision.is_pseudomanifold,
        is_closed=decision.is_closed,
        dimension=decision.dimension,
        num_facets=decision.num_facets,
        obstruction=decision.obstruction,
    )


def compute_shelling_check(request: ShellingCheckRequest) -> ShellingCheckResult:
    """Check whether a submitted facet order is a valid shelling order."""

    is_shelling, failed_at, failure_reason = evaluate_shelling(
        request.complex.facets, request.facet_order
    )
    return ShellingCheckResult(
        complex=request.complex,
        facet_order=request.facet_order,
        is_shelling=is_shelling,
        failed_at=failed_at,
        failure_reason=failure_reason,
    )
