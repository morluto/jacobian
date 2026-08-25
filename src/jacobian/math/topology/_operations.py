"""Exact finite simplicial-complex and prime-field homology operations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math import prime_field_linear_algebra as prime_field
from jacobian.math.chain_complexes.values import (
    MAX_BASIS_SIZE,
    MAX_MATRIX_CELLS,
    ChainComplexValue,
    CoefficientField,
)
from jacobian.math.matrices.certified_snf.operations import (
    certificate_from_reduction,
    inverse_unimodular,
    matrix_columns,
    matrix_multiply,
    matrix_vector_multiply,
    smith_reduce,
)
from jacobian.math.matrices.certified_snf.values import CertifiedIntegerMatrix
from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    BarycentricSubdivisionResult,
    BoundarySquareLedgerEntry,
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
    FacesInDimension,
    FiniteSimplicialComplex,
    FVectorRequest,
    FVectorResult,
    HomologyConvention,
    HomologyGroupResult,
    IntegralFreeGenerator,
    IntegralHomologyGroupResult,
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    IntegralTorsionGenerator,
    IntegralVector,
    JoinRequest,
    JoinResult,
    LinkRequest,
    LinkResult,
    ModularVector,
    PseudomanifoldRequest,
    PseudomanifoldResult,
    ShellingCheckRequest,
    ShellingCheckResult,
    SimplexBasis,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
    SkeletonRequest,
    SkeletonResult,
    SparseBoundaryMatrix,
    SparseMatrixEntry,
    StarRequest,
    StarResult,
    VertexDeletionRequest,
    VertexDeletionResult,
    _all_faces,
    _cover_relations,
    _evaluate_shelling,
    _maximal_chains_from_covers,
    _minimal_face_indices,
    face_closure,
    simplicial_complex_digest,
)


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
        canonical_value=_canonical_value_from_parts(
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


def _canonical_value_from_parts(
    coefficient_ring: ChainCoefficientRing,
    convention: HomologyConvention,
    prime: int | None,
    simplex_bases: tuple[SimplexBasis, ...],
    boundary_matrices: tuple[SparseBoundaryMatrix, ...],
) -> ChainComplexValue | None:
    """The canonical chain-complex value of one simplicial chain complex.

    Only unreduced prime-field results convert: integral boundaries live
    over ZZ rather than QQ or GF(p), and reduced chains carry an
    augmentation map outside the canonical value's representation. The
    ordered lexicographic face bases remain the implicit column/row
    ordering of each dense differential; simplex labels do not survive
    because the canonical value is based but unlabeled.
    """
    if coefficient_ring is not ChainCoefficientRing.PRIME_FIELD:
        return None
    if convention is HomologyConvention.REDUCED:
        return None
    if prime is None:
        raise ValueError("prime-field chains must declare their modulus")
    basis_sizes = tuple(len(basis.simplices) for basis in simplex_bases)
    total_cells = sum(matrix.rows * matrix.columns for matrix in boundary_matrices)
    if any(size > MAX_BASIS_SIZE for size in basis_sizes):
        raise ValueError(
            f"simplicial chain group exceeds the canonical basis bound {MAX_BASIS_SIZE}"
        )
    if total_cells > MAX_MATRIX_CELLS:
        raise ValueError(
            f"simplicial boundary data exceeds the canonical cell bound "
            f"{MAX_MATRIX_CELLS}"
        )
    # Boundary matrix k maps C_k -> C_{k-1}; the canonical value stores
    # differentials[i] as C_{i+1} -> C_i, i.e. boundary_matrices[i + 1].
    differential_matrices = []
    for matrix in boundary_matrices[1:]:
        dense = [[0] * matrix.columns for _ in range(matrix.rows)]
        for entry in matrix.entries:
            dense[entry.row][entry.column] = entry.value % prime
        differential_matrices.append(
            tuple(tuple(str(value) for value in row) for row in dense)
        )
    return ChainComplexValue(
        coefficient_field=CoefficientField.PRIME_FIELD,
        prime=prime,
        degree_min=0,
        degree_max=len(basis_sizes) - 1,
        basis_sizes=basis_sizes,
        differential_matrices=tuple(differential_matrices),
    )


def _canonical_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """The canonical chain-complex value carried by one chain result."""
    value = _canonical_value_from_parts(
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


def compute_f_vector(request: FVectorRequest) -> FVectorResult:
    """Compute the f-vector and h-vector of a simplicial complex."""
    facets = request.complex.facets

    # Build all simplices from facets
    from itertools import combinations as _comb

    all_simplices: set[tuple[str, ...]] = set()
    for facet in facets:
        n = len(facet)
        for r in range(1, n + 1):
            for subset in _comb(facet, r):
                all_simplices.add(tuple(sorted(subset)))

    # Count by dimension
    max_dim = 0
    counts_by_dim: dict[int, int] = {}
    for simplex in all_simplices:
        dim = len(simplex) - 1
        counts_by_dim[dim] = counts_by_dim.get(dim, 0) + 1
        max_dim = max(max_dim, dim)

    f_vector = tuple(counts_by_dim.get(d, 0) for d in range(max_dim + 1))
    euler = sum((-1) ** d * counts_by_dim.get(d, 0) for d in range(max_dim + 1))

    # Compute h-vector from f-vector
    from math import comb as _comb_func

    n = max_dim + 1
    f_with_empty: list[int] = [1, *list(f_vector)]
    h_vector: list[int] = []
    for k in range(n + 1):
        h = 0
        for i in range(k + 1):
            h += ((-1) ** (k - i)) * _comb_func(n - i, k - i) * f_with_empty[i]
        h_vector.append(h)

    return FVectorResult(
        f_vector=f_vector,
        h_vector=tuple(h_vector),
        euler_characteristic=euler,
        dimension=max_dim,
    )


def compute_link(request: LinkRequest) -> LinkResult:
    """Compute the link of a simplex in a simplicial complex."""
    target = frozenset(request.simplex)
    link_simplices: set[frozenset[str]] = set()
    for facet in request.complex.facets:
        remainder = frozenset(facet) - target
        if target.issubset(facet) and remainder:
            link_simplices.add(remainder)

    link_facets = {
        simplex
        for simplex in link_simplices
        if not any(simplex < other for other in link_simplices)
    }
    ordered_facets = tuple(
        tuple(sorted(simplex))
        for simplex in sorted(
            link_facets, key=lambda value: (-len(value), sorted(value))
        )
    )
    return LinkResult(
        simplex=request.simplex,
        link_facets=ordered_facets,
        link_is_empty=not ordered_facets,
    )


# ---------------------------------------------------------------------------
# Structural simplicial operations (#1850)
# ---------------------------------------------------------------------------


def compute_star(request: StarRequest) -> StarResult:
    """Compute the closed star of a simplex."""
    target = frozenset(request.simplex)
    star_facets = {
        frozenset(facet) for facet in request.complex.facets if target.issubset(facet)
    }
    ordered_facets = tuple(
        tuple(sorted(simplex))
        for simplex in sorted(
            star_facets, key=lambda value: (-len(value), sorted(value))
        )
    )
    is_empty = not ordered_facets
    star_complex = None
    if not is_empty:
        star_vertices = tuple(sorted({v for facet in ordered_facets for v in facet}))
        star_complex = _canonical_complex(star_vertices, ordered_facets)
    return StarResult(
        complex=request.complex,
        simplex=request.simplex,
        star_facets=ordered_facets,
        star_is_empty=is_empty,
        star_complex=star_complex,
    )


def compute_vertex_deletion(request: VertexDeletionRequest) -> VertexDeletionResult:
    """Compute the induced subcomplex after deleting a vertex subset."""
    to_delete = set(request.vertices_to_delete)
    all_faces_set = _all_faces(request.complex.facets)
    remaining_faces = {face for face in all_faces_set if not (set(face) & to_delete)}
    sorted_remaining = sorted(remaining_faces, key=lambda f: (-len(f), f))
    maximal_list: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in sorted_remaining:
        face_set = frozenset(face)
        if not any(existing.issuperset(face_set) for existing in seen):
            maximal_list.append(face)
            seen.add(face_set)
    remaining_facets = tuple(maximal_list)
    remaining_vertices = tuple(sorted({v for facet in remaining_facets for v in facet}))
    return VertexDeletionResult(
        complex=request.complex,
        deleted_vertices=tuple(sorted(to_delete)),
        remaining_vertices=remaining_vertices,
        remaining_facets=remaining_facets,
        remaining_complex=_canonical_complex(remaining_vertices, remaining_facets),
    )


def compute_skeleton(request: SkeletonRequest) -> SkeletonResult:
    """Compute the k-skeleton of a simplicial complex."""
    k = request.k
    all_faces_set = _all_faces(request.complex.facets)
    skeleton_faces = {face for face in all_faces_set if len(face) <= k + 1}
    skeleton_list = sorted(skeleton_faces, key=lambda f: (-len(f), f))
    maximal: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in skeleton_list:
        face_set = frozenset(face)
        if not any(existing.issuperset(face_set) for existing in seen):
            maximal.append(face)
            seen.add(face_set)
    skeleton_facets = tuple(maximal)
    skeleton_vertices = tuple(sorted({v for face in skeleton_facets for v in face}))
    skeleton_complex = None
    if skeleton_facets:
        skeleton_complex = _canonical_complex(skeleton_vertices, skeleton_facets)
    return SkeletonResult(
        complex=request.complex,
        k=k,
        skeleton_facets=skeleton_facets,
        skeleton_vertices=skeleton_vertices,
        skeleton_complex=skeleton_complex,
    )


def compute_join(request: JoinRequest) -> JoinResult:
    """Compute the join of two simplicial complexes."""
    all_vertices = tuple(
        sorted(set(request.complex_a.vertices) | set(request.complex_b.vertices))
    )
    join_facets: list[tuple[str, ...]] = []
    for fa in request.complex_a.facets:
        for fb in request.complex_b.facets:
            joined = tuple(sorted(set(fa) | set(fb)))
            join_facets.append(joined)
    sorted_join = sorted(join_facets, key=lambda f: (-len(f), f))
    maximal: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in sorted_join:
        face_set = frozenset(face)
        if not any(existing.issuperset(face_set) for existing in seen):
            maximal.append(face)
            seen.add(face_set)
    join_facets_tuple = tuple(maximal)
    join_dim = max(len(f) - 1 for f in join_facets_tuple) if join_facets_tuple else 0
    join_complex = None
    if join_facets_tuple:
        # join_vertices already sorted unique
        join_complex = _canonical_complex(all_vertices, join_facets_tuple)
    return JoinResult(
        complex_a=request.complex_a,
        complex_b=request.complex_b,
        join_vertices=all_vertices,
        join_facets=join_facets_tuple,
        join_dimension=join_dim,
        join_complex=join_complex,
    )


def compute_barycentric_subdivision(
    request: BarycentricSubdivisionRequest,
) -> BarycentricSubdivisionResult:
    """Compute the barycentric subdivision of a simplicial complex."""
    all_faces_set = _all_faces(request.complex.facets)
    sorted_faces = sorted(all_faces_set, key=lambda f: (len(f), f))
    # Bounded injective encoding: use short valid labels "bv{i}" instead of
    # comma-joined face content which violates VertexLabel.
    new_vertices = [f"bv{i}" for i in range(len(sorted_faces))]
    vertex_map = {face: new_vertices[idx] for idx, face in enumerate(sorted_faces)}
    # Enumerate maximal chains via cover relation (efficient, not power set).
    face_frozens = [frozenset(f) for f in sorted_faces]
    n = len(sorted_faces)
    covers = _cover_relations(face_frozens)
    minimal_indices = _minimal_face_indices(face_frozens)
    maximal_chains = _maximal_chains_from_covers(
        covers, minimal_indices, n, sorted_faces, face_frozens
    )
    subdivision_facets = [
        tuple(sorted(vertex_map[sorted_faces[idx]] for idx in chain))
        for chain in maximal_chains
    ]
    # Ensure maximality (covers already guarantees, but keep dedup)
    # Remove duplicates and sort deterministically
    unique_facets = sorted(set(subdivision_facets), key=lambda f: (-len(f), f))
    # No further filtering needed; they are maximal chains.
    maximal = unique_facets
    subdivision_vertex_faces = tuple(sorted_faces)
    # Build canonical complex for subdivision
    subdivision_complex = None
    if maximal:
        sub_vertices = tuple(sorted(new_vertices))
        # maximal facets for complex must be sorted tuple of sorted vertices
        canon_facets = tuple(sorted(tuple(sorted(f)) for f in maximal))
        subdivision_complex = _canonical_complex(sub_vertices, canon_facets)
    return BarycentricSubdivisionResult(
        original_vertices=request.complex.vertices,
        original_dimension=max(len(f) - 1 for f in request.complex.facets),
        subdivision_vertices=tuple(new_vertices),
        subdivision_facets=tuple(maximal),
        num_new_vertices=len(new_vertices),
        complex=request.complex,
        subdivision_complex=subdivision_complex,
        subdivision_vertex_faces=subdivision_vertex_faces,
    )


def compute_pseudomanifold_decision(
    request: PseudomanifoldRequest,
) -> PseudomanifoldResult:
    """Decide whether a complex is a pseudomanifold."""
    facets = [frozenset(f) for f in request.complex.facets]
    dim = max(len(f) - 1 for f in facets) if facets else 0

    # Purity: all facets must have the same dimension
    if not all(len(f) - 1 == dim for f in facets):
        return PseudomanifoldResult(
            complex=request.complex,
            is_pseudomanifold=False,
            is_closed=False,
            dimension=dim,
            num_facets=len(facets),
            obstruction="not pure: facets have different dimensions",
        )

    # Each codimension-1 face must be in exactly 1 or 2 facets; for a
    # dimension-zero complex that face is the empty face, contained once per
    # vertex facet.
    codim1_count: dict[frozenset[str], int] = {}
    for facet in facets:
        for face in combinations(sorted(facet), len(facet) - 1):
            key = frozenset(face)
            codim1_count[key] = codim1_count.get(key, 0) + 1

    for codim_face, count in codim1_count.items():
        if count > 2:
            return PseudomanifoldResult(
                complex=request.complex,
                is_pseudomanifold=False,
                is_closed=False,
                dimension=dim,
                num_facets=len(facets),
                obstruction=f"codim-1 face {sorted(codim_face)} is in {count} facets",
            )

    is_closed = all(count == 2 for count in codim1_count.values())
    return PseudomanifoldResult(
        complex=request.complex,
        is_pseudomanifold=True,
        is_closed=is_closed,
        dimension=dim,
        num_facets=len(facets),
        obstruction=None if is_closed else "pseudomanifold with boundary",
    )


def compute_shelling_check(request: ShellingCheckRequest) -> ShellingCheckResult:
    """Check whether a submitted facet order is a valid shelling order."""

    is_shelling, failed_at, failure_reason = _evaluate_shelling(
        request.complex.facets, request.facet_order
    )
    return ShellingCheckResult(
        complex=request.complex,
        facet_order=request.facet_order,
        is_shelling=is_shelling,
        failed_at=failed_at,
        failure_reason=failure_reason,
    )


def compute_elementary_collapse(
    request: ElementaryCollapseRequest,
) -> ElementaryCollapseResult:
    """Check and perform one elementary collapse step."""
    free_face_set = frozenset(request.free_face)
    coface_set = frozenset(request.coface)

    all_faces_set = _all_faces(request.complex.facets)
    coface_tuple = tuple(sorted(request.coface))
    free_tuple = tuple(sorted(request.free_face))

    def _collapse_result(
        is_free: bool,
        facets: tuple[tuple[str, ...], ...],
    ) -> ElementaryCollapseResult:
        if facets:
            verts = tuple(sorted({v for f in facets for v in f}))
            rem_facets = tuple(tuple(sorted(f)) for f in facets)
            remaining_complex = _canonical_complex(verts, rem_facets)
            rem_vertices = verts
        else:
            remaining_complex = None
            rem_vertices = ()
            rem_facets = ()
        return ElementaryCollapseResult(
            complex=request.complex,
            is_free_face=is_free,
            free_face=free_tuple,
            coface=coface_tuple,
            remaining_facets=rem_facets,
            remaining_vertices=rem_vertices,
            remaining_complex=remaining_complex,
        )

    # Check that free_face is a face of the complex
    if free_tuple not in all_faces_set:
        return _collapse_result(False, request.complex.facets)

    # Check that coface is a facet of the complex; simplices are vertex
    # sets, so the comparison must not depend on stored label order.
    if coface_set not in {frozenset(facet) for facet in request.complex.facets}:
        return _collapse_result(False, request.complex.facets)

    # Check that free_face is a free face: it is contained in exactly one facet
    containing_facets = [
        frozenset(f)
        for f in request.complex.facets
        if free_face_set.issubset(frozenset(f))
    ]

    if len(containing_facets) != 1:
        return _collapse_result(False, request.complex.facets)

    if containing_facets[0] != coface_set:
        return _collapse_result(False, request.complex.facets)

    # Perform the collapse: remove all faces sigma with free_face <= sigma <= coface
    # Remaining faces are those not in the interval [free_face, coface]
    remaining_faces = {
        face
        for face in all_faces_set
        if not (free_face_set.issubset(set(face)) and set(face).issubset(coface_set))
    }
    # Extract maximal facets from remaining faces
    sorted_remaining = sorted(remaining_faces, key=lambda f: (-len(f), f))
    maximal: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in sorted_remaining:
        fs = frozenset(face)
        if not any(existing.issuperset(fs) for existing in seen):
            maximal.append(face)
            seen.add(fs)
    remaining_facets = tuple(maximal)
    if remaining_facets:
        remaining_vertices = tuple(sorted({v for f in remaining_facets for v in f}))
        remaining_complex = _canonical_complex(remaining_vertices, remaining_facets)
    else:
        remaining_vertices = ()
        remaining_complex = None
    return ElementaryCollapseResult(
        complex=request.complex,
        is_free_face=True,
        free_face=free_tuple,
        coface=coface_tuple,
        remaining_facets=remaining_facets,
        remaining_vertices=remaining_vertices,
        remaining_complex=remaining_complex,
    )
