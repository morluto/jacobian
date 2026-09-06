"""Exact finite simplicial-complex and homology operations."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.finite_fields import linear_algebra as prime_field
from jacobian.math.topology._barycentric import (
    barycentric_subdivision as _barycentric_kernel,
)
from jacobian.math.topology._chain_conversion import (
    canonical_chain_complex_value_from_parts,
)
from jacobian.math.topology._homology import (
    MAX_INLINE_HOMOLOGY_CHAIN_GROUP,
    HomologyGroupResult,
    IntegralSimplicialHomologyResult,
    ModularVector,
    SimplicialHomologyResult,
)
from jacobian.math.topology._models import (
    MAX_BARYCENTRIC_SOURCE_FACES,
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    BarycentricSubdivisionResult,
    BoundarySquareLedgerEntry,
    ChainCoefficientRing,
    ChainComplexResult,
    FiniteSimplicialComplex,
    HomologyConvention,
    ShellingCheckResult,
    SimplexBasis,
    SimplicialComplexCanonicalizationResult,
    SparseBoundaryMatrix,
    SparseMatrixEntry,
    _all_faces,
    _require_canonical_conversion_bounds,
    _require_request_complex,
    canonical_complex,
    face_closure,
    is_bounded_prime,
    require_linear_algebra_bounds,
)
from jacobian.math.topology._pseudomanifold import (
    PseudomanifoldResult,
    pseudomanifold_decision,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
    run_topology_admission,
)
from jacobian.math.topology._shelling import evaluate_shelling
from jacobian.math.topology.chain_complexes.operations import homology_groups


def _admit_chain(
    complex_: FiniteSimplicialComplex,
    coefficient_ring: ChainCoefficientRing,
    prime: int | None,
    convention: HomologyConvention,
) -> None:
    require_canonical_complex_admission(complex_)
    if coefficient_ring is ChainCoefficientRing.INTEGER:
        if prime is not None:
            raise ValueError("integer chain complexes must not declare a prime")
    elif prime is None or not is_bounded_prime(prime):
        raise ValueError("prime-field chain complexes require a bounded prime")
    require_linear_algebra_bounds(complex_)
    _require_canonical_conversion_bounds(complex_, convention)


def _admit_homology(
    complex_: FiniteSimplicialComplex,
    prime: int,
    convention: HomologyConvention,
) -> None:
    if not is_bounded_prime(prime):
        raise ValueError("homology coefficients require a bounded prime")
    if any(size > MAX_INLINE_HOMOLOGY_CHAIN_GROUP for size in complex_.f_vector):
        raise ValueError(
            "inline homology bases require at most "
            f"{MAX_INLINE_HOMOLOGY_CHAIN_GROUP} simplices in each chain group"
        )
    _admit_chain(complex_, ChainCoefficientRing.PRIME_FIELD, prime, convention)


def _admit_integral_homology(
    complex_: FiniteSimplicialComplex,
    convention: HomologyConvention,
) -> None:
    # Simplicial admission owns materialization of the canonical chain value.
    # The chain-complex owner then admits d^2, both Smith reductions,
    # transformation/generator height, exact output, and the shared deadline.
    _admit_chain(complex_, ChainCoefficientRing.INTEGER, None, convention)


def canonicalize(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
) -> SimplicialComplexCanonicalizationResult:
    def admit() -> tuple[
        tuple[tuple[str, ...], ...], tuple[tuple[tuple[str, ...], ...], ...]
    ]:
        canonical_facets = _require_request_complex(
            vertices, facets, check_closure=False
        )
        closure = face_closure(canonical_facets)
        if sum(map(len, closure)) > MAX_TOPOLOGY_FACES:
            raise ValueError(
                f"face closure may contain at most {MAX_TOPOLOGY_FACES} non-empty faces"
            )
        return canonical_facets, closure

    canonical_facets, closure = run_topology_admission(admit, location=("facets",))
    return SimplicialComplexCanonicalizationResult(
        complex=canonical_complex(vertices, canonical_facets, closure=closure)
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


def _chain_parts(
    complex_: FiniteSimplicialComplex,
    coefficient_ring: ChainCoefficientRing,
    prime: int | None,
    convention: HomologyConvention,
) -> tuple[
    tuple[SimplexBasis, ...],
    tuple[SparseBoundaryMatrix, ...],
    SparseBoundaryMatrix | None,
]:
    """Construct the based simplicial groups and differentials once."""

    bases = tuple(
        SimplexBasis(dimension=item.dimension, simplices=item.faces)
        for item in complex_.faces_by_dimension
    )
    boundaries = tuple(
        _boundary_matrix(
            complex_,
            dimension,
            coefficient_ring=coefficient_ring,
            prime=prime,
        )
        for dimension in range(complex_.dimension + 1)
    )
    augmentation = (
        _augmentation(len(complex_.vertices))
        if convention is HomologyConvention.REDUCED
        else None
    )
    return bases, boundaries, augmentation


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


def chain_complex(
    complex_: FiniteSimplicialComplex,
    coefficient_ring: ChainCoefficientRing,
    prime: int | None,
    convention: HomologyConvention,
    *,
    admitted: bool = False,
) -> ChainComplexResult:
    if not admitted:
        run_topology_admission(
            lambda: _admit_chain(complex_, coefficient_ring, prime, convention),
            location=("complex",),
        )
    bases, boundaries, augmentation = _chain_parts(
        complex_,
        coefficient_ring,
        prime,
        convention,
    )
    modulus = prime if coefficient_ring is ChainCoefficientRing.PRIME_FIELD else None
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
    return ChainComplexResult._from_kernel(
        complex=complex_,
        coefficient_ring=coefficient_ring,
        prime=prime,
        convention=convention,
        simplex_bases=bases,
        boundary_matrices=boundaries,
        augmentation=augmentation,
        boundary_squared_zero=tuple(ledger),
        canonical_value=canonical_chain_complex_value_from_parts(
            coefficient_ring,
            convention,
            prime,
            bases,
            boundaries,
            augmentation,
        ),
    )


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


def homology(
    complex_: FiniteSimplicialComplex,
    prime: int,
    convention: HomologyConvention,
) -> SimplicialHomologyResult:
    run_topology_admission(
        lambda: _admit_homology(complex_, prime, convention), location=("complex",)
    )
    chain = chain_complex(
        complex_,
        ChainCoefficientRing.PRIME_FIELD,
        prime,
        convention,
        admitted=True,
    )
    boundaries = tuple(
        _dense(matrix, modulus=prime) for matrix in chain.boundary_matrices
    )
    augmentation = (
        None
        if chain.augmentation is None
        else _dense(chain.augmentation, modulus=prime)
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
        outgoing_matrix = _prime_matrix(outgoing, columns=chain_dimension, prime=prime)
        cycles = prime_field.nullspace(outgoing_matrix)
        outgoing_rank = prime_field.rank(outgoing_matrix)
        if dimension < complex_.dimension:
            incoming = boundaries[dimension + 1]
            incoming_columns = len(chain.simplex_bases[dimension + 1].simplices)
        else:
            incoming = [[] for _ in range(chain_dimension)]
            incoming_columns = 0
        boundary_basis = prime_field.column_basis(
            _prime_matrix(
                incoming,
                columns=incoming_columns,
                prime=prime,
            )
        )
        homology_basis = prime_field.quotient_basis(
            cycles,
            boundary_basis,
            prime=prime,
        )
        quotient_span_rank = _vector_rank(
            (*boundary_basis, *homology_basis),
            prime=prime,
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
    return SimplicialHomologyResult.model_construct(
        complex=complex_,
        prime=prime,
        convention=convention,
        dimension_range=(0, complex_.dimension),
        groups=tuple(groups),
    )


def integral_homology(
    complex_: FiniteSimplicialComplex,
    convention: HomologyConvention,
) -> IntegralSimplicialHomologyResult:
    run_topology_admission(
        lambda: _admit_integral_homology(complex_, convention),
        location=("complex",),
    )
    bases, boundaries, augmentation = _chain_parts(
        complex_,
        ChainCoefficientRing.INTEGER,
        None,
        convention,
    )
    chain_value = canonical_chain_complex_value_from_parts(
        ChainCoefficientRing.INTEGER,
        convention,
        None,
        bases,
        boundaries,
        augmentation,
    )
    homology = homology_groups(chain_value)
    return IntegralSimplicialHomologyResult.model_construct(
        complex=complex_,
        convention=convention,
        homology=homology,
    )


def barycentric_subdivision(
    complex_: FiniteSimplicialComplex,
) -> BarycentricSubdivisionResult:
    """Compute the barycentric subdivision of a simplicial complex."""

    sorted_faces = sorted(
        _all_faces(complex_.maximal_simplices), key=lambda face: (len(face), face)
    )
    if len(sorted_faces) > MAX_BARYCENTRIC_SOURCE_FACES:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.require_barycentric_work_bounds_1",
            message=(
                "barycentric subdivision requires at most "
                f"{MAX_BARYCENTRIC_SOURCE_FACES} faces; input would produce "
                f"more than {MAX_TOPOLOGY_FACETS} subdivision facets"
            ),
        )
    subdivision = _barycentric_kernel(sorted_faces)
    facets = tuple(sorted(tuple(sorted(facet)) for facet in subdivision.facets))
    return BarycentricSubdivisionResult._from_kernel(
        original_vertices=complex_.vertices,
        original_dimension=max(len(facet) - 1 for facet in complex_.maximal_simplices),
        subdivision_vertices=subdivision.vertices,
        subdivision_facets=subdivision.facets,
        num_new_vertices=len(subdivision.vertices),
        complex=complex_,
        subdivision_complex=(
            canonical_complex(tuple(sorted(subdivision.vertices)), facets)
            if facets
            else None
        ),
        subdivision_vertex_faces=subdivision.vertex_faces,
    )


def pseudomanifold(complex_: FiniteSimplicialComplex) -> PseudomanifoldResult:
    """Decide whether a complex is a pseudomanifold."""

    decision = pseudomanifold_decision(complex_.maximal_simplices)
    return PseudomanifoldResult._from_kernel(complex_=complex_, decision=decision)


def shelling_check(
    complex_: FiniteSimplicialComplex,
    facet_order: tuple[int, ...],
) -> ShellingCheckResult:
    """Check whether a submitted facet order is a valid shelling order."""
    if sorted(facet_order) != list(range(len(complex_.maximal_simplices))):
        raise OperationDomainValidationError(
            location=("facet_order",),
            code="topology.shelling_facet_order",
            message="facet_order must be a permutation of facet indices",
        )

    is_shelling, failed_at, failure_reason = evaluate_shelling(
        complex_.maximal_simplices, facet_order
    )
    return ShellingCheckResult._from_kernel(
        complex=complex_,
        facet_order=facet_order,
        is_shelling=is_shelling,
        failed_at=failed_at,
        failure_reason=failure_reason,
    )
