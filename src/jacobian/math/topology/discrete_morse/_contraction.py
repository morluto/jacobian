"""Bounded integral deformation retractions from simplicial chains to Morse chains."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FacesInDimension,
    FiniteSimplicialComplex,
    Simplex,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)
from jacobian.math.topology.discrete_morse._kernel import (
    _admit_acyclic_matching,
    _closure_cells,
)
from jacobian.math.topology.discrete_morse._models import (
    MAX_MORSE_CONTRACTION_CELLS,
    MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS,
    MAX_MORSE_CONTRACTION_FACE_CANDIDATES,
    MAX_MORSE_CONTRACTION_PAIRS,
    CriticalCellBasis,
    MatchingPair,
    MorseChainContractionResult,
)

MAX_MORSE_CONTRACTION_WORK = 1_000_000
MAX_MORSE_CONTRACTION_OUTPUT_BYTES = 1_000_000


def _admission(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("complex", "pairs"),
        code=f"topology.discrete_morse.contraction.{code}",
        message=message,
    )


def _identity(size: int) -> list[list[int]]:
    return [[int(row == column) for column in range(size)] for row in range(size)]


def _digits(value: int) -> int:
    return len(str(abs(value))) if value else 1


def _bounded_product(left: int, right: int) -> int:
    if (
        left
        and right
        and (
            _digits(left) + _digits(right) + 3
            > MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS
        )
    ):
        raise _admission(
            "coefficient_growth",
            "chain-contraction intermediate coefficient growth exceeds 64 decimal digits",
        )
    return left * right


def _bounded_sum(terms: list[int]) -> int:
    if terms and max(_digits(term) for term in terms) + 3 > (
        MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS
    ):
        raise _admission(
            "coefficient_growth",
            "chain-contraction intermediate coefficient growth exceeds 64 decimal digits",
        )
    result = sum(terms)
    if _digits(result) > MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS:
        raise _admission(
            "coefficient_growth",
            "chain-contraction intermediate coefficient growth exceeds 64 decimal digits",
        )
    return result


def _matrix_product(
    left: list[list[int]],
    right: list[list[int]],
    *,
    rows: int,
    middle: int,
    columns: int,
) -> list[list[int]]:
    if not middle:
        return [[0] * columns for _ in range(rows)]
    return [
        [
            _bounded_sum(
                [
                    _bounded_product(left[row][inner], right[inner][column])
                    for inner in range(middle)
                    if left[row][inner] and right[inner][column]
                ]
            )
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def _add_matrices(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [_bounded_sum([a, b]) for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _same_matrix(left: list[list[int]], right: list[list[int]]) -> bool:
    return left == right


def _boundary_matrices(
    cells: tuple[tuple[Simplex, ...], ...],
) -> tuple[list[list[int]], ...]:
    """Build the lex-oriented integral simplicial differential."""
    index = tuple(
        {face: position for position, face in enumerate(group)} for group in cells
    )
    matrices: list[list[list[int]]] = [[]]
    for degree in range(1, len(cells)):
        rows = len(cells[degree - 1])
        columns = len(cells[degree])
        matrix = [[0] * columns for _ in range(rows)]
        for column, simplex in enumerate(cells[degree]):
            for position in range(len(simplex)):
                face = simplex[:position] + simplex[position + 1 :]
                matrix[index[degree - 1][face]][column] = -1 if position % 2 else 1
        matrices.append(matrix)
    return tuple(matrices)


def _replace_lower_pivot(
    degree: int,
    lower_index: int,
    upper_index: int,
    active: list[list[int]],
    basis: list[list[list[int]]],
    inverse_basis: list[list[list[int]]],
    differentials: list[list[list[int]]],
) -> tuple[int, dict[int, int]]:
    """Replace a matched lower basis vector by its unit-pivot boundary."""
    matrix = differentials[degree + 1]
    pivot = matrix[lower_index][upper_index]
    if pivot not in (-1, 1):
        raise RuntimeError("acyclic simplicial matching lost its unit incidence pivot")
    lower_active = active[degree]
    upper_active = active[degree + 1]
    if lower_index not in lower_active or upper_index not in upper_active:
        raise RuntimeError("matching cancellation order reused an already removed cell")

    # Replace the paired lower basis vector by d(upper)/pivot. The inverse row
    # operation clears its other active coordinates from the pivot column.
    column_coefficients = {
        row: matrix[row][upper_index] // pivot for row in lower_active
    }
    if column_coefficients[lower_index] != 1 or any(
        matrix[row][upper_index] != 0
        for row in range(len(matrix))
        if row not in lower_active
    ):
        raise RuntimeError("matched pivot couples to a previously cancelled lower cell")
    old_lower_basis = [row[:] for row in basis[degree]]
    old_lower_inverse = [row[:] for row in inverse_basis[degree]]
    old_lower_differential = [row[:] for row in differentials[degree]]
    for row in range(len(basis[degree])):
        basis[degree][row][lower_index] = _bounded_sum(
            [
                _bounded_product(
                    column_coefficients[position], old_lower_basis[row][position]
                )
                for position in lower_active
                if column_coefficients[position] and old_lower_basis[row][position]
            ]
        )
    for position in lower_active:
        if position != lower_index:
            coefficient = column_coefficients[position]
            if coefficient:
                inverse_basis[degree][position] = [
                    _bounded_sum(
                        [
                            old_lower_inverse[position][column],
                            -_bounded_product(
                                coefficient, old_lower_inverse[lower_index][column]
                            ),
                        ]
                    )
                    for column in range(len(old_lower_inverse[position]))
                ]
    for row in range(len(differentials[degree])):
        differentials[degree][row][lower_index] = _bounded_sum(
            [
                _bounded_product(
                    column_coefficients[position], old_lower_differential[row][position]
                )
                for position in lower_active
                if column_coefficients[position]
                and old_lower_differential[row][position]
            ]
        )
    for row in lower_active:
        if row != lower_index:
            coefficient = column_coefficients[row]
            if coefficient:
                matrix[row] = [
                    _bounded_sum([entry, -_bounded_product(coefficient, pivot_entry)])
                    for entry, pivot_entry in zip(
                        matrix[row], matrix[lower_index], strict=True
                    )
                ]

    return pivot, column_coefficients


def _replace_upper_pivot(
    degree: int,
    lower_index: int,
    upper_index: int,
    pivot: int,
    active: list[list[int]],
    basis: list[list[list[int]]],
    inverse_basis: list[list[list[int]]],
    differentials: list[list[list[int]]],
) -> None:
    """Clear the pivot row in other active upper basis vectors."""
    matrix = differentials[degree + 1]
    lower_active = active[degree]
    upper_active = active[degree + 1]
    row_coefficients = {
        column: matrix[lower_index][column] // pivot
        for column in upper_active
        if column != upper_index
    }
    old_upper_basis = [row[:] for row in basis[degree + 1]]
    old_upper_inverse = [row[:] for row in inverse_basis[degree + 1]]
    old_upper_next_differential = (
        [row[:] for row in differentials[degree + 2]]
        if degree + 2 < len(differentials)
        else []
    )
    for column, coefficient in row_coefficients.items():
        for row in range(len(basis[degree + 1])):
            basis[degree + 1][row][column] = _bounded_sum(
                [
                    old_upper_basis[row][column],
                    -_bounded_product(coefficient, old_upper_basis[row][upper_index]),
                ]
            )
        for row in range(len(matrix)):
            matrix[row][column] = _bounded_sum(
                [
                    matrix[row][column],
                    -_bounded_product(coefficient, matrix[row][upper_index]),
                ]
            )
    if row_coefficients:
        inverse_basis[degree + 1][upper_index] = [
            _bounded_sum(
                [
                    old_upper_inverse[upper_index][column],
                    *(
                        _bounded_product(
                            coefficient, old_upper_inverse[position][column]
                        )
                        for position, coefficient in row_coefficients.items()
                    ),
                ]
            )
            for column in range(len(old_upper_inverse[upper_index]))
        ]
        if old_upper_next_differential:
            next_differential = differentials[degree + 2]
            next_differential[upper_index] = [
                _bounded_sum(
                    [
                        old_upper_next_differential[upper_index][column],
                        *(
                            _bounded_product(
                                coefficient,
                                old_upper_next_differential[position][column],
                            )
                            for position, coefficient in row_coefficients.items()
                        ),
                    ]
                )
                for column in range(len(next_differential[upper_index]))
            ]

    if (
        matrix[lower_index][upper_index] != pivot
        or any(
            matrix[lower_index][column] != 0
            for column in upper_active
            if column != upper_index
        )
        or any(
            matrix[row][upper_index] != 0 for row in lower_active if row != lower_index
        )
    ):
        raise RuntimeError("unit cancellation failed to isolate its contractible pair")


def _cancel_pair(
    degree: int,
    lower_index: int,
    upper_index: int,
    active: list[list[int]],
    basis: list[list[list[int]]],
    inverse_basis: list[list[list[int]]],
    differentials: list[list[list[int]]],
) -> int:
    """Split one unit pivot as a contractible direct summand."""
    if lower_index not in active[degree] or upper_index not in active[degree + 1]:
        raise RuntimeError("matching cancellation order reused an already removed cell")
    pivot, _ = _replace_lower_pivot(
        degree,
        lower_index,
        upper_index,
        active,
        basis,
        inverse_basis,
        differentials,
    )
    _replace_upper_pivot(
        degree,
        lower_index,
        upper_index,
        pivot,
        active,
        basis,
        inverse_basis,
        differentials,
    )
    active[degree].remove(lower_index)
    active[degree + 1].remove(upper_index)
    return pivot


def _preflight_inputs(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> tuple[tuple[tuple[Simplex, ...], ...], int]:
    """Bound constructed source and matching shapes before revalidation."""
    if not isinstance(complex_, FiniteSimplicialComplex):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.discrete_morse.contraction.complex_type",
            message="chain contraction requires a canonical finite simplicial complex",
        )
    if (
        not isinstance(complex_.vertices, tuple)
        or len(complex_.vertices) > MAX_TOPOLOGY_VERTICES
        or not isinstance(complex_.maximal_simplices, tuple)
        or len(complex_.maximal_simplices) > MAX_TOPOLOGY_FACETS
        or not isinstance(complex_.faces_by_dimension, tuple)
        or len(complex_.faces_by_dimension) > MAX_TOPOLOGY_DIMENSION + 1
        or any(
            not isinstance(facet, tuple)
            or not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1
            or any(not isinstance(vertex, str) or len(vertex) > 32 for vertex in facet)
            for facet in complex_.maximal_simplices
        )
        or any(
            not isinstance(vertex, str) or len(vertex) > 32
            for vertex in complex_.vertices
        )
    ):
        raise _admission(
            "axes",
            "source complex axes exceed the bounded chain-contraction envelope",
        )
    possible_faces = sum((1 << len(facet)) - 1 for facet in complex_.maximal_simplices)
    if possible_faces > MAX_MORSE_CONTRACTION_FACE_CANDIDATES:
        raise _admission(
            "face_candidates",
            "the source facets exceed the 512-candidate closure admission bound",
        )
    if any(
        not isinstance(group, FacesInDimension)
        or not isinstance(group.faces, tuple)
        or any(
            not isinstance(face, tuple)
            or len(face) > MAX_TOPOLOGY_DIMENSION + 1
            or any(not isinstance(vertex, str) or len(vertex) > 32 for vertex in face)
            for face in group.faces
        )
        for group in complex_.faces_by_dimension
    ):
        raise _admission("axes", "source face groups must be bounded immutable tuples")
    if (
        sum(len(group.faces) for group in complex_.faces_by_dimension)
        > MAX_MORSE_CONTRACTION_CELLS
    ):
        raise _admission("cells", "the supplied face closure exceeds 32 source cells")
    if not isinstance(pairs, tuple) or len(pairs) > MAX_MORSE_CONTRACTION_PAIRS:
        raise _admission("pairs", "the supplied matching exceeds 14 pairs")
    if any(
        not isinstance(pair, MatchingPair)
        or not isinstance(pair.face, tuple)
        or not isinstance(pair.coface, tuple)
        or len(pair.face) > MAX_TOPOLOGY_DIMENSION + 1
        or len(pair.coface) > MAX_TOPOLOGY_DIMENSION + 1
        or any(
            not isinstance(vertex, str) or len(vertex) > 32
            for cell in (pair.face, pair.coface)
            for vertex in cell
        )
        for pair in pairs
    ):
        raise _admission("matching_shape", "matching cells exceed their canonical axes")
    cells = _closure_cells(complex_)
    return cells, sum(map(len, cells))


def _admit_contraction_work(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
    cells: tuple[tuple[Simplex, ...], ...],
    total_cells: int,
) -> FiniteSimplicialComplex:
    """Admit exact matrix work and worst-case serialized values."""
    if total_cells > MAX_MORSE_CONTRACTION_CELLS:
        raise _admission("cells", "chain contraction is limited to 32 source cells")
    if len(pairs) > MAX_MORSE_CONTRACTION_PAIRS:
        raise _admission("pairs", "chain contraction is limited to 14 matched pairs")
    closure_candidates = sum(
        (1 << len(facet)) - 1 for facet in complex_.maximal_simplices
    )
    base_work = (
        closure_candidates + 16 * total_cells**3 + 16 * len(pairs) * total_cells**2
    )
    if base_work > MAX_MORSE_CONTRACTION_WORK:
        raise _admission(
            "work",
            "exact chain-contraction matrix work exceeds its one-million-unit bound",
        )
    label_characters = sum(map(len, complex_.vertices))
    label_characters += sum(
        len(vertex) for facet in complex_.maximal_simplices for vertex in facet
    )
    label_characters += sum(
        len(vertex) for group in cells for cell in group for vertex in cell
    )
    label_characters += sum(
        len(vertex)
        for pair in pairs
        for cell in (pair.face, pair.coface)
        for vertex in cell
    )
    output_bound = (
        20_000
        + 5 * total_cells**2 * (MAX_MORSE_CONTRACTION_COEFFICIENT_DIGITS + 2)
        + 6 * label_characters
    )
    if output_bound > MAX_MORSE_CONTRACTION_OUTPUT_BYTES:
        raise _admission(
            "output",
            "chain-contraction endpoints and maps exceed the serialized output bound",
        )

    try:
        admitted = FiniteSimplicialComplex.model_validate(
            complex_.model_dump(), strict=True
        )
        require_canonical_complex_admission(admitted)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.discrete_morse.contraction.complex_shape",
            message="source complex must have its exact canonical face closure and axes",
        ) from exc
    return admitted


def compute_chain_contraction(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> MorseChainContractionResult:
    """Return an exact ZZ contraction from simplicial to Morse chains."""
    cells, total_cells = _preflight_inputs(complex_, pairs)
    complex_ = _admit_contraction_work(complex_, pairs, cells, total_cells)
    cells = _closure_cells(complex_)
    dimensions = len(cells)
    ranks = tuple(map(len, cells))

    admitted = _admit_acyclic_matching(complex_, pairs)
    matching = admitted.result
    source_differentials = _boundary_matrices(cells)
    differentials = [[row[:] for row in matrix] for matrix in source_differentials]
    basis = [_identity(rank) for rank in ranks]
    inverse_basis = [_identity(rank) for rank in ranks]
    active = [list(range(rank)) for rank in ranks]
    index = tuple(
        {face: position for position, face in enumerate(group)} for group in cells
    )
    topo = {cell: position for position, cell in enumerate(matching.topological_order)}
    ordered_pairs = sorted(matching.pairs, key=lambda pair: topo[pair.coface])
    pivots: list[tuple[int, int, int, int]] = []
    for pair in ordered_pairs:
        degree = len(pair.face) - 1
        lower_index = index[degree][pair.face]
        upper_index = index[degree + 1][pair.coface]
        pivot = _cancel_pair(
            degree,
            lower_index,
            upper_index,
            active,
            basis,
            inverse_basis,
            differentials,
        )
        pivots.append((degree, lower_index, upper_index, pivot))

    critical_cells = tuple(
        CriticalCellBasis(
            dimension=degree,
            cells=tuple(
                cell
                for cell in cells[degree]
                if cell not in {pair.face for pair in matching.pairs}
                and cell not in {pair.coface for pair in matching.pairs}
            ),
        )
        for degree in range(dimensions)
    )
    # Active coordinate positions retain the canonical critical-cell order.
    target_differentials = tuple(
        tuple(
            tuple(differentials[degree][row][column] for column in active[degree])
            for row in active[degree - 1]
        )
        for degree in range(1, dimensions)
    )
    source_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=complex_.dimension,
        basis_sizes=ranks,
        differential_matrices=tuple(
            tuple(tuple(row) for row in matrix) for matrix in source_differentials[1:]
        ),
    )
    target_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=complex_.dimension,
        basis_sizes=tuple(len(group.cells) for group in critical_cells),
        differential_matrices=target_differentials,
    )

    inclusion = tuple(
        tuple(
            tuple(basis[degree][row][column] for column in active[degree])
            for row in range(ranks[degree])
        )
        for degree in range(dimensions)
    )
    projection = tuple(
        tuple(
            tuple(inverse_basis[degree][row][column] for column in range(ranks[degree]))
            for row in active[degree]
        )
        for degree in range(dimensions)
    )
    homotopy_matrices: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(dimensions):
        next_rank = ranks[degree + 1] if degree + 1 < dimensions else 0
        h = [[0] * ranks[degree] for _ in range(next_rank)]
        for pair_degree, lower, upper, pivot in pivots:
            if pair_degree == degree:
                h[upper][lower] = pivot
        transformed = _matrix_product(
            basis[degree + 1] if degree + 1 < dimensions else [],
            h,
            rows=next_rank,
            middle=next_rank,
            columns=ranks[degree],
        )
        homotopy = _matrix_product(
            transformed,
            inverse_basis[degree],
            rows=next_rank,
            middle=ranks[degree],
            columns=ranks[degree],
        )
        homotopy_matrices.append(tuple(tuple(row) for row in homotopy))

    # Verify both chain-map equations, P I = 1, and id - I P = dH + Hd.
    for degree in range(dimensions):
        source_rank = ranks[degree]
        target_rank = len(active[degree])
        identity_target = _identity(target_rank)
        projection_after_inclusion = _matrix_product(
            [list(row) for row in projection[degree]],
            [list(row) for row in inclusion[degree]],
            rows=target_rank,
            middle=source_rank,
            columns=target_rank,
        )
        if not _same_matrix(projection_after_inclusion, identity_target):
            raise RuntimeError("computed Morse contraction violates P I = identity")
        if degree > 0:
            d_source = source_differentials[degree]
            d_target = [list(row) for row in target_differentials[degree - 1]]
            left_i = _matrix_product(
                d_source,
                [list(row) for row in inclusion[degree]],
                rows=ranks[degree - 1],
                middle=source_rank,
                columns=target_rank,
            )
            right_i = _matrix_product(
                [list(row) for row in inclusion[degree - 1]],
                d_target,
                rows=ranks[degree - 1],
                middle=len(active[degree - 1]),
                columns=target_rank,
            )
            if not _same_matrix(left_i, right_i):
                raise RuntimeError("computed Morse inclusion is not a chain map")
            left_p = _matrix_product(
                [list(row) for row in projection[degree - 1]],
                d_source,
                rows=len(active[degree - 1]),
                middle=ranks[degree - 1],
                columns=source_rank,
            )
            right_p = _matrix_product(
                d_target,
                [list(row) for row in projection[degree]],
                rows=len(active[degree - 1]),
                middle=len(active[degree]),
                columns=source_rank,
            )
            if not _same_matrix(left_p, right_p):
                raise RuntimeError("computed Morse projection is not a chain map")
        identity_source = _identity(source_rank)
        ip = _matrix_product(
            [list(row) for row in inclusion[degree]],
            [list(row) for row in projection[degree]],
            rows=source_rank,
            middle=target_rank,
            columns=source_rank,
        )
        left_h = [
            [
                _bounded_sum([identity_source[row][column], -ip[row][column]])
                for column in range(source_rank)
            ]
            for row in range(source_rank)
        ]
        right_h = [[0] * source_rank for _ in range(source_rank)]
        if degree + 1 < dimensions:
            term = _matrix_product(
                source_differentials[degree + 1],
                [list(row) for row in homotopy_matrices[degree]],
                rows=source_rank,
                middle=ranks[degree + 1],
                columns=source_rank,
            )
            right_h = _add_matrices(right_h, term)
        if degree > 0:
            term = _matrix_product(
                [list(row) for row in homotopy_matrices[degree - 1]],
                source_differentials[degree],
                rows=ranks[degree],
                middle=ranks[degree - 1],
                columns=source_rank,
            )
            right_h = _add_matrices(right_h, term)
        if not _same_matrix(left_h, right_h):
            raise RuntimeError("computed Morse contraction violates id - I P = dH + Hd")

    return MorseChainContractionResult._from_kernel(
        complex=complex_,
        pairs=matching.pairs,
        source_chain_complex=source_value,
        critical_cells_by_dimension=critical_cells,
        morse_chain_complex=target_value,
        inclusion_matrices=inclusion,
        projection_matrices=projection,
        homotopy_matrices=tuple(homotopy_matrices),
    )


__all__ = ["compute_chain_contraction"]
