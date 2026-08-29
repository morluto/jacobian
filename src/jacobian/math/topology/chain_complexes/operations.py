"""Exact finite based chain complex operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.math.topology.chain_complexes._models import (
    _require_chain_map_components,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_MATRIX_ENTRY_CHARS,
    MAX_TENSOR_COEFFICIENT_DIGITS,
    MAX_TENSOR_GROUP_DIMENSION,
    MAX_TENSOR_TOTAL_CELLS,
    ChainComplexValue,
    CoefficientField,
    HomologyGroupValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
    VerificationResult,
)

MapMatrices = tuple[tuple[tuple[str, ...], ...], ...]

__all__ = [
    "chain_map_commutes",
    "construct_chain_complex",
    "differential_squares_to_zero",
    "homology_groups",
    "mapping_cone",
    "tensor_product_complex",
]


class ChainComplexAdmissionError(ValueError):
    """Expected boundedness failure before a derived complex is allocated."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _require_tensor_admission(
    left: ChainComplexValue, right: ChainComplexValue
) -> None:
    if left.coefficient_field != right.coefficient_field or left.prime != right.prime:
        raise ChainComplexAdmissionError(
            "tensor_context_mismatch",
            "tensor product requires same coefficient field and prime",
        )
    group_count = len(left.basis_sizes) + len(right.basis_sizes) - 1
    group_sizes: list[int] = []
    for degree in range(group_count):
        size = sum(
            left.basis_sizes[i] * right.basis_sizes[degree - i]
            for i in range(min(degree + 1, len(left.basis_sizes)))
            if degree - i < len(right.basis_sizes)
        )
        if size > MAX_TENSOR_GROUP_DIMENSION:
            raise ChainComplexAdmissionError(
                "tensor_group_dimension_budget_exceeded",
                f"tensor product group dimension {size} exceeds the "
                f"{MAX_TENSOR_GROUP_DIMENSION}-dimension work bound",
            )
        group_sizes.append(size)
    allocated_cells = sum(
        group_sizes[index - 1] * group_sizes[index] for index in range(1, group_count)
    )
    if (
        sum(group_sizes) > MAX_TENSOR_TOTAL_CELLS
        or allocated_cells > MAX_TENSOR_TOTAL_CELLS
    ):
        raise ChainComplexAdmissionError(
            "tensor_cell_budget_exceeded",
            f"tensor product allocates {max(sum(group_sizes), allocated_cells)} cells, "
            f"exceeding the {MAX_TENSOR_TOTAL_CELLS}-cell work bound",
        )
    for complex_value in (left, right):
        for matrix in complex_value.differential_matrices:
            for row in matrix:
                for entry in row:
                    numerator, _, denominator = entry.partition("/")
                    if (
                        len(numerator.lstrip("-")) > MAX_TENSOR_COEFFICIENT_DIGITS
                        or len(denominator.lstrip("-")) > MAX_TENSOR_COEFFICIENT_DIGITS
                    ):
                        raise ChainComplexAdmissionError(
                            "tensor_coefficient_digit_budget_exceeded",
                            "tensor product inputs are limited to "
                            f"{MAX_TENSOR_COEFFICIENT_DIGITS}-digit coefficients",
                        )
    _require_square_zero(
        _parsed_differentials(left, left.prime),
        left.prime,
        label="tensor product left",
        group_columns=list(left.basis_sizes),
        degree_min=left.degree_min,
    )
    _require_square_zero(
        _parsed_differentials(right, right.prime),
        right.prime,
        label="tensor product right",
        group_columns=list(right.basis_sizes),
        degree_min=right.degree_min,
    )
    tensor_degree_min = left.degree_min + right.degree_min
    placeholder_diffs = tuple(
        tuple(("0",) * group_sizes[index + 1] for _ in range(group_sizes[index]))
        for index in range(max(0, group_count - 1))
    )
    ChainComplexValue(
        coefficient_field=left.coefficient_field,
        prime=left.prime,
        degree_min=tensor_degree_min,
        degree_max=tensor_degree_min + group_count - 1,
        basis_sizes=tuple(group_sizes),
        differential_matrices=placeholder_diffs,
    )

    def max_entry_length(value: ChainComplexValue) -> int:
        return max(
            (
                len(entry)
                for matrix in value.differential_matrices
                for row in matrix
                for entry in row
            ),
            default=1,
        )

    worst_entry_chars = max(max_entry_length(left), max_entry_length(right)) + 1
    if allocated_cells * worst_entry_chars > MAX_MATRIX_ENTRY_CHARS:
        raise ChainComplexAdmissionError(
            "tensor_output_budget_exceeded",
            f"tensor product serialization exceeds the canonical {MAX_MATRIX_ENTRY_CHARS}-character budget",
        )


def _require_cone_admission(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> None:
    cone_basis_sizes = tuple(
        (target.basis_sizes[index] if index < len(target.basis_sizes) else 0)
        + (source.basis_sizes[index - 1] if 0 < index <= len(source.basis_sizes) else 0)
        for index in range(max(len(source.basis_sizes), len(target.basis_sizes)) + 1)
    )
    placeholder_diffs = tuple(
        tuple(
            ("0",) * cone_basis_sizes[index + 1] for _ in range(cone_basis_sizes[index])
        )
        for index in range(max(0, len(cone_basis_sizes) - 1))
    )
    ChainComplexValue(
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        degree_min=source.degree_min,
        degree_max=source.degree_min + len(cone_basis_sizes) - 1,
        basis_sizes=cone_basis_sizes,
        differential_matrices=placeholder_diffs,
    )
    cone_cells = sum(
        cone_basis_sizes[index] * cone_basis_sizes[index + 1]
        for index in range(len(cone_basis_sizes) - 1)
    )
    entry_chars = sum(
        len(entry)
        for value in (source, target)
        for matrix in value.differential_matrices
        for row in matrix
        for entry in row
    ) + sum(len(entry) for matrix in map_matrices for row in matrix for entry in row)
    if entry_chars + cone_cells > MAX_MATRIX_ENTRY_CHARS:
        raise ChainComplexAdmissionError(
            "mapping_cone_output_budget_exceeded",
            f"mapping cone serialization exceeds the canonical {MAX_MATRIX_ENTRY_CHARS}-character budget",
        )


def _parse_fraction(s: str, prime: int | None = None) -> Fraction:
    if prime is not None:
        # Prime-field entries are canonical residues 0..p-1; accept integer strings only
        try:
            val = int(s)
        except ValueError as error:
            raise ValueError(
                f"prime-field entry '{s}' must be an integer residue"
            ) from error
        return Fraction(val % prime)
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _matrix_to_fractions(
    matrix: tuple[tuple[str, ...], ...],
    rows: int,
    cols: int,
    prime: int | None = None,
) -> list[list[Fraction]]:
    result = [[Fraction(0)] * cols for _ in range(rows)]
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            result[i][j] = _parse_fraction(val, prime)
    return result


def _rank_over_prime_field(mat: list[list[Fraction]], prime: int) -> int:
    """Gauss-Jordan elimination inside GF(p) using modular inverses."""
    rows = len(mat)
    cols = len(mat[0])
    work = [[int(v) % prime for v in row] for row in mat]
    rank = 0
    for col in range(cols):
        pivot = next(
            (row for row in range(rank, rows) if work[row][col] % prime != 0),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][col], -1, prime)
        for j in range(cols):
            work[rank][j] = (work[rank][j] * inverse) % prime
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][col] % prime
            if factor != 0:
                for j in range(cols):
                    work[row][j] = (work[row][j] - factor * work[rank][j]) % prime
        rank += 1
        if rank == rows:
            break
    return rank


def _rank_over_rationals(mat: list[list[Fraction]]) -> int:
    """Gauss-Jordan elimination over QQ."""
    rows = len(mat)
    cols = len(mat[0])
    work = [row[:] for row in mat]
    rank = 0
    for col in range(cols):
        pivot = next(
            (row for row in range(rank, rows) if work[row][col] != 0),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for row in range(rows):
            if row == rank or work[row][col] == 0:
                continue
            factor = work[row][col] / work[rank][col]
            for j in range(cols):
                work[row][j] -= factor * work[rank][j]
        rank += 1
        if rank == rows:
            break
    return rank


def _matrix_rank(matrix: list[list[Fraction]], prime: int | None = None) -> int:
    """Compute rank of a matrix over QQ or GF(p)."""
    if not matrix or not matrix[0]:
        return 0
    if prime is not None:
        return _rank_over_prime_field(matrix, prime)
    return _rank_over_rationals(matrix)


def _matrix_multiply(
    a: list[list[Fraction]],
    b: list[list[Fraction]],
    prime: int | None = None,
    result_columns: int | None = None,
    left_declared_columns: int | None = None,
) -> list[list[Fraction]]:
    rows_a = len(a)
    # A zero-row left operand keeps its declared column count so the
    # inner product dimension matches the shape contract, mirroring how
    # the right operand's outer width is preserved below.
    cols_a = len(a[0]) if a else (left_declared_columns or 0)
    if b:
        cols_b = len(b[0])
        if len(b) != cols_a:
            raise ValueError("inner product dimensions do not match")
    else:
        # A zero-row right operand keeps its declared column count so the
        # outer dimensions of a zero-width product are preserved.
        cols_b = result_columns if result_columns is not None else 0
        if cols_a and not b:
            raise ValueError("inner product dimensions do not match")
    result = [[Fraction(0)] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = Fraction(0)
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            if prime is not None:
                # Reduce modulo p: convert to int residue then back to Fraction
                # Since entries are residues, product sums are integers mod p
                # We need to map via int(s) % prime
                # Handle Fractions that are already residues (denominator 1)
                # For general prime field, s should be integer; reduce mod p
                if s.denominator != 1:
                    # For prime field, denominators should be 1; if not, it's an error, but convert via modular inverse
                    # Compute numerator * inv(denominator) mod p
                    inv_den = pow(s.denominator, -1, prime)
                    s = Fraction((s.numerator * inv_den) % prime, 1)
                else:
                    s = Fraction(int(s) % prime, 1)
            result[i][j] = s
    return result


def _parsed_differentials(
    cx: ChainComplexValue,
    prime: int | None = None,
) -> list[list[list[Fraction]]]:
    """Parse a complex's differentials into exact fraction matrices."""
    return [
        _matrix_to_fractions(m, cx.basis_sizes[i], cx.basis_sizes[i + 1], prime)
        for i, m in enumerate(cx.differential_matrices)
    ]


def _require_square_zero(
    diffs: list[list[list[Fraction]]],
    prime: int | None = None,
    *,
    label: str,
    group_columns: list[int] | None = None,
    degree_min: int = 0,
) -> None:
    """Require d^2 = 0 for a parsed differential sequence.

    ``group_columns`` carries chain-group dimensions so zero-width groups
    preserve outer product dimensions; diagnostics report the declared
    chain degree of the composed pair (``degree_min + index + 1``, matching
    the verification operation's verdict), not the tuple index.
    """
    for i in range(len(diffs) - 1):
        result_columns = group_columns[i + 2] if group_columns is not None else None
        left_declared_columns = (
            group_columns[i + 1] if group_columns is not None else None
        )
        product = _matrix_multiply(
            diffs[i],
            diffs[i + 1],
            prime,
            result_columns,
            left_declared_columns=left_declared_columns,
        )
        if any(value != 0 for row in product for value in row):
            raise ValueError(
                f"{label} complex violates d^2=0 at chain degree {degree_min + i + 1}"
            )


def _require_chain_map_relation(
    source_diffs: list[list[list[Fraction]]],
    target_diffs: list[list[list[Fraction]]],
    map_mats: list[list[list[Fraction]]],
    prime: int | None = None,
    source_group_columns: list[int] | None = None,
    target_group_columns: list[int] | None = None,
) -> None:
    """Require d_target * f_{i+1} == f_i * d_source at every differential.

    Declared group widths keep empty operands' inner dimensions intact:
    ``target_diffs[i]`` spans ``target_group_columns[i + 1]`` columns and
    ``map_mats[i]`` spans ``source_group_columns[i]``, so a zero-row
    differential or component still carries its mathematical width.
    """
    for i in range(len(source_diffs)):
        result_columns = (
            source_group_columns[i + 1] if source_group_columns is not None else None
        )
        left_declared_columns = (
            target_group_columns[i + 1] if target_group_columns is not None else None
        )
        right_declared_columns = (
            source_group_columns[i] if source_group_columns is not None else None
        )
        left = _matrix_multiply(
            target_diffs[i],
            map_mats[i + 1],
            prime,
            result_columns,
            left_declared_columns=left_declared_columns,
        )
        right = _matrix_multiply(
            map_mats[i],
            source_diffs[i],
            prime,
            result_columns,
            left_declared_columns=right_declared_columns,
        )
        if left != right:
            raise ValueError(
                f"chain map does not commute with differentials at degree index {i}"
            )


def _differential_verdict(complex_value: ChainComplexValue) -> tuple[bool, str]:
    """Decide d^2 = 0 exactly and derive the authoritative detail string."""
    cx = complex_value
    prime = cx.prime
    diffs = [
        _matrix_to_fractions(m, cx.basis_sizes[i], cx.basis_sizes[i + 1], prime)
        for i, m in enumerate(cx.differential_matrices)
    ]

    for i in range(len(diffs) - 1):
        # Correct order: d_i * d_{i+1} where diffs[i] is C_i x C_{i+1}, diffs[i+1] is C_{i+1} x C_{i+2}
        d_i = diffs[i]
        d_ip1 = diffs[i + 1]
        product = _matrix_multiply(
            d_i,
            d_ip1,
            prime,
            cx.basis_sizes[i + 2],
            left_declared_columns=cx.basis_sizes[i + 1],
        )
        is_zero = all(all(val == 0 for val in row) for row in product)
        if not is_zero:
            return False, f"d^2 != 0 at degree {cx.degree_min + i + 1}"

    return True, "d^2 = 0 for all degrees"


def _chain_map_verdict(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> tuple[bool, str]:
    """Decide the chain-map relation exactly and derive its detail string.

    Endpoints violating d^2 = 0 make the verification false by definition;
    the relation itself must commute at every differential degree.
    """
    prime = source.prime

    for label, complex_value in (("source", source), ("target", target)):
        if len(complex_value.differential_matrices) > 1:
            try:
                _require_square_zero(
                    _parsed_differentials(complex_value, prime),
                    prime,
                    label=label,
                    group_columns=list(complex_value.basis_sizes),
                    degree_min=complex_value.degree_min,
                )
            except ValueError as error:
                return (
                    False,
                    str(error) + ", so no chain map between these endpoints exists",
                )

    map_mats = [
        _matrix_to_fractions(m, target.basis_sizes[i], source.basis_sizes[i], prime)
        for i, m in enumerate(map_matrices)
    ]
    source_diffs = [
        _matrix_to_fractions(m, source.basis_sizes[i], source.basis_sizes[i + 1], prime)
        for i, m in enumerate(source.differential_matrices)
    ]
    target_diffs = [
        _matrix_to_fractions(m, target.basis_sizes[i], target.basis_sizes[i + 1], prime)
        for i, m in enumerate(target.differential_matrices)
    ]

    for i in range(len(source_diffs)):
        result_columns = source.basis_sizes[i + 1]
        left = _matrix_multiply(
            target_diffs[i],
            map_mats[i + 1],
            prime,
            result_columns,
            left_declared_columns=target.basis_sizes[i + 1],
        )
        right = _matrix_multiply(
            map_mats[i],
            source_diffs[i],
            prime,
            result_columns,
            left_declared_columns=source.basis_sizes[i],
        )
        if left != right:
            return (
                False,
                f"chain map does not commute at degree {source.degree_min + i}",
            )

    return True, "chain map commutes with differentials"


def _compute_homology_groups(
    cx: ChainComplexValue,
) -> tuple[HomologyGroupValue, ...]:
    """Exact homology groups shared by the operation and its validator."""

    prime = cx.prime
    n = len(cx.basis_sizes)

    diffs = [
        _matrix_to_fractions(m, cx.basis_sizes[i], cx.basis_sizes[i + 1], prime)
        for i, m in enumerate(cx.differential_matrices)
    ]

    # Require square-zero before computing homology; declared group
    # widths keep zero-row differentials shape-faithful.
    _require_square_zero(
        diffs,
        prime,
        label="chain complex",
        group_columns=list(cx.basis_sizes),
        degree_min=cx.degree_min,
    )

    groups = []
    for idx in range(n):
        chain_rank = cx.basis_sizes[idx]
        actual_degree = cx.degree_min + idx

        # Correct assignment: diffs[idx-1] is outgoing from C_{idx} -> C_{idx-1} (if idx>0), diffs[idx] is incoming from C_{idx+1} -> C_{idx}
        if idx > 0:
            d_out = diffs[idx - 1]
            outgoing_rank = _matrix_rank(d_out, prime)
        else:
            outgoing_rank = 0

        if idx < len(diffs):
            d_in = diffs[idx]
            incoming_rank = _matrix_rank(d_in, prime)
        else:
            incoming_rank = 0

        cycle_rank = chain_rank - outgoing_rank
        betti = cycle_rank - incoming_rank
        if betti < 0:
            raise ValueError(
                f"negative Betti number at degree {actual_degree}: indicates invalid complex"
            )

        groups.append(
            HomologyGroupValue(
                degree=actual_degree,
                cycle_rank=cycle_rank,
                boundary_rank=incoming_rank,
                betti_number=betti,
            )
        )

    return tuple(groups)


def _serialize_entry(value: Fraction, prime: int | None) -> str:
    """One canonical matrix-entry spelling; GF(p) entries are residues.

    Prime-field coefficients reduce modulo the modulus so an accepted
    GF(p) request serializes canonical residues in ``[0, p)`` instead of
    signed representatives that downstream values would reject.
    """
    if value.denominator != 1:
        if prime is not None:
            raise ValueError("prime-field coefficients must be integers")
        return (
            f"{format_canonical_integer(value.numerator)}/"
            f"{format_canonical_integer(value.denominator)}"
        )
    coefficient = int(value)
    if prime is not None:
        return format_canonical_integer(coefficient % prime)
    return format_canonical_integer(coefficient)


def _serialized_matrix(
    mat: list[list[Fraction]], prime: int | None
) -> tuple[tuple[str, ...], ...]:
    """Canonical string form of one exact derived matrix."""
    return tuple(tuple(_serialize_entry(v, prime) for v in row) for row in mat)


def _require_mapping_cone_parents(
    source: ChainComplexValue,
    target: ChainComplexValue,
) -> None:
    """A cone is defined only over one coefficient field on one interval."""
    if (
        source.coefficient_field != target.coefficient_field
        or source.prime != target.prime
    ):
        raise ValueError("mapping cone requires same coefficient field and prime")
    if (source.degree_min, source.degree_max) != (
        target.degree_min,
        target.degree_max,
    ):
        raise ValueError(
            "mapping cone requires source and target concentrated on "
            "the same degree interval"
        )


def _cone_group_sizes(
    source: ChainComplexValue, target: ChainComplexValue
) -> tuple[int, ...]:
    """Cone groups are cone_n = C_{n-1} ⊕ D_n."""
    max_len = max(len(source.basis_sizes), len(target.basis_sizes))
    sizes = []
    for i in range(max_len + 1):
        d_size = target.basis_sizes[i] if i < len(target.basis_sizes) else 0
        c_size2 = (
            source.basis_sizes[i - 1]
            if i > 0 and i - 1 < len(source.basis_sizes)
            else 0
        )
        sizes.append(c_size2 + d_size)
    return tuple(sizes)


def _cone_block_dimensions(
    n: int,
    source_sizes: tuple[int, ...],
    target_sizes: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Sizes (c_{n-2}, c_{n-1}, d_{n-1}, d_n) around cone degree n."""
    c_n_minus2 = source_sizes[n - 2] if n - 2 >= 0 and n - 2 < len(source_sizes) else 0
    c_n_minus1 = source_sizes[n - 1] if 0 <= n - 1 < len(source_sizes) else 0
    d_n = target_sizes[n] if n < len(target_sizes) else 0
    d_n_minus1 = target_sizes[n - 1] if n - 1 < len(target_sizes) else 0
    return c_n_minus2, c_n_minus1, d_n_minus1, d_n


def _fill_source_block(
    block: list[list[Fraction]],
    source_diffs: list[list[list[Fraction]]],
    n: int,
    c_n_minus2: int,
    c_n_minus1: int,
) -> None:
    """Top-left block carries -d_C^{n-1}: C_{n-1} -> C_{n-2}."""
    if not (c_n_minus2 and c_n_minus1 and n - 2 < len(source_diffs)):
        return
    d_c = source_diffs[n - 2]
    for i in range(c_n_minus2):
        for j in range(c_n_minus1):
            block[i][j] = -d_c[i][j]


def _fill_target_block(
    block: list[list[Fraction]],
    target_diffs: list[list[list[Fraction]]],
    n: int,
    c_n_minus2: int,
    c_n_minus1: int,
    d_n_minus1: int,
    d_n: int,
) -> None:
    """Bottom-right block carries d_D^{n}: D_n -> D_{n-1}."""
    if not (d_n_minus1 and d_n and n - 1 < len(target_diffs)):
        return
    d_d = target_diffs[n - 1]
    for i in range(d_n_minus1):
        for j in range(d_n):
            block[c_n_minus2 + i][c_n_minus1 + j] = d_d[i][j]


def _fill_map_block(
    block: list[list[Fraction]],
    map_mats: list[list[list[Fraction]]],
    n: int,
    c_n_minus2: int,
    c_n_minus1: int,
    d_n_minus1: int,
) -> None:
    """Bottom-left block carries f_{n-1}: C_{n-1} -> D_{n-1}."""
    if not (d_n_minus1 and c_n_minus1 and n - 1 < len(map_mats)):
        return
    f = map_mats[n - 1]
    for i in range(d_n_minus1):
        for j in range(c_n_minus1):
            if i < len(f) and j < len(f[0]):
                block[c_n_minus2 + i][j] = f[i][j]


def _cone_differential_for_degree(
    n: int,
    cone_basis_sizes: tuple[int, ...],
    source_basis_sizes: tuple[int, ...],
    target_basis_sizes: tuple[int, ...],
    source_diffs: list[list[list[Fraction]]],
    target_diffs: list[list[list[Fraction]]],
    map_mats: list[list[list[Fraction]]],
    prime: int | None,
) -> tuple[tuple[str, ...], ...]:
    """One cone differential cone_n -> cone_{n-1} as canonical strings.

    Blocks: top-left = -d_C^{n-1}, top-right = 0, bottom-left = f_{n-1},
    bottom-right = d_D^{n}.
    """
    rows = cone_basis_sizes[n - 1]
    cols = cone_basis_sizes[n]
    if rows == 0 or cols == 0:
        # Zero-cell differentials keep their declared row count so the
        # outer matrix shape stays reconstructible.
        return tuple(() for _ in range(rows))
    block = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    c_n_minus2, c_n_minus1, d_n_minus1, d_n = _cone_block_dimensions(
        n, source_basis_sizes, target_basis_sizes
    )

    _fill_source_block(block, source_diffs, n, c_n_minus2, c_n_minus1)
    _fill_target_block(block, target_diffs, n, c_n_minus2, c_n_minus1, d_n_minus1, d_n)
    _fill_map_block(block, map_mats, n, c_n_minus2, c_n_minus1, d_n_minus1)
    return _serialized_matrix(block, prime)


def _compute_mapping_cone(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> tuple[tuple[int, ...], tuple[tuple[tuple[str, ...], ...], ...]]:
    """Exact mapping-cone construction shared by the operation and its
    result validator."""
    _require_mapping_cone_parents(source, target)
    prime = source.prime

    # Compute cone basis sizes: cone_n = C_{n-1} ⊕ D_n.
    cone_basis_sizes = _cone_group_sizes(source, target)

    # Parse source and target differentials plus the map components.
    # map matrices: f_i: C_i -> D_i, shape target_basis[i] x source_basis[i].
    source_diffs = [
        _matrix_to_fractions(m, source.basis_sizes[i], source.basis_sizes[i + 1], prime)
        for i, m in enumerate(source.differential_matrices)
    ]
    target_diffs = [
        _matrix_to_fractions(m, target.basis_sizes[i], target.basis_sizes[i + 1], prime)
        for i, m in enumerate(target.differential_matrices)
    ]
    map_mats = [
        _matrix_to_fractions(m, target.basis_sizes[i], source.basis_sizes[i], prime)
        for i, m in enumerate(map_matrices)
    ]

    # A mapping cone is a chain complex only when both inputs are chain
    # complexes and the map is a genuine chain map; validate both defining
    # equations before returning any exact decomposition.
    _require_square_zero(
        source_diffs,
        prime,
        label="source",
        group_columns=list(source.basis_sizes),
        degree_min=source.degree_min,
    )
    _require_square_zero(
        target_diffs,
        prime,
        label="target",
        group_columns=list(target.basis_sizes),
        degree_min=target.degree_min,
    )
    _require_chain_map_relation(
        source_diffs,
        target_diffs,
        map_mats,
        prime,
        source_group_columns=list(source.basis_sizes),
        target_group_columns=list(target.basis_sizes),
    )

    cone_diffs = tuple(
        _cone_differential_for_degree(
            n,
            cone_basis_sizes,
            source.basis_sizes,
            target.basis_sizes,
            source_diffs,
            target_diffs,
            map_mats,
            prime,
        )
        for n in range(1, len(cone_basis_sizes))
    )

    # Defining invariant of the returned decomposition: the cone
    # differentials must themselves be square-zero. Declared cone group
    # widths keep zero-cell groups shape-faithful.
    cone_parsed = [
        _matrix_to_fractions(
            cone_diffs[i], cone_basis_sizes[i], cone_basis_sizes[i + 1], prime
        )
        for i in range(len(cone_diffs))
    ]
    _require_square_zero(
        cone_parsed,
        prime,
        label="mapping cone",
        group_columns=list(cone_basis_sizes),
        degree_min=source.degree_min,
    )

    return cone_basis_sizes, cone_diffs


def _tensor_group_sizes(
    left: ChainComplexValue, right: ChainComplexValue
) -> tuple[int, ...]:
    """Tensor groups are the graded sums (C⊗D)_n = ⊕_{i+j=n} C_i ⊗ D_j."""
    group_count = len(left.basis_sizes) + len(right.basis_sizes) - 1
    sizes = []
    for degree in range(group_count):
        size = 0
        for i in range(min(degree + 1, len(left.basis_sizes))):
            j = degree - i
            if j < len(right.basis_sizes):
                size += left.basis_sizes[i] * right.basis_sizes[j]
        sizes.append(size)
    return tuple(sizes)


def _tensor_group_offsets(
    degree: int,
    left_sizes: tuple[int, ...],
    right_sizes: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    """Column offsets of every (i, j) summand contributing to one degree."""
    offsets: dict[tuple[int, int], int] = {}
    off = 0
    for i in range(min(degree + 1, len(left_sizes))):
        j = degree - i
        if j < len(right_sizes) and j >= 0:
            offsets[(i, j)] = off
            off += left_sizes[i] * right_sizes[j]
    return offsets


def _apply_left_factor_differential(
    block: list[list[Fraction]],
    *,
    i: int,
    col_off: int,
    row_off: int,
    left: ChainComplexValue,
    left_diffs: list[list[list[Fraction]]],
    d_j_size: int,
) -> None:
    """d_C ⊗ id_D contribution from C_i ⊗ D_j down to C_{i-1} ⊗ D_j."""
    d_left = left_diffs[i - 1]
    for a in range(left.basis_sizes[i - 1]):
        for b in range(left.basis_sizes[i]):
            coeff = d_left[a][b]
            if coeff == 0:
                continue
            for c in range(d_j_size):
                row = row_off + a * d_j_size + c
                col = col_off + b * d_j_size + c
                block[row][col] += coeff


def _apply_right_factor_differential(
    block: list[list[Fraction]],
    *,
    i: int,
    j: int,
    col_off: int,
    row_off: int,
    left: ChainComplexValue,
    right: ChainComplexValue,
    right_diffs: list[list[list[Fraction]]],
) -> None:
    """(-1)^i id_C ⊗ d_D contribution from C_i ⊗ D_j down to C_i ⊗ D_{j-1}.

    The sign uses ``i`` as the actual chain degree of the left factor,
    not its tuple index.
    """
    d_right = right_diffs[j - 1]
    sign = -1 if (left.degree_min + i) % 2 == 1 else 1
    for a in range(left.basis_sizes[i]):
        for c2 in range(right.basis_sizes[j - 1]):
            for d in range(right.basis_sizes[j]):
                coeff = d_right[c2][d] * sign
                if coeff == 0:
                    continue
                row = row_off + a * right.basis_sizes[j - 1] + c2
                col = col_off + a * right.basis_sizes[j] + d
                block[row][col] += coeff


def _tensor_differential_for_degree(
    deg: int,
    group_sizes: tuple[int, ...],
    left: ChainComplexValue,
    right: ChainComplexValue,
    left_diffs: list[list[list[Fraction]]],
    right_diffs: list[list[list[Fraction]]],
    prime: int | None,
) -> tuple[tuple[str, ...], ...]:
    """One tensor differential (C⊗D)_deg -> (C⊗D)_{deg-1} as strings.

    Each summand C_i ⊗ D_j contributes via d_C down to C_{i-1} ⊗ D_j and
    via the Koszul-signed id ⊗ d_D down to C_i ⊗ D_{j-1}.
    """
    rows = group_sizes[deg - 1]
    cols = group_sizes[deg]
    if rows == 0 or cols == 0:
        # Zero-cell differentials keep their declared row count so the
        # outer matrix shape stays reconstructible.
        return tuple(() for _ in range(rows))
    block = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    offsets_deg = _tensor_group_offsets(deg, left.basis_sizes, right.basis_sizes)
    offsets_prev = _tensor_group_offsets(deg - 1, left.basis_sizes, right.basis_sizes)
    for (i, j), col_off in offsets_deg.items():
        if i > 0 and (i - 1, j) in offsets_prev:
            _apply_left_factor_differential(
                block,
                i=i,
                col_off=col_off,
                row_off=offsets_prev[(i - 1, j)],
                left=left,
                left_diffs=left_diffs,
                d_j_size=right.basis_sizes[j],
            )
        if j > 0 and (i, j - 1) in offsets_prev:
            _apply_right_factor_differential(
                block,
                i=i,
                j=j,
                col_off=col_off,
                row_off=offsets_prev[(i, j - 1)],
                left=left,
                right=right,
                right_diffs=right_diffs,
            )
    return _serialized_matrix(block, prime)


def _compute_tensor_product(
    left: ChainComplexValue,
    right: ChainComplexValue,
) -> tuple[tuple[int, ...], tuple[tuple[tuple[str, ...], ...], ...]]:
    """Exact tensor-product construction shared by the operation and its
    result validator."""
    if left.coefficient_field != right.coefficient_field or left.prime != right.prime:
        raise ValueError("tensor product requires same coefficient field and prime")
    prime = left.prime

    tensor_basis_sizes = _tensor_group_sizes(left, right)

    left_diffs = [
        _matrix_to_fractions(m, left.basis_sizes[i], left.basis_sizes[i + 1], prime)
        for i, m in enumerate(left.differential_matrices)
    ]
    right_diffs = [
        _matrix_to_fractions(m, right.basis_sizes[i], right.basis_sizes[i + 1], prime)
        for i, m in enumerate(right.differential_matrices)
    ]

    # A tensor product of chain complexes is a chain complex only when both
    # factors satisfy d^2 = 0; validate before building anything.
    _require_square_zero(
        left_diffs,
        prime,
        label="left",
        group_columns=list(left.basis_sizes),
        degree_min=left.degree_min,
    )
    _require_square_zero(
        right_diffs,
        prime,
        label="right",
        group_columns=list(right.basis_sizes),
        degree_min=right.degree_min,
    )

    tensor_diffs = tuple(
        _tensor_differential_for_degree(
            deg, tensor_basis_sizes, left, right, left_diffs, right_diffs, prime
        )
        for deg in range(1, len(tensor_basis_sizes))
    )

    return tensor_basis_sizes, tensor_diffs


def construct_chain_complex(
    basis_sizes: tuple[int, ...],
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...],
    *,
    coefficient_field: CoefficientField = CoefficientField.RATIONAL,
    prime: int | None = None,
) -> ChainComplexValue:
    """Construct an admitted canonical chain-complex value.

    Differential entries use the canonical exact string grammar carried by
    :class:`ChainComplexValue`; adjacent differentials must compose to zero.
    """
    value = ChainComplexValue(
        coefficient_field=coefficient_field,
        prime=prime,
        degree_min=0,
        degree_max=len(basis_sizes) - 1,
        basis_sizes=basis_sizes,
        differential_matrices=differential_matrices,
    )
    _require_square_zero(
        _parsed_differentials(value, value.prime),
        value.prime,
        label="constructed",
        group_columns=list(value.basis_sizes),
        degree_min=value.degree_min,
    )
    return value


def homology_groups(complex_value: ChainComplexValue) -> HomologyResult:
    """Return exact homology groups for a canonical chain complex value."""
    groups = _compute_homology_groups(complex_value)
    return HomologyResult._from_kernel(
        homology_groups=tuple(groups),
        source_complex=complex_value,
    )


def differential_squares_to_zero(
    complex_value: ChainComplexValue,
) -> VerificationResult:
    """Verify d^2 = 0 for one canonical chain-complex value."""
    is_valid, detail = _differential_verdict(complex_value)
    return VerificationResult(is_valid=is_valid, detail=detail, complex=complex_value)


def chain_map_commutes(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: MapMatrices,
) -> VerificationResult:
    """Verify that a component-wise chain map commutes with differentials."""
    _require_chain_map_components(
        source, target, map_matrices, label="chain-map verification"
    )
    is_valid, detail = _chain_map_verdict(source, target, map_matrices)
    return VerificationResult._from_chain_map_kernel(
        is_valid=is_valid,
        detail=detail,
        source=source,
        target=target,
        map_matrices=map_matrices,
    )


def mapping_cone(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: MapMatrices,
) -> MappingConeResult:
    """Compute the mapping cone of a chain-map value."""
    _require_chain_map_components(source, target, map_matrices, label="mapping cone")
    _require_cone_admission(source, target, map_matrices)
    cone_basis_sizes, cone_diffs = _compute_mapping_cone(source, target, map_matrices)
    value = ChainComplexValue(
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        degree_min=source.degree_min,
        degree_max=source.degree_min + len(cone_basis_sizes) - 1,
        basis_sizes=cone_basis_sizes,
        differential_matrices=cone_diffs,
    )
    return MappingConeResult._from_kernel(
        cone_basis_sizes=cone_basis_sizes,
        cone_differential_matrices=cone_diffs,
        source=source,
        target=target,
        map_matrices=map_matrices,
        value=value,
    )


def tensor_product_complex(
    left: ChainComplexValue,
    right: ChainComplexValue,
) -> TensorProductResult:
    """Compute the tensor product of two canonical chain-complex values."""
    _require_tensor_admission(left, right)
    tensor_basis_sizes, tensor_diffs = _compute_tensor_product(left, right)
    group_count = len(tensor_basis_sizes)
    degree_min = left.degree_min + right.degree_min
    degree_max = degree_min + group_count - 1
    value = ChainComplexValue(
        coefficient_field=left.coefficient_field,
        prime=left.prime,
        degree_min=degree_min,
        degree_max=degree_max,
        basis_sizes=tensor_basis_sizes,
        differential_matrices=tensor_diffs,
    )
    return TensorProductResult._from_kernel(
        tensor_basis_sizes=tensor_basis_sizes,
        tensor_differential_matrices=tensor_diffs,
        left=left,
        right=right,
        value=value,
    )
