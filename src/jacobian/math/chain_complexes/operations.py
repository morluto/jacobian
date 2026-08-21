"""Exact finite based chain complex operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.chain_complexes._models import (
    ComputeHomologyRequest,
    ConstructChainComplexRequest,
    MappingConeRequest,
    TensorProductRequest,
    VerifyChainMapRequest,
    VerifyDifferentialRequest,
)
from jacobian.math.chain_complexes.values import (
    VerificationResult,
    ChainComplexValue,
    CoefficientField,
    HomologyGroupValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
)


def _parse_fraction(s: str, prime: int | None = None) -> Fraction:
    if prime is not None:
        return Fraction(int(s) % prime)
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


def _matrix_rank(matrix: list[list[Fraction]], prime: int | None = None) -> int:
    """Compute rank of a matrix over QQ or GF(p)."""
    if not matrix or not matrix[0]:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])
    mat = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for row in range(rows):
            if row == rank:
                continue
            if mat[row][col] != 0:
                factor = mat[row][col] / mat[rank][col]
                for j in range(cols):
                    mat[row][j] -= factor * mat[rank][j]
                if prime is not None:
                    for j in range(cols):
                        mat[row][j] = mat[row][j] % prime
        rank += 1
        if rank == rows:
            break
    return rank


def _matrix_multiply(
    a: list[list[Fraction]],
    b: list[list[Fraction]],
) -> list[list[Fraction]]:
    rows_a = len(a)
    cols_a = len(a[0]) if a else 0
    cols_b = len(b[0]) if b else 0
    result = [[Fraction(0)] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def construct_chain_complex(request: ConstructChainComplexRequest) -> ChainComplexValue:
    """Construct a chain complex from differential matrices."""
    basis_sizes = request.basis_sizes
    n = len(basis_sizes)
    degree_min = 0
    degree_max = n - 1

    return ChainComplexValue(
        coefficient_field=request.coefficient_field,
        prime=request.prime,
        degree_min=degree_min,
        degree_max=degree_max,
        basis_sizes=basis_sizes,
        differential_matrices=request.differential_matrices,
    )


def verify_differential(request: VerifyDifferentialRequest) -> VerificationResult:
    """Verify that d^2 = 0 for a chain complex.

    Returns a dictionary with 'is_valid' (bool) and 'detail' (str).
    """
    cx = request.complex
    prime = cx.prime
    diffs = [
        _matrix_to_fractions(m, cx.basis_sizes[i], cx.basis_sizes[i + 1], prime)
        for i, m in enumerate(cx.differential_matrices)
    ]

    for i in range(len(diffs) - 1):
        d_n = diffs[i + 1]
        d_n_minus_1 = diffs[i]
        product = _matrix_multiply(d_n, d_n_minus_1)
        is_zero = all(
            all(val == 0 for val in row) for row in product
        )
        if not is_zero:
            return VerificationResult(
                is_valid=False,
                detail= f"d^2 != 0 at degree {i + 1}",
            )

    return VerificationResult(is_valid=True, detail="d^2 = 0 for all degrees")


def verify_chain_map(request: VerifyChainMapRequest) -> VerificationResult:
    """Verify that a chain map commutes with differentials.

    Returns a dictionary with 'is_valid' (bool) and 'detail' (str).
    """
    source = request.source
    target = request.target
    prime = source.prime or target.prime

    source_diffs = [
        _matrix_to_fractions(m, source.basis_sizes[i], source.basis_sizes[i + 1], prime)
        for i, m in enumerate(source.differential_matrices)
    ]
    target_diffs = [
        _matrix_to_fractions(m, target.basis_sizes[i], target.basis_sizes[i + 1], prime)
        for i, m in enumerate(target.differential_matrices)
    ]
    map_mats = [
        _matrix_to_fractions(m, source.basis_sizes[i], target.basis_sizes[i], prime)
        for i, m in enumerate(request.map_matrices)
    ]

    for i in range(len(source_diffs)):
        d_source = source_diffs[i]
        d_target = target_diffs[i]
        f_n = map_mats[i]
        f_n_minus_1 = map_mats[i + 1] if i + 1 < len(map_mats) else None

        # d_target * f_n should equal f_{n-1} * d_source
        left = _matrix_multiply(f_n, d_target)  # f_n: C_n -> D_n, d_target: D_n -> D_{n-1}
        if f_n_minus_1 is not None:
            right = _matrix_multiply(d_source, f_n_minus_1)  # d_source: C_n -> C_{n-1}, f_{n-1}: C_{n-1} -> D_{n-1}
        else:
            right = [[Fraction(0)] * len(left[0]) for _ in range(len(left))]

        if left != right:
            return VerificationResult(
                is_valid=False,
                detail= f"chain map does not commute at degree {i}",
            )

    return VerificationResult(is_valid=True, detail="chain map commutes with differentials")


def compute_homology(request: ComputeHomologyRequest) -> HomologyResult:
    """Compute the homology groups of a chain complex."""
    cx = request.complex
    prime = cx.prime
    n = len(cx.basis_sizes)

    diffs = [
        _matrix_to_fractions(m, cx.basis_sizes[i], cx.basis_sizes[i + 1], prime)
        for i, m in enumerate(cx.differential_matrices)
    ]

    groups = []
    for degree in range(n):
        chain_rank = cx.basis_sizes[degree]

        # outgoing differential d_n: C_n -> C_{n-1}
        if degree < len(diffs):
            d_out = diffs[degree]
            outgoing_rank = _matrix_rank(d_out, prime)
        else:
            outgoing_rank = 0

        # incoming differential d_{n+1}: C_{n+1} -> C_n
        if degree > 0:
            d_in = diffs[degree - 1]
            incoming_rank = _matrix_rank(d_in, prime)
        else:
            incoming_rank = 0

        cycle_rank = chain_rank - outgoing_rank
        betti = cycle_rank - incoming_rank

        groups.append(
            HomologyGroupValue(
                degree=degree,
                cycle_rank=cycle_rank,
                boundary_rank=incoming_rank,
                betti_number=max(0, betti),
            )
        )

    return HomologyResult(
        homology_groups=tuple(groups),
        coefficient_field=cx.coefficient_field,
    )


def compute_mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Compute the mapping cone of a chain map f: C -> D.

    The mapping cone has groups cone_n = C_{n-1} ⊕ D_n.
    """
    source = request.source
    target = request.target
    prime = source.prime or target.prime

    n = max(len(source.basis_sizes), len(target.basis_sizes))
    cone_basis_sizes = tuple(
        source.basis_sizes[i - 1] + target.basis_sizes[i]
        if i > 0 and i - 1 < len(source.basis_sizes) and i < len(target.basis_sizes)
        else source.basis_sizes[i - 1] if i > 0 and i - 1 < len(source.basis_sizes)
        else target.basis_sizes[i] if i < len(target.basis_sizes)
        else 0
        for i in range(n + 1)
    )

    return MappingConeResult(
        cone_basis_sizes=cone_basis_sizes,
        cone_differential_matrices=(),  # Simplified: full cone differentials need block matrices
        source_degree_min=source.degree_min,
        target_degree_min=target.degree_min,
    )


def compute_tensor_product(request: TensorProductRequest) -> TensorProductResult:
    """Compute the tensor product of two chain complexes.

    (C ⊗ D)_n = ⊕_{i+j=n} C_i ⊗ D_j
    """
    left = request.left
    right = request.right
    n = len(left.basis_sizes) + len(right.basis_sizes) - 1

    # Basis sizes for the tensor product
    tensor_basis_sizes = []
    for degree in range(n):
        size = 0
        for i in range(min(degree + 1, len(left.basis_sizes))):
            j = degree - i
            if j < len(right.basis_sizes):
                size += left.basis_sizes[i] * right.basis_sizes[j]
        tensor_basis_sizes.append(size)

    return TensorProductResult(
        tensor_basis_sizes=tuple(tensor_basis_sizes),
        tensor_differential_matrices=(),  # Full differentials need block construction
    )
