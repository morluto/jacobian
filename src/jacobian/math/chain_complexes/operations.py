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
    ChainComplexValue,
    HomologyGroupValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
    VerificationResult,
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


def _matrix_rank(matrix: list[list[Fraction]], prime: int | None = None) -> int:
    """Compute rank of a matrix over QQ or GF(p)."""
    if not matrix or not matrix[0]:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])
    if prime is not None:
        # Perform elimination inside GF(p) using modular inverses
        # Convert Fractions to ints modulo p (they are already residues)
        mat = [[int(v) % prime for v in row] for row in matrix]
        rank = 0
        for col in range(cols):
            # Find pivot with non-zero residue
            pivot = None
            for row in range(rank, rows):
                if mat[row][col] % prime != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            inv = pow(int(mat[rank][col]), -1, prime)  # modular inverse
            # Normalize pivot row
            for j in range(cols):
                mat[rank][j] = (mat[rank][j] * inv) % prime
            # Eliminate other rows
            for row in range(rows):
                if row == rank:
                    continue
                factor = mat[row][col] % prime
                if factor != 0:
                    for j in range(cols):
                        mat[row][j] = (mat[row][j] - factor * mat[rank][j]) % prime
            rank += 1
            if rank == rows:
                break
        return rank
    else:
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
            rank += 1
            if rank == rows:
                break
        return rank


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
    group_columns=None,
    degree_min: int = 0,
) -> None:
    """Require d^2 = 0 for a parsed differential sequence.

    ``group_columns`` carries chain-group dimensions so zero-width groups
    preserve outer product dimensions; diagnostics report the declared
    chain degree (``degree_min + index``), not the tuple index.
    """
    for i in range(len(diffs) - 1):
        result_columns = group_columns[i + 2] if group_columns is not None else None
        left_declared_columns = group_columns[i + 1] if group_columns is not None else None
        product = _matrix_multiply(
            diffs[i],
            diffs[i + 1],
            prime,
            result_columns,
            left_declared_columns=left_declared_columns,
        )
        if any(value != 0 for row in product for value in row):
            raise ValueError(
                f"{label} complex violates d^2=0 at chain degree {degree_min + i}"
            )


def _require_chain_map_relation(
    source_diffs: list[list[list[Fraction]]],
    target_diffs: list[list[list[Fraction]]],
    map_mats: list[list[list[Fraction]]],
    prime: int | None = None,
    source_group_columns=None,
) -> None:
    """Require d_target * f_{i+1} == f_i * d_source at every differential."""
    for i in range(len(source_diffs)):
        result_columns = (
            source_group_columns[i + 1] if source_group_columns is not None else None
        )
        left = _matrix_multiply(target_diffs[i], map_mats[i + 1], prime, result_columns)
        right = _matrix_multiply(map_mats[i], source_diffs[i], prime, result_columns)
        if left != right:
            raise ValueError(
                f"chain map does not commute with differentials at degree index {i}"
            )


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
            return VerificationResult(
                is_valid=False,
                detail=f"d^2 != 0 at degree {cx.degree_min + i + 1}",
                complex=request.complex,
            )

    return VerificationResult(
        is_valid=True,
        detail="d^2 = 0 for all degrees",
        complex=request.complex,
    )


def verify_chain_map(request: VerifyChainMapRequest) -> VerificationResult:
    """Verify that a chain map commutes with differentials.

    Returns a dictionary with 'is_valid' (bool) and 'detail' (str).

    Request admission guarantees equal coefficient fields and prime, so a
    field mismatch can never reach this kernel as a verdict.
    """
    source = request.source
    target = request.target
    prime = source.prime

    # A chain map exists only between chain complexes; endpoints violating
    # d^2=0 make the verification false by definition. Check every
    # consecutive differential product of each endpoint.
    for label, complex_value in (("source", source), ("target", target)):
        if len(complex_value.differential_matrices) > 1:
            try:
                _require_square_zero(
                    _parsed_differentials(complex_value, prime),
                    prime,
                    label=label,
                    group_columns=list(complex_value.basis_sizes),
                )
            except ValueError as error:
                return VerificationResult(
                    is_valid=False,
                    detail=str(error) + ", so no chain map between these "
                    "endpoints exists",
                    source=source,
                    target=target,
                    map_matrices=request.map_matrices,
                )

    # Request admission guarantees one component per degree, equal shape
    # (target rows x source columns) per component, and coinciding degree
    # intervals, so tuple index equals actual chain degree.
    map_mats = [
        _matrix_to_fractions(m, target.basis_sizes[i], source.basis_sizes[i], prime)
        for i, m in enumerate(request.map_matrices)
    ]
    source_diffs = [
        _matrix_to_fractions(m, source.basis_sizes[i], source.basis_sizes[i + 1], prime)
        for i, m in enumerate(source.differential_matrices)
    ]
    target_diffs = [
        _matrix_to_fractions(m, target.basis_sizes[i], target.basis_sizes[i + 1], prime)
        for i, m in enumerate(target.differential_matrices)
    ]

    # f_i: C_i -> D_i has shape target_basis[i] x source_basis[i], so the
    # chain-map equation at every differential is
    # d_target_i * f_{i+1} == f_i * d_source_i.
    for i in range(len(source_diffs)):
        result_columns = source.basis_sizes[i + 1]
        left = _matrix_multiply(target_diffs[i], map_mats[i + 1], prime, result_columns)
        right = _matrix_multiply(map_mats[i], source_diffs[i], prime, result_columns)
        if left != right:
            return VerificationResult(
                is_valid=False,
                detail=f"chain map does not commute at degree {source.degree_min + i}",
                source=source,
                target=target,
                map_matrices=request.map_matrices,
            )

    return VerificationResult(
        is_valid=True,
        detail="chain map commutes with differentials",
        source=source,
        target=target,
        map_matrices=request.map_matrices,
    )


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

    # Require square-zero before computing homology
    for i in range(len(diffs) - 1):
        prod = _matrix_multiply(diffs[i], diffs[i + 1], prime, cx.basis_sizes[i + 2])
        if any(any(v != 0 for v in row) for row in prod):
            raise ValueError(
                f"chain complex does not satisfy d^2=0 at chain degree "
                f"{cx.degree_min + i}: product non-zero"
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


def compute_homology(request: ComputeHomologyRequest) -> HomologyResult:
    """Compute the homology groups of a chain complex."""
    cx = request.complex
    prime = cx.prime
    groups = _compute_homology_groups(cx)
    return HomologyResult(
        homology_groups=tuple(groups),
        coefficient_field=cx.coefficient_field,
        prime=prime,
        degree_min=cx.degree_min,
        degree_max=cx.degree_max,
        complex=cx,
    )


def _serialize_entry(value: Fraction, prime: int | None) -> str:
    """One canonical matrix-entry spelling; GF(p) entries are residues."""
    if prime is not None:
        # Canonical GF(p) residues in [0, p): signed contributions must not
        # serialize as negative integers.
        return str(int(value) % prime)
    if value.denominator == 1:
        return str(int(value))
    return f"{value.numerator}/{value.denominator}"


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


def _compute_mapping_cone(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> tuple[tuple[int, ...], tuple[tuple[tuple[str, ...], ...], ...]]:
    """Exact mapping-cone construction shared by the operation and its
    result validator."""
    _require_mapping_cone_parents(source, target)
    prime = source.prime

    # Verify degree alignment: source and target should have same degree_min for simplicity, else shift
    # Compute cone basis sizes: cone_n = C_{n-1} ⊕ D_n
    max_len = max(len(source.basis_sizes), len(target.basis_sizes))
    cone_basis_sizes = tuple(
        (target.basis_sizes[i] if i < len(target.basis_sizes) else 0)
        + (
            source.basis_sizes[i - 1]
            if i > 0 and i - 1 < len(source.basis_sizes)
            else 0
        )
        for i in range(max_len + 1)
    )

    # Build block differentials for each degree
    # For each cone differential cone_{n} -> cone_{n-1}, we need matrix of size cone_basis_sizes[n-1] x cone_basis_sizes[n]
    # Blocks: top-left = -d_C^{n-1}, top-right = 0, bottom-left = f_{n-1}, bottom-right = d_D^{n}
    # For simplicity, we construct string matrices with entries as fractions strings.
    # Need to parse source and target diffs and maps.

    source_diffs = [
        _matrix_to_fractions(m, source.basis_sizes[i], source.basis_sizes[i + 1], prime)
        for i, m in enumerate(source.differential_matrices)
    ]
    target_diffs = [
        _matrix_to_fractions(m, target.basis_sizes[i], target.basis_sizes[i + 1], prime)
        for i, m in enumerate(target.differential_matrices)
    ]
    # map matrices: f_i: C_i -> D_i, shape target_basis[i] x source_basis[i]
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
    )

    def _to_str_matrix(mat: list[list[Fraction]]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(_serialize_entry(v, prime) for v in row) for row in mat
        )

    cone_diffs: list[tuple[tuple[str, ...], ...]] = []
    for n in range(1, len(cone_basis_sizes)):
        rows = cone_basis_sizes[n - 1]
        cols = cone_basis_sizes[n]
        if rows == 0 or cols == 0:
            # Zero-cell differentials keep their declared row count so the
            # outer matrix shape stays reconstructible.
            cone_diffs.append(tuple(() for _ in range(rows)))
            continue
        # Build zero matrix
        block = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
        # Sizes for decomposition: cone_{n} = C_{n-1} ⊕ D_n, cone_{n-1}= C_{n-2} ⊕ D_{n-1}
        c_n_minus1 = (
            source.basis_sizes[n - 1] if 0 <= n - 1 < len(source.basis_sizes) else 0
        )
        d_n = target.basis_sizes[n] if n < len(target.basis_sizes) else 0
        c_n_minus2 = (
            source.basis_sizes[n - 2]
            if n - 2 >= 0 and n - 2 < len(source.basis_sizes)
            else 0
        )
        d_n_minus1 = target.basis_sizes[n - 1] if n - 1 < len(target.basis_sizes) else 0

        # Fill blocks
        # Top-left: -d_C^{n-1} : size c_{n-2} x c_{n-1}
        if c_n_minus2 and c_n_minus1 and n - 2 < len(source_diffs):
            d_c = source_diffs[n - 2]  # C_{n-1} -> C_{n-2}
            for i in range(c_n_minus2):
                for j in range(c_n_minus1):
                    block[i][j] = -d_c[i][j]
        # Bottom-right: d_D^{n}: size d_{n-1} x d_n
        if d_n_minus1 and d_n and n - 1 < len(target_diffs):
            d_d = target_diffs[n - 1]
            for i in range(d_n_minus1):
                for j in range(d_n):
                    block[c_n_minus2 + i][c_n_minus1 + j] = d_d[i][j]
        # Bottom-left: f_{n-1}: size d_{n-1} x c_{n-1}
        if d_n_minus1 and c_n_minus1 and n - 1 < len(map_mats):
            f = map_mats[n - 1]
            for i in range(d_n_minus1):
                for j in range(c_n_minus1):
                    # f is target x source, so need to map correctly
                    if i < len(f) and j < len(f[0]):
                        block[c_n_minus2 + i][j] = f[i][j]

        cone_diffs.append(_to_str_matrix(block))

    # Defining invariant of the returned decomposition: the cone
    # differentials must themselves be square-zero.
    cone_parsed = [
        _matrix_to_fractions(
            cone_diffs[i], cone_basis_sizes[i], cone_basis_sizes[i + 1], prime
        )
        for i in range(len(cone_diffs))
    ]
    _require_square_zero(cone_parsed, prime, label="mapping cone")

    return cone_basis_sizes, tuple(cone_diffs)


def compute_mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Compute the mapping cone of a chain map f: C -> D.

    The mapping cone has groups cone_n = C_{n-1} ⊕ D_n and differential
    [[ -d_C, 0 ], [ f, d_D ]] as block matrices.
    """
    cone_basis_sizes, cone_diffs = _compute_mapping_cone(
        request.source, request.target, request.map_matrices
    )

    return MappingConeResult(
        cone_basis_sizes=cone_basis_sizes,
        cone_differential_matrices=cone_diffs,
        source_degree_min=request.source.degree_min,
        target_degree_min=request.target.degree_min,
        source=request.source,
        target=request.target,
        map_matrices=request.map_matrices,
    )


def compute_tensor_product(request: TensorProductRequest) -> TensorProductResult:
    """Compute the tensor product of two chain complexes.

    (C ⊗ D)_n = ⊕_{i+j=n} C_i ⊗ D_j with differential d_C ⊗ id + (-1)^i id ⊗ d_D.
    """
    left = request.left
    right = request.right
    if left.coefficient_field != right.coefficient_field or left.prime != right.prime:
        raise ValueError("tensor product requires same coefficient field and prime")
    prime = left.prime

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
    tensor_basis_sizes = tuple(tensor_basis_sizes)

    # Build signed differentials for each degree.
    # For each n, differential d_n: (C⊗D)_n -> (C⊗D)_{n-1} is block matrix with
    # contributions from d^C and d^D with signs.
    # We construct rational matrices as Fractions and convert to strings.

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

    def _to_str_matrix(mat: list[list[Fraction]]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(_serialize_entry(v, prime) for v in row) for row in mat
        )

    tensor_diffs: list[tuple[tuple[str, ...], ...]] = []
    for deg in range(1, n):
        rows = tensor_basis_sizes[deg - 1]
        cols = tensor_basis_sizes[deg]
        if rows == 0 or cols == 0:
            # Zero-cell differentials keep their declared row count so the
            # outer matrix shape stays reconstructible.
            tensor_diffs.append(tuple(() for _ in range(rows)))
            continue
        block = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
        # We need to map block structure: (C⊗D)_deg = ⊕_{i+j=deg} C_i⊗D_j
        # Similarly for deg-1 = ⊕_{p+q=deg-1} C_p⊗D_q
        # For each i,j contributing to deg, its differential contributes to two possible targets:
        # - via d_C: C_i⊗D_j -> C_{i-1}⊗D_j (if i>0)
        # - via d_D: C_i⊗D_j -> C_i⊗D_{j-1} with sign (-1)^i
        # We need to compute offset positions for each summand.
        # First compute offsets for deg and deg-1
        offsets_deg = {}
        off = 0
        for i in range(min(deg + 1, len(left.basis_sizes))):
            j = deg - i
            if j < len(right.basis_sizes):
                offsets_deg[(i, j)] = off
                off += left.basis_sizes[i] * right.basis_sizes[j]
        offsets_deg_minus1 = {}
        off = 0
        for p in range(min(deg, len(left.basis_sizes))):
            q = deg - 1 - p
            if q < len(right.basis_sizes) and q >= 0:
                offsets_deg_minus1[(p, q)] = off
                off += left.basis_sizes[p] * right.basis_sizes[q]

        for (i, j), col_off in offsets_deg.items():
            # d_C contribution
            if i > 0 and (i - 1, j) in offsets_deg_minus1:
                d_left = left_diffs[i - 1]  # shape left_basis[i-1] x left_basis[i]
                row_off = offsets_deg_minus1[(i - 1, j)]
                # Tensor product with identity on D_j
                for a in range(left.basis_sizes[i - 1]):
                    for b in range(left.basis_sizes[i]):
                        coeff = d_left[a][b]
                        if coeff == 0:
                            continue
                        for c in range(right.basis_sizes[j]):
                            row = row_off + a * right.basis_sizes[j] + c
                            col = col_off + b * right.basis_sizes[j] + c
                            block[row][col] += coeff
            # d_D contribution with sign (-1)^i where i is the actual chain
            # degree of the left factor, not its tuple index.
            if j > 0 and (i, j - 1) in offsets_deg_minus1:
                d_right = right_diffs[j - 1]
                left_degree = left.degree_min + i
                sign = -1 if left_degree % 2 == 1 else 1
                row_off = offsets_deg_minus1[(i, j - 1)]
                for a in range(left.basis_sizes[i]):
                    for c2 in range(right.basis_sizes[j - 1]):
                        for d in range(right.basis_sizes[j]):
                            coeff = d_right[c2][d] * sign
                            if coeff == 0:
                                continue
                            row = row_off + a * right.basis_sizes[j - 1] + c2
                            col = col_off + a * right.basis_sizes[j] + d
                            block[row][col] += coeff
        tensor_diffs.append(_to_str_matrix(block))

    # Tensor degrees are pairwise sums: the derived complex concentrates
    # on [deg_min, deg_min + group_count - 1].
    group_count = len(left.basis_sizes) + len(right.basis_sizes) - 1
    degree_min = left.degree_min + right.degree_min
    from jacobian.math.chain_complexes.values import ChainComplexValue

    return TensorProductResult(
        tensor_basis_sizes=tensor_basis_sizes,
        tensor_differential_matrices=tuple(tensor_diffs),
        coefficient_field=left.coefficient_field,
        prime=prime,
        degree_min=degree_min,
        degree_max=degree_min + group_count - 1,
        value=ChainComplexValue(
            coefficient_field=left.coefficient_field,
            prime=prime,
            degree_min=degree_min,
            degree_max=degree_min + group_count - 1,
            basis_sizes=tensor_basis_sizes,
            differential_matrices=tuple(tensor_diffs),
        ),
    )
