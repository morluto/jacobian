"""Domain functions for linear code structural operations."""

from __future__ import annotations

from jacobian.math.code_linear._models import (
    CodeEqualRequest,
    CodeEqualResult,
    CodewordCheckRequest,
    CodewordCheckResult,
    DualCodeResult,
    FromGeneratorResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    ParityCheckRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ShortenRequest,
    ShortenResult,
    SyndromeRequest,
    SyndromeResult,
)


def _rref(matrix: list[list[int]], field_order: int) -> tuple[list[list[int]], int]:
    """Reduced row echelon form and rank over a prime field."""
    rows = [list(row) for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0
    pivot_row = 0
    for col in range(col_count):
        pivot = None
        for i in range(pivot_row, row_count):
            if rows[i][col] % field_order != 0:
                pivot = i
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inv = pow(rows[pivot_row][col] % field_order, -1, field_order)
        rows[pivot_row] = [v * inv % field_order for v in rows[pivot_row]]
        for i, row in enumerate(rows):
            if i == pivot_row:
                continue
            factor = row[col] % field_order
            if factor == 0:
                continue
            rows[i] = [
                (a - factor * b) % field_order
                for a, b in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return rows, pivot_row


def _nullspace(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    """Compute a basis for the nullspace of the matrix over a prime field."""
    rows, rank = _rref(matrix, field_order)
    n = len(matrix[0])
    piv_cols: list[int] = []
    for i in range(rank):
        piv = n
        for j in range(n):
            if rows[i][j] % field_order != 0:
                piv = j
                break
        piv_cols.append(piv)
    free_cols = [j for j in range(n) if j not in piv_cols]
    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * n
        vec[fc] = 1
        for i in range(rank):
            piv = piv_cols[i]
            vec[piv] = (-rows[i][fc]) % field_order
        basis.append(vec)
    return basis


def _canonical_generator(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    """Return canonical RREF rows for a matrix's row space."""
    rref, rank = _rref(matrix, field_order)
    return list(rref[:rank])


def _mat_mul_vec(
    matrix: list[list[int]], vec: list[int], field_order: int
) -> list[int]:
    return [
        sum(row[j] * vec[j] for j in range(len(vec))) % field_order for row in matrix
    ]


def _hamming_weight(word: list[int] | tuple[int, ...]) -> int:
    return sum(1 for v in word if v != 0)


def compute_from_generator(request: GeneratorMatrixRequest) -> FromGeneratorResult:
    matrix = [list(row) for row in request.generator_matrix]
    canonical = _canonical_generator(matrix, request.field_order)
    dim = len(canonical)
    length = len(request.generator_matrix[0])
    cardinality = request.field_order**dim
    return FromGeneratorResult(
        canonical_generator=tuple(tuple(row) for row in canonical),
        dimension=dim,
        length=length,
        cardinality=cardinality,
    )


def compute_dual_code(request: GeneratorMatrixRequest) -> DualCodeResult:
    matrix = [list(row) for row in request.generator_matrix]
    _, rank = _rref(matrix, request.field_order)
    null = _nullspace(matrix, request.field_order)
    length = len(request.generator_matrix[0])
    return DualCodeResult(
        dual_generator=tuple(tuple(row) for row in null),
        dimension=rank,
        dual_dimension=length - rank,
        length=length,
    )


def compute_parity_check(request: ParityCheckRequest) -> ParityCheckResult:
    matrix = [list(row) for row in request.generator_matrix]
    _, rank = _rref(matrix, request.field_order)
    null = _nullspace(matrix, request.field_order)
    length = len(request.generator_matrix[0])
    return ParityCheckResult(
        parity_check=tuple(tuple(row) for row in null),
        dimension=rank,
        rank_h=length - rank,
        length=length,
    )


def compute_codeword_check(
    request: CodewordCheckRequest,
) -> CodewordCheckResult:
    matrix = [list(row) for row in request.generator_matrix]
    word = list(request.word)
    q = request.field_order

    # Word is a codeword iff it lies in the row space of the generator.
    # Augment the generator with the word as a new row and check
    # whether rank increases.
    _, rank_g = _rref([list(row) for row in matrix], q)
    augmented = [list(row) for row in matrix] + [word]
    _, rank_aug = _rref(augmented, q)
    is_member = rank_aug == rank_g

    coefficients: tuple[int, ...] = ()
    if is_member:
        # Solve x * G = word over GF(q) by RREF on the augmented
        # transpose [G^T | word^T].
        gt = [
            [matrix[r][c] % q for r in range(len(matrix))]
            for c in range(len(matrix[0]))
        ]
        aug_t = [gt[c] + [word[c] % q] for c in range(len(matrix[0]))]
        rref_aug, rank_aug2 = _rref(aug_t, q)
        # Extract solution from augmented column
        coeffs: list[int] = [0] * len(matrix)
        pivot_cols: list[int] = []
        for r in range(rank_aug2):
            for c in range(len(matrix)):
                if rref_aug[r][c] != 0:
                    pivot_cols.append(c)
                    break
        for r in range(rank_aug2):
            for c in range(len(matrix)):
                if rref_aug[r][c] != 0:
                    coeffs[c] = rref_aug[r][-1] % q
                    break
        coefficients = tuple(coeffs)

    syndrome_vec = _mat_mul_vec(
        _nullspace([list(row) for row in request.generator_matrix], q), word, q
    )
    hamming = _hamming_weight(word)
    return CodewordCheckResult(
        is_member=is_member,
        hamming_weight=hamming,
        coefficients=coefficients,
        syndrome=tuple(syndrome_vec),
    )


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    h = [list(row) for row in request.parity_check_matrix]
    word = list(request.word)
    q = request.field_order
    syndrome = _mat_mul_vec(h, word, q)
    is_member = all(v == 0 for v in syndrome)
    return SyndromeResult(
        syndrome=tuple(syndrome),
        is_member=is_member,
    )


def _rowspace_contains(
    g: list[list[int]], target_rref: list[list[int]], target_rank: int, q: int
) -> bool:
    """Check whether the row space of g contains the target row space."""
    augmented = [list(row) for row in g]
    for row in target_rref[:target_rank]:
        augmented.append(list(row))
    _, aug_rank = _rref(augmented, q)
    _, g_rank = _rref([list(r) for r in g], q)
    return aug_rank == g_rank


def _enumerate_code(
    rref: list[list[int]], rank: int, n: int, q: int
) -> set[tuple[int, ...]]:
    """Enumerate all codewords from a RREF basis."""
    from itertools import product

    code = set()
    for coeffs in product(range(q), repeat=rank):
        codeword = [0] * n
        for ci, c in enumerate(coeffs):
            for j in range(n):
                codeword[j] = (codeword[j] + c * rref[ci][j]) % q
        code.add(tuple(codeword))
    return code


def compute_code_equal(request: CodeEqualRequest) -> CodeEqualResult:
    q = request.field_order
    mat_a = [list(row) for row in request.generator_matrix_a]
    mat_b = [list(row) for row in request.generator_matrix_b]

    rref_a, rank_a = _rref([list(r) for r in mat_a], q)
    rref_b, rank_b = _rref([list(r) for r in mat_b], q)

    contain_ab = _rowspace_contains(mat_a, rref_b, rank_b, q)
    contain_ba = _rowspace_contains(mat_b, rref_a, rank_a, q)
    equal = contain_ab and contain_ba

    witness = None
    if not equal:
        n = len(mat_a[0])
        code_a = _enumerate_code(rref_a, rank_a, n, q)
        code_b = _enumerate_code(rref_b, rank_b, n, q)
        diff = code_a.symmetric_difference(code_b)
        if diff:
            witness = sorted(diff)[0]

    return CodeEqualResult(
        equal=equal,
        dimension_a=rank_a,
        dimension_b=rank_b,
        witness_word=witness,
    )


def compute_macwilliams_transform(request: MacWilliamsRequest) -> MacWilliamsResult:
    from math import comb

    q = request.field_order
    primal = list(request.weights)
    n = request.length
    dual: list[int] = []
    for k in range(n + 1):
        s = 0
        for i in range(n + 1):
            for j in range(i + 1):
                if j <= k <= j + (n - i):
                    term = (
                        comb(i, j)
                        * comb(n - i, k - j)
                        * ((-1) ** j)
                        * (q - 1) ** (k - j)
                    )
                    s += primal[i] * term
        dual.append(s // request.code_cardinality)
    return MacWilliamsResult(dual_weights=tuple(dual))


def compute_puncture(request: PunctureRequest) -> PunctureResult:
    matrix = [list(row) for row in request.generator_matrix]
    punctured = [
        row[: request.coordinate] + row[request.coordinate + 1 :] for row in matrix
    ]
    rref, rank = _rref(punctured, request.field_order)
    new_len = len(matrix[0]) - 1
    gen = tuple(tuple(row) for row in rref[:rank]) if rank > 0 else ()
    return PunctureResult(
        generator=gen,
        dimension=rank,
        length=new_len,
    )


def compute_shorten(request: ShortenRequest) -> ShortenResult:
    matrix = [list(row) for row in request.generator_matrix]
    q = request.field_order
    col = request.coordinate

    # Shortening: keep codewords c with c[col] = 0, then delete col.
    # RREF the generator to get a basis, then find the subcode vanishing at col.
    rref, rank = _rref([list(row) for row in matrix], q)
    n = len(matrix[0])

    # Build the column of coordinate values from the RREF basis
    col_values = [rref[i][col] % q for i in range(rank)]

    # Find rows where col is nonzero (pivot rows for the coordinate functional)
    nonzero_rows = [i for i in range(rank) if col_values[i] != 0]

    if not nonzero_rows:
        # All rows already have 0 at col: shortened = punctured code
        shortened_result = [rref[i][:col] + rref[i][col + 1 :] for i in range(rank)]
    else:
        # Keep rows with 0 at col, plus combinations that zero out col
        piv0 = nonzero_rows[0]
        shortened_rows: list[list[int]] = []
        for i in range(rank):
            if i not in nonzero_rows:
                shortened_rows.append(rref[i][:col] + rref[i][col + 1 :])
        for p in nonzero_rows:
            if p == piv0:
                continue
            factor = (col_values[p] * pow(col_values[piv0], -1, q)) % q
            combined = []
            for j in range(n):
                if j == col:
                    continue
                combined.append((rref[p][j] - factor * rref[piv0][j]) % q)
            shortened_rows.append(combined)
        shortened_result = shortened_rows

    final_rref, final_rank = _rref(shortened_result, q) if shortened_result else ([], 0)
    new_len = len(matrix[0]) - 1
    gen = tuple(tuple(row) for row in final_rref[:final_rank]) if final_rank > 0 else ()
    return ShortenResult(
        generator=gen,
        dimension=final_rank,
        length=new_len,
    )
