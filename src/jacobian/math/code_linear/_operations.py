"""Domain functions for linear code structural operations."""

from __future__ import annotations

from itertools import product
from typing import NamedTuple

from jacobian.math.code_linear._models import (
    CodeEqualRequest,
    CodeEqualResult,
    CodewordCheckRequest,
    CodewordCheckResult,
    DualCodeRequest,
    DualCodeResult,
    FromGeneratorResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    ParityCheckMatrix,
    ParityCheckRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ReceivedWordProfileRequest,
    ReceivedWordProfileResult,
    ReceivedWordWitness,
    ShortenRequest,
    ShortenResult,
    SyndromeRequest,
    SyndromeResult,
    _threshold_matches_distance,
)
from jacobian.math.code_linear.values import PrimeFieldLinearEncoder
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rref,
)


class _ReceivedWordProfileData(NamedTuple):
    distance_histogram: tuple[int, ...]
    codeword_count: int
    minimum_distance: int
    maximum_agreement: int
    threshold_match_count: int | None
    witnesses: tuple[ReceivedWordWitness, ...]


def _received_word_profile_data(
    request: ReceivedWordProfileRequest,
) -> _ReceivedWordProfileData:
    encoder = request.encoder
    field_order = encoder.field_order
    dimension = len(encoder.message_axis)
    length = len(encoder.coordinate_axis)
    histogram = [0] * (length + 1)
    threshold_match_count = 0
    witnesses: list[ReceivedWordWitness] = []

    for message in product(range(field_order), repeat=dimension):
        codeword = tuple(
            sum(
                message[row] * encoder.generator_matrix[row][column]
                for row in range(dimension)
            )
            % field_order
            for column in range(length)
        )
        distance = sum(
            left != right
            for left, right in zip(codeword, request.received_word, strict=True)
        )
        agreement = length - distance
        histogram[distance] += 1

        if request.threshold is None or not _threshold_matches_distance(
            request.threshold,
            distance=distance,
            length=length,
        ):
            continue

        threshold_match_count += 1
        if request.witness_mode == "ALL" or (
            request.witness_mode == "FIRST" and not witnesses
        ):
            witnesses.append(
                ReceivedWordWitness(
                    message=message,
                    codeword=codeword,
                    distance=distance,
                    agreement=agreement,
                )
            )

    minimum_distance = next(index for index, count in enumerate(histogram) if count)
    return _ReceivedWordProfileData(
        distance_histogram=tuple(histogram),
        codeword_count=sum(histogram),
        minimum_distance=minimum_distance,
        maximum_agreement=length - minimum_distance,
        threshold_match_count=(
            threshold_match_count if request.threshold is not None else None
        ),
        witnesses=tuple(witnesses),
    )


def compute_received_word_profile(
    request: ReceivedWordProfileRequest,
) -> ReceivedWordProfileResult:
    data = _received_word_profile_data(request)
    return ReceivedWordProfileResult(
        source=request,
        distance_histogram=data.distance_histogram,
        codeword_count=data.codeword_count,
        minimum_distance=data.minimum_distance,
        maximum_agreement=data.maximum_agreement,
        threshold_match_count=data.threshold_match_count,
        witnesses=data.witnesses,
    )


def _rref(matrix: list[list[int]], field_order: int) -> tuple[list[list[int]], int]:
    """Reduced row echelon form and rank over a prime field."""
    shared_matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in matrix),
        columns=len(matrix[0]) if matrix else 0,
    )
    reduced, pivots = rref(shared_matrix)
    return [list(row) for row in reduced], len(pivots)


def _nullspace(
    matrix: list[list[int]], field_order: int, columns: int
) -> list[list[int]]:
    """Compute a basis for the nullspace of a GF(p) matrix with ``columns`` columns."""
    shared_matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in matrix),
        columns=columns,
    )
    return [list(row) for row in nullspace(shared_matrix)]


def _canonical_generator(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    """Return canonical RREF rows for a matrix's row space."""
    rref, rank = _rref(matrix, field_order)
    return list(rref[:rank])


def _canonical_encoder(
    *,
    field_order: int,
    coordinate_axis: tuple[str, ...],
    generator_matrix: list[list[int]],
) -> PrimeFieldLinearEncoder:
    """Build an encoder whose ``m0``, ``m1``, ... labels follow basis order."""

    rows = tuple(tuple(row) for row in generator_matrix)
    return PrimeFieldLinearEncoder(
        field_order=field_order,
        message_axis=tuple(f"m{index}" for index in range(len(rows))),
        coordinate_axis=coordinate_axis,
        generator_matrix=rows,
    )


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
        encoder=_canonical_encoder(
            field_order=request.field_order,
            coordinate_axis=request.coordinate_axis,
            generator_matrix=canonical,
        ),
        dimension=dim,
        length=length,
        cardinality=cardinality,
    )


def compute_dual_code(request: DualCodeRequest) -> DualCodeResult:
    encoder = request.encoder
    q = encoder.field_order
    length = len(encoder.coordinate_axis)
    matrix = [list(row) for row in encoder.generator_matrix]
    _, rank = _rref(matrix, q)
    null = _canonical_generator(_nullspace(matrix, q, length), q)
    dual_encoder = _canonical_encoder(
        field_order=q,
        coordinate_axis=encoder.coordinate_axis,
        generator_matrix=null,
    )
    return DualCodeResult(
        encoder=dual_encoder,
        parity_check=ParityCheckMatrix(
            field_order=q,
            coordinate_axis=encoder.coordinate_axis,
            rows=dual_encoder.generator_matrix,
        ),
        dimension=rank,
        dual_dimension=length - rank,
        length=length,
    )


def compute_parity_check(request: ParityCheckRequest) -> ParityCheckResult:
    encoder = request.encoder
    q = encoder.field_order
    length = len(encoder.coordinate_axis)
    matrix = [list(row) for row in encoder.generator_matrix]
    _, rank = _rref(matrix, q)
    null = _nullspace(matrix, q, length)
    return ParityCheckResult(
        parity_check=ParityCheckMatrix(
            field_order=q,
            coordinate_axis=encoder.coordinate_axis,
            rows=tuple(map(tuple, null)),
        ),
        dimension=rank,
        rank_h=length - rank,
        length=length,
    )


def compute_codeword_check(
    request: CodewordCheckRequest,
) -> CodewordCheckResult:
    encoder = request.encoder
    matrix = [list(row) for row in encoder.generator_matrix]
    word = list(request.word)
    q = encoder.field_order
    length = len(encoder.coordinate_axis)

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
        gt = [[matrix[r][c] % q for r in range(len(matrix))] for c in range(length)]
        aug_t = [gt[c] + [word[c] % q] for c in range(length)]
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

    syndrome_vec = _mat_mul_vec(_nullspace(matrix, q, length), word, q)
    hamming = _hamming_weight(word)
    return CodewordCheckResult(
        is_member=is_member,
        hamming_weight=hamming,
        coefficients=coefficients,
        syndrome=tuple(syndrome_vec),
    )


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    h = [list(row) for row in request.parity_check.rows]
    word = list(request.word)
    q = request.parity_check.field_order
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
    q = request.encoder_a.field_order
    mat_a = [list(row) for row in request.encoder_a.generator_matrix]
    mat_b = [list(row) for row in request.encoder_b.generator_matrix]

    rref_a, rank_a = _rref([list(r) for r in mat_a], q)
    rref_b, rank_b = _rref([list(r) for r in mat_b], q)

    contain_ab = _rowspace_contains(mat_a, rref_b, rank_b, q)
    contain_ba = _rowspace_contains(mat_b, rref_a, rank_a, q)
    equal = contain_ab and contain_ba

    witness = None
    if not equal:
        n = len(request.encoder_a.coordinate_axis)
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
    encoder = request.encoder
    column = request.coordinate
    punctured = [
        list(row[:column] + row[column + 1 :]) for row in encoder.generator_matrix
    ]
    rref, rank = _rref(punctured, encoder.field_order)
    gen = tuple(tuple(row) for row in rref[:rank]) if rank > 0 else ()
    return PunctureResult(
        encoder=_canonical_encoder(
            field_order=encoder.field_order,
            coordinate_axis=(
                encoder.coordinate_axis[:column] + encoder.coordinate_axis[column + 1 :]
            ),
            generator_matrix=[list(row) for row in gen],
        ),
        dimension=rank,
        length=len(encoder.coordinate_axis) - 1,
    )


def compute_shorten(request: ShortenRequest) -> ShortenResult:
    encoder = request.encoder
    q = encoder.field_order
    col = request.coordinate

    # Shortening: keep codewords c with c[col] = 0, then delete col.
    # RREF the generator to get a basis, then find the subcode vanishing at col.
    rref, rank = _rref([list(row) for row in encoder.generator_matrix], q)
    n = len(encoder.coordinate_axis)

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
    new_len = n - 1
    gen = tuple(tuple(row) for row in final_rref[:final_rank]) if final_rank > 0 else ()
    return ShortenResult(
        encoder=_canonical_encoder(
            field_order=q,
            coordinate_axis=(
                encoder.coordinate_axis[:col] + encoder.coordinate_axis[col + 1 :]
            ),
            generator_matrix=[list(row) for row in gen],
        ),
        dimension=final_rank,
        length=new_len,
    )
