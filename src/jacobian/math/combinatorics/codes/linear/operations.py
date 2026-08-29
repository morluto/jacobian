"""Domain functions for linear code structural operations."""

from __future__ import annotations

from itertools import product
from math import comb
from typing import Literal, NamedTuple

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.linear._models import (
    MAX_CODEWORDS,
    MAX_RECEIVED_PROFILE_EXECUTION_WORK,
    MAX_RECEIVED_PROFILE_WITNESS_CELLS,
    CodeEqualResult,
    CodewordCheckResult,
    DualCodeResult,
    FromGeneratorResult,
    MacWilliamsResult,
    ParityCheckMatrix,
    ParityCheckResult,
    PunctureResult,
    ReceivedWordProfileResult,
    ReceivedWordThreshold,
    ReceivedWordWitness,
    ShortenResult,
    SyndromeResult,
    _require_selected_coordinate,
    _threshold_matches_distance,
    _validate_coordinate_axis,
    _validate_prime_matrix,
)
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rref,
)

__all__ = [
    "code_equal",
    "codeword_check",
    "dual_code",
    "from_generator",
    "macwilliams_transform",
    "parity_check",
    "puncture",
    "received_word_profile",
    "shorten",
    "syndrome",
]


class _ReceivedWordProfileData(NamedTuple):
    distance_histogram: tuple[int, ...]
    codeword_count: int
    minimum_distance: int
    maximum_agreement: int
    threshold_match_count: int | None
    witnesses: tuple[ReceivedWordWitness, ...]


WitnessMode = Literal["NONE", "COUNT", "FIRST", "ALL"]


def _domain_error(location: tuple[str, ...], error: PydanticCustomError) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=error.type,
        message=error.message(),
    )


def _fail(location: tuple[str, ...], code: str, message: str) -> None:
    _domain_error(location, PydanticCustomError(f"code_linear.{code}", message))


def _admit_word(encoder: PrimeFieldLinearEncoder, word: tuple[int, ...]) -> None:
    if len(word) != len(encoder.coordinate_axis):
        _fail(
            ("word",),
            "word_length_must_match_the_encoder_coordinate_axis",
            "word length must match the encoder coordinate axis",
        )
    if any(value < 0 or value >= encoder.field_order for value in word):
        _fail(
            ("word",),
            "word_entries_must_be_canonical_field_residues",
            "word entries must be canonical field residues",
        )


def _admit_syndrome(
    parity_check: ParityCheckMatrix,
    coordinate_axis: tuple[str, ...],
    word: tuple[int, ...],
) -> None:
    if coordinate_axis != parity_check.coordinate_axis:
        _fail(
            ("coordinate_axis",),
            "word_axis_must_match_the_parity_check_column_axis",
            "word axis must match the parity-check column axis",
        )
    if len(word) != len(coordinate_axis):
        _fail(
            ("word",),
            "word_length_must_match_code_length",
            "word length must match code length",
        )
    if any(value < 0 or value >= parity_check.field_order for value in word):
        _fail(
            ("word",),
            "word_entries_must_be_canonical_field_residues",
            "word entries must be canonical field residues",
        )


def _admit_comparable_encoders(
    encoder_a: PrimeFieldLinearEncoder, encoder_b: PrimeFieldLinearEncoder
) -> None:
    if encoder_a.field_order != encoder_b.field_order:
        _fail(
            ("encoder_b",),
            "encoders_must_share_one_prime_field_order",
            "encoders must share one prime field order",
        )
    if encoder_a.coordinate_axis != encoder_b.coordinate_axis:
        _fail(
            ("encoder_b",),
            "encoders_must_share_one_ordered_coordinate_axis",
            "encoders must share one ordered coordinate axis",
        )


def _admit_macwilliams(
    field_order: int, code_cardinality: int, length: int, weights: tuple[int, ...]
) -> None:
    from sympy import isprime

    if not isprime(field_order):
        _fail(
            ("field_order",), "field_order_must_be_prime", "field_order must be prime"
        )
    if code_cardinality < 1:
        _fail(
            ("code_cardinality",),
            "code_cardinality_positive",
            "code cardinality must be positive",
        )
    if not 1 <= length <= 64:
        _fail(("length",), "length_out_of_range", "length must be between 1 and 64")
    if len(weights) != length + 1:
        _fail(
            ("weights",),
            "weights_must_have_length_1_entries",
            "weights must have length + 1 entries",
        )
    if any(weight < 0 for weight in weights):
        _fail(
            ("weights",),
            "weight_counts_must_be_non_negative",
            "weight counts must be non-negative",
        )
    if not weights or weights[0] != 1:
        _fail(
            ("weights",),
            "first_weight_count_must_be_1_zero_codeword",
            "first weight count must be 1 (zero codeword)",
        )
    if sum(weights) != code_cardinality:
        _fail(
            ("weights",),
            "weight_counts_must_sum_to_code_cardinality",
            "weight counts must sum to code cardinality",
        )


def _admit_coordinate(encoder: PrimeFieldLinearEncoder, coordinate: int) -> None:
    if coordinate < 0:
        _fail(
            ("coordinate",),
            "coordinate_index_out_of_range",
            "coordinate index out of range",
        )
    try:
        _require_selected_coordinate(encoder, coordinate)
    except PydanticCustomError as error:
        _domain_error(("coordinate",), error)


def _admit_received_word_profile(
    encoder: PrimeFieldLinearEncoder,
    received_word: tuple[int, ...],
    threshold: ReceivedWordThreshold | None,
    witness_mode: WitnessMode,
) -> None:
    length = len(encoder.coordinate_axis)
    if len(received_word) != length:
        _domain_error(
            ("received_word",),
            PydanticCustomError(
                "code_linear.received_word_must_match_the_encoder_coordinate_axis",
                "received word must match the encoder coordinate axis",
            ),
        )
    if any(value < 0 or value >= encoder.field_order for value in received_word):
        _domain_error(
            ("received_word",),
            PydanticCustomError(
                "code_linear.received_word_entries_must_be_canonical_field_residues",
                "received-word entries must be canonical field residues",
            ),
        )
    if threshold is None and witness_mode != "NONE":
        _domain_error(
            ("witness_mode",),
            PydanticCustomError(
                "code_linear.a_witness_mode_requires_an_exact_threshold",
                "a witness mode requires an exact threshold",
            ),
        )
    if threshold is not None:
        if witness_mode == "NONE":
            _domain_error(
                ("witness_mode",),
                PydanticCustomError(
                    "code_linear.a_threshold_requires_count_first_or_all_mode",
                    "a threshold requires COUNT, FIRST, or ALL mode",
                ),
            )
        if threshold.value > length:
            _domain_error(
                ("threshold",),
                PydanticCustomError(
                    "code_linear.threshold_value_cannot_exceed_the_code_length",
                    "threshold value cannot exceed the code length",
                ),
            )


def _maximum_witness_cells(
    encoder: PrimeFieldLinearEncoder, threshold: ReceivedWordThreshold | None
) -> int:
    if threshold is None:
        return 0
    length = len(encoder.coordinate_axis)
    field_order = int(encoder.field_order)
    row_width = len(encoder.message_axis) + length + 2
    ambient_match_count: int = sum(
        comb(length, distance) * (field_order - 1) ** distance
        for distance in range(length + 1)
        if _threshold_matches_distance(threshold, distance=distance, length=length)
    )
    return min(int(encoder.codeword_count), ambient_match_count) * row_width


def _received_word_profile_data(
    encoder: PrimeFieldLinearEncoder,
    received_word: tuple[int, ...],
    threshold: ReceivedWordThreshold | None,
    witness_mode: WitnessMode,
) -> _ReceivedWordProfileData:
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
            left != right for left, right in zip(codeword, received_word, strict=True)
        )
        agreement = length - distance
        histogram[distance] += 1

        if threshold is None or not _threshold_matches_distance(
            threshold,
            distance=distance,
            length=length,
        ):
            continue

        threshold_match_count += 1
        if witness_mode == "ALL" or (witness_mode == "FIRST" and not witnesses):
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
            threshold_match_count if threshold is not None else None
        ),
        witnesses=tuple(witnesses),
    )


def received_word_profile(
    encoder: PrimeFieldLinearEncoder,
    received_word: tuple[int, ...],
    threshold: ReceivedWordThreshold | None = None,
    witness_mode: WitnessMode = "NONE",
) -> ReceivedWordProfileResult:
    _admit_received_word_profile(encoder, received_word, threshold, witness_mode)
    execution_work = (
        2
        * encoder.codeword_count
        * len(encoder.coordinate_axis)
        * (len(encoder.message_axis) + 1)
    )
    if execution_work > MAX_RECEIVED_PROFILE_EXECUTION_WORK:
        raise OperationDomainValidationError(
            location=("encoder",),
            code="code_linear.profile_execution_work_exceeded",
            message="received-word profile execution work exceeds its bound",
        )
    if (
        witness_mode == "ALL"
        and _maximum_witness_cells(encoder, threshold)
        > MAX_RECEIVED_PROFILE_WITNESS_CELLS
    ):
        raise OperationDomainValidationError(
            location=("witness_mode",),
            code="code_linear.witness_cells_exceeded",
            message="all-witness result exceeds its aggregate witness-cell bound",
        )
    data = _received_word_profile_data(encoder, received_word, threshold, witness_mode)
    return ReceivedWordProfileResult._from_kernel(
        encoder=encoder,
        received_word=received_word,
        threshold=threshold,
        witness_mode=witness_mode,
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


def from_generator(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
    coordinate_axis: tuple[str, ...],
) -> FromGeneratorResult:
    try:
        width = _validate_prime_matrix(field_order, generator_matrix)
        _validate_coordinate_axis(coordinate_axis, width=width)
    except PydanticCustomError as error:
        _domain_error(("generator_matrix", "coordinate_axis"), error)
    if field_order ** len(generator_matrix) > MAX_CODEWORDS:
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_linear.generator_matrix_exceeds_exact_enumeration_bound",
            message="generator matrix exceeds exact enumeration bound",
        )
    matrix = [list(row) for row in generator_matrix]
    canonical = _canonical_generator(matrix, field_order)
    dim = len(canonical)
    length = len(generator_matrix[0])
    cardinality = field_order**dim
    return FromGeneratorResult(
        encoder=_canonical_encoder(
            field_order=field_order,
            coordinate_axis=coordinate_axis,
            generator_matrix=canonical,
        ),
        dimension=dim,
        length=length,
        cardinality=cardinality,
    )


def dual_code(encoder: PrimeFieldLinearEncoder) -> DualCodeResult:
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


def parity_check(encoder: PrimeFieldLinearEncoder) -> ParityCheckResult:
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


def codeword_check(
    encoder: PrimeFieldLinearEncoder, word: tuple[int, ...]
) -> CodewordCheckResult:
    matrix = [list(row) for row in encoder.generator_matrix]
    _admit_word(encoder, word)
    word_values = list(word)
    q = encoder.field_order
    length = len(encoder.coordinate_axis)

    # Word is a codeword iff it lies in the row space of the generator.
    # Augment the generator with the word as a new row and check
    # whether rank increases.
    _, rank_g = _rref([list(row) for row in matrix], q)
    augmented = [list(row) for row in matrix] + [word_values]
    _, rank_aug = _rref(augmented, q)
    is_member = rank_aug == rank_g

    coefficients: tuple[int, ...] = ()
    if is_member:
        # Solve x * G = word over GF(q) by RREF on the augmented
        # transpose [G^T | word^T].
        gt = [[matrix[r][c] % q for r in range(len(matrix))] for c in range(length)]
        aug_t = [gt[c] + [word_values[c] % q] for c in range(length)]
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

    syndrome_vec = _mat_mul_vec(_nullspace(matrix, q, length), word_values, q)
    hamming = _hamming_weight(word_values)
    return CodewordCheckResult(
        is_member=is_member,
        hamming_weight=hamming,
        coefficients=coefficients,
        syndrome=tuple(syndrome_vec),
    )


def syndrome(
    parity_check_matrix: ParityCheckMatrix,
    coordinate_axis: tuple[str, ...],
    word: tuple[int, ...],
) -> SyndromeResult:
    _admit_syndrome(parity_check_matrix, coordinate_axis, word)
    h = [list(row) for row in parity_check_matrix.rows]
    word_values = list(word)
    q = parity_check_matrix.field_order
    syndrome = _mat_mul_vec(h, word_values, q)
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


def code_equal(
    encoder_a: PrimeFieldLinearEncoder, encoder_b: PrimeFieldLinearEncoder
) -> CodeEqualResult:
    if (
        encoder_a.codeword_count > MAX_CODEWORDS
        or encoder_b.codeword_count > MAX_CODEWORDS
    ):
        raise OperationDomainValidationError(
            location=("encoder",),
            code="code_linear.code_cardinality_exceeds_exact_enumeration_bound",
            message="code cardinality exceeds exact enumeration bound",
        )
    _admit_comparable_encoders(encoder_a, encoder_b)
    q = encoder_a.field_order
    mat_a = [list(row) for row in encoder_a.generator_matrix]
    mat_b = [list(row) for row in encoder_b.generator_matrix]

    rref_a, rank_a = _rref([list(r) for r in mat_a], q)
    rref_b, rank_b = _rref([list(r) for r in mat_b], q)

    contain_ab = _rowspace_contains(mat_a, rref_b, rank_b, q)
    contain_ba = _rowspace_contains(mat_b, rref_a, rank_a, q)
    equal = contain_ab and contain_ba

    witness = None
    if not equal:
        n = len(encoder_a.coordinate_axis)
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


def macwilliams_transform(
    field_order: int,
    code_cardinality: int,
    length: int,
    weights: tuple[int, ...],
) -> MacWilliamsResult:
    _admit_macwilliams(field_order, code_cardinality, length, weights)
    q = field_order
    primal = list(weights)
    n = length
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
        dual.append(s // code_cardinality)
    return MacWilliamsResult(dual_weights=tuple(dual))


def puncture(encoder: PrimeFieldLinearEncoder, coordinate: int) -> PunctureResult:
    _admit_coordinate(encoder, coordinate)
    column = coordinate
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


def shorten(encoder: PrimeFieldLinearEncoder, coordinate: int) -> ShortenResult:
    _admit_coordinate(encoder, coordinate)
    q = encoder.field_order
    col = coordinate

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
