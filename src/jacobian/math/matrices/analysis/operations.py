"""Domain-owned matrix analysis operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, factorial, lcm, log10
from typing import Any, Literal

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
    format_canonical_rational,
    require_bounded_rational,
)
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices._number_field import (
    EmbeddedNumberFieldRecognitionError,
    RecognizedRealSimpleNumberField,
    domain_matrix_from_embedded,
    field_element_sign,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.analysis._models import (
    _RATIONAL_SPECTRUM_CLAIM_BYTES,
    _RATIONAL_SPECTRUM_MATRIX_ENTRY_BYTES,
    _RATIONAL_SPECTRUM_RESULT_BASE_BYTES,
    _RESULT_ENVELOPE_RESERVE_BYTES,
    MAX_INERTIA_DIGIT_WORK,
    MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
    MAX_RATIONAL_SPECTRUM_MINOR_DIGITS,
    MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES,
    MAX_RATIONAL_SPECTRUM_ORDER,
    MAX_RATIONAL_SPECTRUM_RANK_WORK,
    MAX_RATIONAL_SPECTRUM_RESULT_BYTES,
    MAX_RATIONAL_SPECTRUM_SHIFTED_DIGITS,
    FarkasCertificateResult,
    InertiaResult,
    RationalSpectrumClaimResult,
    RationalSpectrumFailure,
    RationalSpectrumMultiplicityClaim,
    RationalSpectrumNullityLedgerEntry,
    _validation_error,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    ExactRealMatrix,
    RationalMatrix,
    require_matrix_scalar_digits,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
)

type _InertiaRegime = Literal["DIAGONAL", "GENERAL"]
type _ExecutionCheckpoint = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class _InertiaExecutionPlan:
    """One source-derived exact inertia envelope and congruence regime."""

    regime: _InertiaRegime
    order: int
    algebraic: bool
    digit_work: int


_INERTIA_WALL_SECONDS: float = 600.0


def _require_inertia_execution_active(phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and execution.deadline <= time.monotonic()
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _admit_rational_spectrum_claim(
    matrix: RationalMatrix,
    claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...],
) -> None:
    order = len(matrix.entries)
    if order != len(matrix.entries[0]):
        raise _validation_error(
            "shape_mismatch", "rational spectrum claims require a square matrix"
        )
    if order > MAX_RATIONAL_SPECTRUM_ORDER:
        raise _validation_error(
            "budget_exceeded",
            f"rational spectrum claims support matrix order at most {MAX_RATIONAL_SPECTRUM_ORDER}",
        )
    if any(
        matrix.entries[row][column] != matrix.entries[column][row]
        for row in range(order)
        for column in range(row + 1, order)
    ):
        raise _validation_error(
            "budget_exceeded", "rational spectrum claims require a symmetric matrix"
        )
    require_matrix_scalar_digits(
        matrix.entries,
        maximum=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
        label="rational spectrum matrix",
    )
    nonzero_entries = sum(entry.num != "0" for row in matrix.entries for entry in row)
    if nonzero_entries > MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES:
        raise _validation_error(
            "budget_exceeded",
            "rational spectrum matrix exceeds the nonzero-entry budget",
        )
    eigenvalues = tuple(claim.eigenvalue for claim in claimed_profile)
    for eigenvalue in eigenvalues:
        require_bounded_rational(
            eigenvalue,
            max_digits=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
            label="claimed eigenvalue",
        )
    if len(set(eigenvalues)) != len(eigenvalues):
        raise _validation_error(
            "budget_exceeded", "claimed rational eigenvalues must be pairwise distinct"
        )
    shifted_digits = max(
        canonical_rational_component_digits(
            CanonicalRational.from_fraction(
                matrix.entries[index][index].as_fraction() - eigenvalue.as_fraction()
            )
        )
        for eigenvalue in eigenvalues
        for index in range(order)
    )
    if shifted_digits > MAX_RATIONAL_SPECTRUM_SHIFTED_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "shifted diagonal entries exceed the rational spectrum digit budget",
        )
    if len(eigenvalues) * order**3 > MAX_RATIONAL_SPECTRUM_RANK_WORK:
        raise _validation_error(
            "budget_exceeded",
            "shifted-rank computations exceed the aggregate work budget",
        )
    minor_digits = order * order * shifted_digits + len(str(factorial(order))) + 1
    if minor_digits > MAX_RATIONAL_SPECTRUM_MINOR_DIGITS:
        raise _validation_error(
            "budget_exceeded", "exact shifted-rank minors exceed the digit budget"
        )
    result_bytes = (
        _RATIONAL_SPECTRUM_RESULT_BASE_BYTES
        + order
        * order
        * (
            2 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS
            + _RATIONAL_SPECTRUM_MATRIX_ENTRY_BYTES
        )
        + len(eigenvalues)
        * (4 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS + _RATIONAL_SPECTRUM_CLAIM_BYTES)
    )
    if result_bytes > MAX_RATIONAL_SPECTRUM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded", "rational spectrum ledger exceeds the result-size budget"
        )


def _is_diagonal(matrix: ExactRealMatrix) -> bool:
    order = len(matrix.entries)
    if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
        return all(
            all(
                coordinate.num == "0"
                for coordinate in matrix.entries[row][column].coefficients_ascending
            )
            for row in range(order)
            for column in range(order)
            if row != column
        )
    return all(
        matrix.entries[row][column].num == "0"
        for row in range(order)
        for column in range(order)
        if row != column
    )


def _denominator_product_digit_bound(denominators: tuple[str, ...]) -> int:
    """Bound product width without charging harmless unit denominators."""

    return max(1, sum(len(value) for value in denominators if value != "1"))


def _rational_general_inertia_work(matrix: RationalMatrix) -> int:
    """Bound fraction elimination through minors after diagonal clearing.

    If ``d_i`` is the least common multiple of denominators in row ``i``,
    ``diag(d_i) A diag(d_i)`` is an integral congruent matrix. Every Schur
    complement entry produced by symmetric elimination is a ratio of minors of
    that matrix. Hadamard's determinant bound therefore controls both rational
    components before any elimination begins.
    """

    order = len(matrix.entries)
    # The product of a row's denominators is a denominator multiple, so its
    # decimal width gives a cheap conservative bound without constructing an
    # enormous LCM during admission.
    row_denominator_digits = tuple(
        _denominator_product_digit_bound(tuple(value.den for value in row))
        for row in matrix.entries
    )
    scaled_component_digits = max(
        len(value.num.lstrip("-"))
        + row_denominator_digits[row]
        + row_denominator_digits[column]
        + 2
        for row, entries in enumerate(matrix.entries)
        for column, value in enumerate(entries)
    )
    minor_digits = (
        order * scaled_component_digits
        + len(str(factorial(order)))
        + ceil(order * 0.5 * log10(max(order, 1)))
        + 2
    )
    return 4 * order**3 * minor_digits


def _algebraic_source_height(
    matrix: EmbeddedRealSimpleNumberFieldMatrix,
) -> tuple[int, int, int]:
    """Return degree, scaled coordinate height, and reduction-cost digits."""

    degree = matrix.embedding.presentation.degree
    row_denominator_digits = tuple(
        _denominator_product_digit_bound(
            tuple(
                coordinate.den
                for value in row
                for coordinate in value.coefficients_ascending
            )
        )
        for row in matrix.entries
    )
    scaled_component_digits = max(
        len(coordinate.num.lstrip("-"))
        + row_denominator_digits[row]
        + row_denominator_digits[column]
        + 2
        for row, entries in enumerate(matrix.entries)
        for column, value in enumerate(entries)
        for coordinate in value.coefficients_ascending
    )
    field_digits = max(
        len(coefficient.lstrip("-"))
        for coefficient in matrix.embedding.presentation.coefficients_descending
    )
    leading_digits = len(
        matrix.embedding.presentation.coefficients_descending[0].lstrip("-")
    )
    reduction_digits = degree * (
        field_digits + leading_digits + ceil(log10(degree + 1)) + 2
    )
    return degree, scaled_component_digits, reduction_digits


def _algebraic_inertia_work(
    matrix: EmbeddedRealSimpleNumberFieldMatrix,
    *,
    diagonal: bool,
) -> int:
    """Bound exact field arithmetic and embedding-aware root isolation."""

    order = len(matrix.entries)
    degree, source_digits, reduction_digits = _algebraic_source_height(matrix)
    conversion_work = order**2 * degree * (source_digits + reduction_digits)
    if diagonal:
        nonzero_diagonal = sum(
            any(coordinate.num != "0" for coordinate in value.coefficients_ascending)
            for value in (matrix.entries[index][index] for index in range(order))
        )
        # A nonzero reduced polynomial p has degree < d and is coprime to the
        # irreducible defining polynomial f. Exact root isolation of f*p has
        # degree at most 2d-1 and coefficient height bounded here.
        sign_height = source_digits + reduction_digits + 2
        sign_work = nonzero_diagonal * (2 * degree - 1) ** 3 * sign_height
        return conversion_work + order**2 * degree + sign_work

    minor_digits = (
        order * (source_digits + reduction_digits) + len(str(factorial(order))) + 2
    )
    # A Schur entry is a ratio of field-valued minors. Inverting the
    # denominator element is a degree-d rational linear solve; Cramer's rule
    # contributes at most 2d+3 minor-height factors.
    ratio_digits = (2 * degree + 3) * (minor_digits + reduction_digits)
    congruence_work = 4 * order**3 * degree**2 * ratio_digits
    sign_work = order * (2 * degree - 1) ** 3 * ratio_digits
    return conversion_work + congruence_work + sign_work


def _diagonal_rational_inertia_work(matrix: RationalMatrix) -> int:
    order = len(matrix.entries)
    scalar_digits = max(
        len(component.lstrip("-"))
        for index in range(order)
        for component in (
            matrix.entries[index][index].num,
            matrix.entries[index][index].den,
        )
    )
    return order**2 + order * scalar_digits


def _require_inertia_digit_work(digit_work: int) -> None:
    if digit_work > MAX_INERTIA_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "exact inertia arithmetic and intermediate heights exceed the "
            f"{MAX_INERTIA_DIGIT_WORK:,}-unit digit-work bound",
        )


def _admit_inertia_from_bounds(
    *,
    order: int,
    algebraic: bool,
    algebraic_degree: int,
    numerator_digits: int,
    denominator_digits: int,
    field_digits: int,
    leading_digits: int,
    zero_matrix: bool,
) -> _InertiaExecutionPlan:
    """Admit a not-yet-materialized symmetric matrix from source height bounds."""

    if zero_matrix:
        digit_work = (
            order**2
            if not algebraic
            else order**2
            * algebraic_degree
            * (numerator_digits + denominator_digits + field_digits + leading_digits)
        )
        regime: _InertiaRegime = "DIAGONAL"
    elif not algebraic:
        row_denominator_digits = order * denominator_digits
        scaled_component_digits = numerator_digits + 2 * row_denominator_digits + 2
        minor_digits = (
            order * scaled_component_digits
            + len(str(factorial(order)))
            + ceil(order * 0.5 * log10(max(order, 1)))
            + 2
        )
        digit_work = 4 * order**3 * minor_digits
        regime = "GENERAL"
    else:
        row_denominator_digits = order * algebraic_degree * denominator_digits
        source_digits = numerator_digits + 2 * row_denominator_digits + 2
        reduction_digits = algebraic_degree * (
            field_digits + leading_digits + ceil(log10(algebraic_degree + 1)) + 2
        )
        conversion_work = (
            order**2 * algebraic_degree * (source_digits + reduction_digits)
        )
        minor_digits = (
            order * (source_digits + reduction_digits) + len(str(factorial(order))) + 2
        )
        ratio_digits = (2 * algebraic_degree + 3) * (minor_digits + reduction_digits)
        congruence_work = 4 * order**3 * algebraic_degree**2 * ratio_digits
        sign_work = order * (2 * algebraic_degree - 1) ** 3 * ratio_digits
        digit_work = conversion_work + congruence_work + sign_work
        regime = "GENERAL"
    _require_inertia_digit_work(digit_work)
    return _InertiaExecutionPlan(
        regime=regime,
        order=order,
        algebraic=algebraic,
        digit_work=digit_work,
    )


def _admit_inertia(matrix: ExactRealMatrix) -> _InertiaExecutionPlan:
    order = len(matrix.entries)
    if order != len(matrix.entries[0]):
        raise _validation_error("shape_mismatch", "inertia requires a square matrix")
    if any(
        matrix.entries[row][column] != matrix.entries[column][row]
        for row in range(order)
        for column in range(row + 1, order)
    ):
        raise _validation_error("shape_mismatch", "inertia requires a symmetric matrix")
    output_limit = CanonicalLimits().max_output_bytes
    try:
        retained_bytes = len(encode_strict_json(matrix.model_dump(mode="json")))
    except CanonicalizationError:
        retained_bytes = output_limit + 1
    if retained_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
        raise _validation_error(
            "budget_exceeded",
            "the inertia result retains its source matrix and would exceed "
            "the canonical output limit",
        )

    diagonal = _is_diagonal(matrix)
    if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
        degree = matrix.embedding.presentation.degree
        if degree > MAX_NUMBER_FIELD_EMBEDDING_DEGREE:
            raise _validation_error(
                "budget_exceeded",
                "exact algebraic inertia supports field degree at most "
                f"{MAX_NUMBER_FIELD_EMBEDDING_DEGREE}",
            )
        digit_work = _algebraic_inertia_work(matrix, diagonal=diagonal)
    elif diagonal:
        digit_work = _diagonal_rational_inertia_work(matrix)
    else:
        digit_work = _rational_general_inertia_work(matrix)
    _require_inertia_digit_work(digit_work)
    return _InertiaExecutionPlan(
        regime="DIAGONAL" if diagonal else "GENERAL",
        order=order,
        algebraic=isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix),
        digit_work=digit_work,
    )


def _exact_shifted_nullities(
    matrix: RationalMatrix,
    eigenvalues: tuple[CanonicalRational, ...],
) -> tuple[int, ...]:
    """Return nullities of ``matrix - eigenvalue * I`` over QQ.

    Each rational row is scaled by the least common multiple of its
    denominators. This preserves rank and gives FLINT an exact integer matrix;
    no floating-point or symbolic expression crosses the adapter boundary.
    """
    from flint import fmpz_mat

    source = tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.entries
    )
    order = len(source)
    nullities: list[int] = []
    for canonical_eigenvalue in eigenvalues:
        eigenvalue = canonical_eigenvalue.as_fraction()
        integer_rows: list[list[int]] = []
        for row_index, source_row in enumerate(source):
            shifted_row = [
                entry - eigenvalue if row_index == column_index else entry
                for column_index, entry in enumerate(source_row)
            ]
            row_denominator = lcm(*(entry.denominator for entry in shifted_row))
            integer_rows.append(
                [
                    entry.numerator * (row_denominator // entry.denominator)
                    for entry in shifted_row
                ]
            )
        nullities.append(order - int(fmpz_mat(integer_rows).rank()))
    return tuple(nullities)


def check_rational_spectrum_claim(
    matrix: RationalMatrix,
    claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...],
) -> RationalSpectrumClaimResult:
    """Check a claimed complete rational spectrum of a symmetric QQ matrix."""
    try:
        _admit_rational_spectrum_claim(matrix, claimed_profile)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("matrix", "claimed_profile"),
            code=exc.type,
            message=exc.message(),
        ) from exc
    ledger, claimed_sum, established_sum, mismatch, failure = (
        _evaluate_rational_spectrum_claim(matrix, claimed_profile)
    )
    return RationalSpectrumClaimResult._from_kernel(
        matrix=matrix,
        claimed_profile=claimed_profile,
        nullity_ledger=ledger,
        claimed_multiplicity_sum=claimed_sum,
        established_multiplicity_sum=established_sum,
        first_failed_claim_index=mismatch,
        first_failed_condition=failure,
    )


def _evaluate_rational_spectrum_claim(
    matrix: RationalMatrix,
    claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...],
) -> tuple[
    tuple[RationalSpectrumNullityLedgerEntry, ...],
    int,
    int,
    int | None,
    RationalSpectrumFailure | None,
]:
    """Recompute the complete source-bound claim ledger."""
    nullities = _exact_shifted_nullities(
        matrix,
        tuple(claim.eigenvalue for claim in claimed_profile),
    )
    ledger = tuple(
        RationalSpectrumNullityLedgerEntry(
            eigenvalue=claim.eigenvalue,
            claimed_multiplicity=claim.multiplicity,
            exact_nullity=nullity,
            multiplicity_matches=nullity == claim.multiplicity,
        )
        for claim, nullity in zip(claimed_profile, nullities, strict=True)
    )
    claimed_sum = sum(claim.multiplicity for claim in claimed_profile)
    mismatch = next(
        (index for index, entry in enumerate(ledger) if not entry.multiplicity_matches),
        None,
    )
    if mismatch is not None:
        failure: RationalSpectrumFailure | None = "MULTIPLICITY_MISMATCH"
    elif claimed_sum != len(matrix.entries):
        failure = "CLAIMED_MULTIPLICITY_SUM_DOES_NOT_EQUAL_MATRIX_ORDER"
    else:
        failure = None
    return ledger, claimed_sum, sum(nullities), mismatch, failure


def _swap_symmetric(matrix: list[list[Fraction]], left: int, right: int) -> None:
    if left == right:
        return
    matrix[left], matrix[right] = matrix[right], matrix[left]
    for row in matrix:
        row[left], row[right] = row[right], row[left]


def _count_2x2_inertia(
    aa: Fraction, bb: Fraction, cc: Fraction
) -> tuple[int, int, int]:
    det = aa * cc - bb * bb
    trace = aa + cc
    if det < 0:
        return 1, 1, 0
    if det > 0:
        if aa > 0 or (aa == 0 and cc > 0):
            return 2, 0, 0
        return 0, 2, 0
    if trace > 0:
        return 1, 0, 1
    if trace < 0:
        return 0, 1, 1
    return 0, 0, 2


def _eliminate_1x1(matrix: list[list[Fraction]], index: int, pivot: int) -> int:
    _swap_symmetric(matrix, index, pivot)
    original = [row[:] for row in matrix]
    diagonal = matrix[index][index]
    for row in range(index + 1, len(matrix)):
        if original[row][index] == 0:
            continue
        factor = original[row][index] / diagonal
        for col in range(index + 1, len(matrix)):
            matrix[row][col] = (
                original[row][col] - factor * original[index][col]
            )
    for row in range(index + 1, len(matrix)):
        for col in range(row + 1, len(matrix)):
            matrix[col][row] = matrix[row][col]
        matrix[row][index] = Fraction(0)
        matrix[index][row] = Fraction(0)
    return 1 if diagonal > 0 else -1


def _find_off_diagonal(
    matrix: list[list[Fraction]], index: int
) -> tuple[int, int] | None:
    for row in range(index, len(matrix)):
        for col in range(row + 1, len(matrix)):
            if matrix[row][col] != 0:
                return row, col
    return None


def _eliminate_2x2(matrix: list[list[Fraction]], index: int) -> tuple[int, int, int]:
    first, second = _find_off_diagonal(matrix, index) or (index, index)
    _swap_symmetric(matrix, index, first)
    if second == index:
        second = first
    _swap_symmetric(matrix, index + 1, second)
    pos, neg, zero = _count_2x2_inertia(
        matrix[index][index],
        matrix[index][index + 1],
        matrix[index + 1][index + 1],
    )
    det = (
        matrix[index][index] * matrix[index + 1][index + 1]
        - matrix[index][index + 1] ** 2
    )
    if index + 2 < len(matrix) and det != 0:
        original = [row[:] for row in matrix]
        inv00 = matrix[index + 1][index + 1] / det
        inv01 = -matrix[index][index + 1] / det
        inv11 = matrix[index][index] / det
        pivot_row0 = tuple(matrix[index][col] for col in range(index + 2, len(matrix)))
        pivot_row1 = tuple(
            matrix[index + 1][col] for col in range(index + 2, len(matrix))
        )
        for row in range(index + 2, len(matrix)):
            left = original[row][index]
            right = original[row][index + 1]
            coeff0 = left * inv00 + right * inv01
            coeff1 = left * inv01 + right * inv11
            for offset, col in enumerate(range(index + 2, len(matrix))):
                matrix[row][col] = (
                    original[row][col]
                    - coeff0 * pivot_row0[offset]
                    - coeff1 * pivot_row1[offset]
                )
        for row in range(index + 2, len(matrix)):
            for col in range(row + 1, len(matrix)):
                matrix[col][row] = matrix[row][col]
            matrix[row][index] = Fraction(0)
            matrix[row][index + 1] = Fraction(0)
            matrix[index][row] = Fraction(0)
            matrix[index + 1][row] = Fraction(0)
    return pos, neg, zero


def _symmetric_inertia(
    matrix: list[list[Fraction]],
    *,
    checkpoint: _ExecutionCheckpoint,
) -> tuple[int, int, int]:
    """Reduce a symmetric rational matrix to a congruence-diagonal form."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    n_pos = n_neg = n_zero = 0
    index = 0
    while index < n:
        checkpoint("during exact rational congruence elimination")
        pivot = next((row for row in range(index, n) if a[row][row] != 0), None)
        if pivot is not None:
            sign = _eliminate_1x1(a, index, pivot)
            n_pos += sign > 0
            n_neg += sign < 0
            index += 1
            continue
        if _find_off_diagonal(a, index) is None:
            n_zero += n - index
            break
        pos, neg, zero = _eliminate_2x2(a, index)
        n_pos += pos
        n_neg += neg
        n_zero += zero
        index += 2
    return n_pos, n_neg, n_zero


def _symmetric_algebraic_inertia(
    matrix: list[list[Any]],
    *,
    sign: Callable[[Any], int],
    checkpoint: _ExecutionCheckpoint,
) -> tuple[int, int, int]:
    """Reduce a symmetric number-field matrix by exact congruences."""

    a = [row[:] for row in matrix]
    n = len(a)
    n_pos = n_neg = n_zero = 0
    index = 0
    while index < n:
        checkpoint("during exact algebraic congruence elimination")
        pivot = next((row for row in range(index, n) if bool(a[row][row])), None)
        if pivot is not None:
            _swap_symmetric(a, index, pivot)
            diagonal = a[index][index]
            diagonal_sign = sign(diagonal)
            if diagonal_sign > 0:
                n_pos += 1
            else:
                n_neg += 1
            original = [row[:] for row in a]
            for row in range(index + 1, n):
                if not original[row][index]:
                    continue
                factor = original[row][index] / diagonal
                for column in range(row, n):
                    a[row][column] = (
                        original[row][column] - factor * original[index][column]
                    )
            for row in range(index + 1, n):
                for column in range(row + 1, n):
                    a[column][row] = a[row][column]
                a[row][index] = 0
                a[index][row] = 0
            index += 1
            continue

        pair = next(
            (
                (row, column)
                for row in range(index, n)
                for column in range(row + 1, n)
                if bool(a[row][column])
            ),
            None,
        )
        if pair is None:
            n_zero += n - index
            break
        first, second = pair
        _swap_symmetric(a, index, first)
        if second == index:
            second = first
        _swap_symmetric(a, index + 1, second)
        # Every remaining diagonal is zero, so the selected block is
        # [[0,b],[b,0]] with negative determinant and inertia (1,1).
        b = a[index][index + 1]
        n_pos += 1
        n_neg += 1
        if index + 2 < n:
            original = [row[:] for row in a]
            inverse_off_diagonal = b**-1
            for row in range(index + 2, n):
                left = original[row][index]
                right = original[row][index + 1]
                coefficient0 = right * inverse_off_diagonal
                coefficient1 = left * inverse_off_diagonal
                for column in range(index + 2, n):
                    a[row][column] = (
                        original[row][column]
                        - coefficient0 * original[index][column]
                        - coefficient1 * original[index + 1][column]
                    )
            for row in range(index + 2, n):
                for column in range(row + 1, n):
                    a[column][row] = a[row][column]
                a[row][index] = 0
                a[row][index + 1] = 0
                a[index][row] = 0
                a[index + 1][row] = 0
        index += 2
    return n_pos, n_neg, n_zero


def _diagonal_inertia(
    diagonal: list[Any],
    *,
    sign: Callable[[Any], int],
    checkpoint: _ExecutionCheckpoint,
) -> tuple[int, int, int]:
    """Count a recognized diagonal source without cubic elimination."""

    n_positive = n_negative = n_zero = 0
    for value in diagonal:
        checkpoint("during exact diagonal sign isolation")
        value_sign = sign(value)
        n_positive += value_sign > 0
        n_negative += value_sign < 0
        n_zero += value_sign == 0
    return n_positive, n_negative, n_zero


def _compute_inertia(
    matrix: ExactRealMatrix,
    *,
    admission: _InertiaExecutionPlan,
    recognized_field: RecognizedRealSimpleNumberField | None = None,
    execution_checkpoint: _ExecutionCheckpoint = _require_inertia_execution_active,
) -> InertiaResult:
    """Compute Sylvester inertia over QQ or one exact real simple field."""

    if admission.order != len(matrix.entries) or admission.algebraic != isinstance(
        matrix, EmbeddedRealSimpleNumberFieldMatrix
    ):
        raise RuntimeError("the inertia plan does not match its admitted source")
    if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
        try:
            execution_checkpoint("before exact number-field recognition")
            if recognized_field is None:
                recognized = recognize_real_simple_number_field(matrix.embedding)
            else:
                recognized = recognized_field
                if recognized.embedding != matrix.embedding:
                    raise EmbeddedNumberFieldRecognitionError(
                        "embedding_mismatch",
                        "the recognized field must use the matrix's selected embedding",
                    )
            execution_checkpoint("after exact number-field recognition")
            source = domain_matrix_from_embedded(matrix, recognized).rep.to_ddm()
            execution_checkpoint("after exact number-field matrix conversion")
            source_rows = [list(row) for row in source]
            if admission.regime == "DIAGONAL":
                n_pos, n_neg, n_zero = _diagonal_inertia(
                    [source_rows[index][index] for index in range(len(source_rows))],
                    sign=lambda value: field_element_sign(value, recognized),
                    checkpoint=execution_checkpoint,
                )
            else:
                n_pos, n_neg, n_zero = _symmetric_algebraic_inertia(
                    source_rows,
                    sign=lambda value: field_element_sign(value, recognized),
                    checkpoint=execution_checkpoint,
                )
        except EmbeddedNumberFieldRecognitionError as exc:
            raise OperationDomainValidationError(
                location=("matrix", "embedding"),
                code=f"matrix.{exc.reason}",
                message=str(exc),
            ) from exc
    else:
        rational_source = [
            [entry.as_fraction() for entry in row] for row in matrix.entries
        ]
        if admission.regime == "DIAGONAL":
            n_pos, n_neg, n_zero = _diagonal_inertia(
                [
                    rational_source[index][index]
                    for index in range(len(rational_source))
                ],
                sign=lambda value: 1 if value > 0 else -1 if value < 0 else 0,
                checkpoint=execution_checkpoint,
            )
        else:
            n_pos, n_neg, n_zero = _symmetric_inertia(
                rational_source,
                checkpoint=execution_checkpoint,
            )
    execution_checkpoint("after exact inertia computation")
    result = InertiaResult._from_kernel(
        matrix=matrix,
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
    )
    execution_checkpoint("after exact inertia result construction")
    return result


def compute_inertia(matrix: ExactRealMatrix) -> InertiaResult:
    """Compute Sylvester inertia over QQ or one exact real simple field."""

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + _INERTIA_WALL_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    try:
        _require_inertia_execution_active("before exact inertia admission")
        admission = _admit_inertia(matrix)
        _require_inertia_execution_active("after exact inertia admission")
        return _compute_inertia(matrix, admission=admission)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("matrix",), code=exc.type, message=exc.message()
        ) from exc


def check_farkas_certificate(
    constraint_matrix: tuple[tuple[CanonicalRational, ...], ...],
    rhs_vector: tuple[CanonicalRational, ...],
    multipliers: tuple[CanonicalRational, ...],
) -> FarkasCertificateResult:
    """Check a rational Farkas infeasibility certificate.

    Given system Ax <= b and multiplier vector y >= 0, the certificate is
    valid if y^T A = 0 and y^T b < 0.
    """
    n_constraints = len(constraint_matrix)
    if not n_constraints or any(not row for row in constraint_matrix):
        raise OperationDomainValidationError(
            location=("constraint_matrix",),
            code="matrix.shape_mismatch",
            message="constraint matrix must have positive dimensions",
        )
    width = len(constraint_matrix[0])
    if any(len(row) != width for row in constraint_matrix):
        raise OperationDomainValidationError(
            location=("constraint_matrix",),
            code="matrix.shape_mismatch",
            message="constraint matrix must be rectangular",
        )
    if len(rhs_vector) != n_constraints or len(multipliers) != n_constraints:
        raise OperationDomainValidationError(
            location=("rhs_vector", "multipliers"),
            code="matrix.shape_mismatch",
            message="rhs and multiplier lengths must match constraint count",
        )
    y = [multiplier.as_fraction() for multiplier in multipliers]
    matrix_fractions = [
        [entry.as_fraction() for entry in row] for row in constraint_matrix
    ]
    b = [entry.as_fraction() for entry in rhs_vector]

    if any(entry < 0 for entry in y):
        ytb = sum((yi * bi for yi, bi in zip(y, b, strict=True)), Fraction(0))
        return FarkasCertificateResult(
            valid=False,
            y_t_a=(),
            y_t_b=format_canonical_rational(ytb),
            reason="multiplier vector has a negative entry",
        )

    n_vars = len(matrix_fractions[0])
    yta = [Fraction(0)] * n_vars
    for i, yi in enumerate(y):
        for j in range(n_vars):
            yta[j] += yi * matrix_fractions[i][j]
    ytb = sum((yi * bi for yi, bi in zip(y, b, strict=True)), Fraction(0))
    yta_str = tuple(format_canonical_rational(value) for value in yta)

    if all(value == 0 for value in yta) and ytb < 0:
        return FarkasCertificateResult(
            valid=True,
            y_t_a=yta_str,
            y_t_b=format_canonical_rational(ytb),
            reason="y^T A = 0 and y^T b < 0",
        )
    reasons = []
    if any(value != 0 for value in yta):
        reasons.append("y^T A != 0")
    if ytb >= 0:
        reasons.append("y^T b >= 0")
    return FarkasCertificateResult(
        valid=False,
        y_t_a=yta_str,
        y_t_b=format_canonical_rational(ytb),
        reason="; ".join(reasons) if reasons else "unknown",
    )


__all__ = [
    "check_farkas_certificate",
    "check_rational_spectrum_claim",
    "compute_inertia",
]
