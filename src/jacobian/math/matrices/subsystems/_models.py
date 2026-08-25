"""Wire contracts for exact subsystem-aware matrix operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math.matrices.subsystems.values import (
    MAX_SUBSYSTEM_DIMENSION,
    MAX_SUBSYSTEM_FACTORS,
    FactorizedHermitianMatrix,
    partial_trace_measured_entries,
)
from jacobian.math.matrices.values import RationalMatrix

MAX_KRONECKER_RESULT_COMPONENT_DIGITS = 256
MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS = 4_098
MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS = 4 * MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS
MAX_PSD_DIFFERENCE_COMPONENT_DIGITS = 513
_PSD_RESULT_ENVELOPE_RESERVE_BYTES = 4_096
_PSD_WITNESS_COMPONENT_RESERVE_BYTES = 2 * MAX_CANONICAL_RATIONAL_DIGITS + 32
_TRACE_RESULT_ENVELOPE_RESERVE_BYTES = 4_096


def _fraction_component_digits(value: Fraction) -> tuple[int, int]:
    """Count one exact fraction's signed-numerator and denominator digits.

    Formatting goes through the canonical chunked formatter so measured
    intermediates wider than Python's integer-string conversion limit stay
    countable instead of raising a host exception.
    """

    return (
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _entry_fractions(
    matrix: FactorizedHermitianMatrix,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.matrix.entries
    )


def _require_trace_work_envelope(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Charge one contraction's actual folded intermediates against the work bound.

    ``Fraction`` folds reduce between additions, and cancellation inside a
    cell can collapse the running value below any aggregate of its input
    widths -- adjacent pairs ``1/d, -1/d`` over distinct denominators never
    widen past one denominator, and cancellation can also arrive late, when
    matching terms of opposite sign sit far apart in the fold order -- so no
    per-input estimate bounds every fold safely.  Admission therefore runs
    the exact kernel itself through :func:`partial_trace_measured_entries`,
    whose per-denominator grouping cancels surviving equal-denominator
    numerators before any cross-denominator addition, and charges the widest
    signed-numerator or denominator width the executed walk reaches.
    Measuring the real arithmetic keeps cancelling folds admissible while any
    genuinely growing intermediate is rejected fail-closed.
    """

    entries, peak_component_digits = partial_trace_measured_entries(
        matrix, traced_factor_labels
    )
    if peak_component_digits > MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "partial-trace contraction work exceeds the "
            f"{MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS}-digit intermediate bound",
        )
    return entries


def _require_trace_result_envelope(
    expected_entries: tuple[tuple[Fraction, ...], ...],
) -> None:
    """Admit one trace whose exact reduced coefficients fit the result bound.

    Both request validation and result replay derive the exact reduced
    entries first and admit those measured components rather than a
    per-input estimate, so an emitted value re-enters its own consumer
    unchanged unless its next exact result genuinely exceeds the bound, and
    a transported or authored source can never drive the common-denominator
    sums outside the admitted envelope.
    """

    measured = max(
        (
            max(_fraction_component_digits(entry))
            for row in expected_entries
            for entry in row
        ),
        default=1,
    )
    if measured > MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "partial-trace coefficient growth exceeds the "
            f"{MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS}-digit result bound",
        )


def _require_trace_transport_envelope(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
    expected_entries: tuple[tuple[Fraction, ...], ...],
) -> None:
    """Reserve the serialized result's share of the canonical output budget.

    The result retains its source matrix and adds the reduced matrix, so a
    source near the transport limit can fit every coefficient envelope and
    still overflow canonical output encoding outside the request-validation
    handler.  Measuring the exact encoded result -- plus one envelope
    reserve -- keeps every accepted call returning its typed result.
    """

    output_limit = CanonicalLimits().max_output_bytes
    try:
        reduced = FactorizedHermitianMatrix(
            matrix=RationalMatrix(
                entries=tuple(
                    tuple(CanonicalRational.from_fraction(entry) for entry in row)
                    for row in expected_entries
                )
            ),
            factors=tuple(
                factor
                for factor in matrix.factors
                if factor.label not in traced_factor_labels
            ),
        )
        result_bytes = (
            len(encode_strict_json(matrix.model_dump(mode="json")))
            + len(encode_strict_json(reduced.model_dump(mode="json")))
            + _TRACE_RESULT_ENVELOPE_RESERVE_BYTES
        )
    except CanonicalizationError:
        result_bytes = output_limit + 1
    if result_bytes > output_limit:
        raise _validation_error(
            "budget_exceeded",
            "the partial-trace result retains its source matrix and would "
            f"exceed the {output_limit}-byte canonical output limit; "
            "use smaller or sparser operands",
        )


def _psd_witness_digit_bound(
    matrix: FactorizedHermitianMatrix,
    *,
    difference_component_digits: int,
) -> int:
    """Conservatively bound the rational congruence witness coordinates.

    Symmetric elimination transports one direction through ratios of bounded
    minors of the measured reduced difference; charging two such scaled minors
    and their quadratic evaluation per measured difference-component digit
    gives this safe integer-digit cap.
    """

    dimension = len(matrix.matrix.entries)
    return 3 * dimension * dimension * difference_component_digits


def _require_psd_pair_admission(
    left: FactorizedHermitianMatrix,
    right: FactorizedHermitianMatrix,
) -> None:
    """Admit one ordered pair through the coupled PSD digit envelopes.

    The exact reduced right-minus-left components are measured before any
    witness bound is applied, so identical operands -- whose difference is
    the zero matrix and admits no negative witness -- and nearly equal
    operands whose reduced difference stays tiny admit trivially, while no
    unreduced cross-term estimate can reject a pair whose actual difference
    fits.  Because the source-bound result echoes both operands and their
    difference, admission also reserves the serialized transport budget --
    measured exactly, plus a component-capped witness allowance and one
    result envelope -- so every accepted request returns its typed result
    instead of overflowing canonical output encoding.
    """

    if left.factors != right.factors:
        raise _validation_error(
            "budget_exceeded",
            "PSD order requires exactly equal subsystem labels, dimensions, and "
            "basis linearization",
        )
    difference_rows = tuple(
        tuple(
            right_entry - left_entry
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(
            _entry_fractions(left), _entry_fractions(right), strict=True
        )
    )
    difference_component_digits = max(
        (
            max(_fraction_component_digits(entry))
            for row in difference_rows
            for entry in row
        ),
        default=1,
    )
    if difference_component_digits > MAX_PSD_DIFFERENCE_COMPONENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "PSD-order difference growth exceeds the "
            f"{MAX_PSD_DIFFERENCE_COMPONENT_DIGITS}-digit result bound",
        )
    if (
        _psd_witness_digit_bound(
            left,
            difference_component_digits=difference_component_digits,
        )
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _validation_error(
            "budget_exceeded",
            "PSD-order witness growth exceeds the canonical rational component bound",
        )
    output_limit = CanonicalLimits().max_output_bytes
    witness_reserve = (
        0
        if all(entry == 0 for row in difference_rows for entry in row)
        else (len(left.matrix.entries) + 1) * _PSD_WITNESS_COMPONENT_RESERVE_BYTES
    )
    try:
        result_bytes = (
            len(encode_strict_json(left.model_dump(mode="json")))
            + len(encode_strict_json(right.model_dump(mode="json")))
            + len(
                encode_strict_json(
                    FactorizedHermitianMatrix(
                        matrix=RationalMatrix(
                            entries=tuple(
                                tuple(
                                    CanonicalRational.from_fraction(entry)
                                    for entry in row
                                )
                                for row in difference_rows
                            )
                        ),
                        factors=left.factors,
                    ).model_dump(mode="json")
                )
            )
            + witness_reserve
            + _PSD_RESULT_ENVELOPE_RESERVE_BYTES
        )
    except CanonicalizationError:
        result_bytes = output_limit + 1
    if result_bytes > output_limit:
        raise _validation_error(
            "budget_exceeded",
            "the PSD-order result retains both operands and their exact "
            f"difference and would exceed the {output_limit}-byte canonical "
            "output limit; use smaller or sparser operands",
        )


class SubsystemKroneckerProductRequest(StrictModel):
    """Two factorized rational Hermitian matrices for one product."""

    left: FactorizedHermitianMatrix = Field(
        description=(
            "First operand; no fixed per-operand digit ceiling applies. "
            "Admission evaluates the exact product coefficients and admits "
            "the pair when every numerator and denominator stays within the "
            "256-digit product component envelope."
        ),
    )
    right: FactorizedHermitianMatrix = Field(
        description=(
            "Second operand; no fixed per-operand digit ceiling applies. "
            "Admission evaluates the exact product coefficients and admits "
            "the pair when every numerator and denominator stays within the "
            "256-digit product component envelope."
        ),
    )

    @model_validator(mode="after")
    def require_product_envelope(self) -> Self:
        product_dimension = len(self.left.matrix.entries) * len(
            self.right.matrix.entries
        )
        if product_dimension > MAX_SUBSYSTEM_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                "Kronecker product dimension exceeds the "
                f"{MAX_SUBSYSTEM_DIMENSION} bound",
            )
        labels = (*self.left.factors, *self.right.factors)
        if len(labels) > 4:
            raise _validation_error(
                "budget_exceeded",
                "Kronecker product exceeds the subsystem-factor bound",
            )
        if len({factor.label for factor in labels}) != len(labels):
            raise _validation_error(
                "budget_exceeded",
                "Kronecker product subsystem labels must remain unique",
            )
        left_entries = _entry_fractions(self.left)
        right_entries = _entry_fractions(self.right)
        for left_row in left_entries:
            for left_entry in left_row:
                for right_row in right_entries:
                    for right_entry in right_row:
                        if (
                            max(_fraction_component_digits(left_entry * right_entry))
                            > MAX_KRONECKER_RESULT_COMPONENT_DIGITS
                        ):
                            raise _validation_error(
                                "budget_exceeded",
                                "Kronecker product coefficient growth exceeds the "
                                f"{MAX_KRONECKER_RESULT_COMPONENT_DIGITS}-digit "
                                "result bound",
                            )
        return self


class SubsystemKroneckerProductResult(StrictModel):
    """The factorized exact product of two rational Hermitian matrices."""

    product: FactorizedHermitianMatrix


class SubsystemPartialTraceRequest(StrictModel):
    """One factorized matrix and the named factors to trace out."""

    matrix: FactorizedHermitianMatrix = Field(
        description=(
            "Source operand; no fixed per-operand digit ceiling applies. "
            "Admission measures contraction intermediates, cancelling "
            "equal-denominator terms first, against the 16392-digit work "
            "envelope, admits reduced coefficients within the 4098-digit "
            "result envelope, and reserves the serialized result's canonical "
            "output budget."
        ),
    )
    traced_factor_labels: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SUBSYSTEM_FACTORS,
        description=(
            "Distinct subsystem labels to trace out; every label must occur in "
            "matrix.factors, and remaining factors retain their source order."
        ),
    )

    @model_validator(mode="after")
    def require_traceable_factors(self) -> Self:
        if len(set(self.traced_factor_labels)) != len(self.traced_factor_labels):
            raise _validation_error(
                "status_mismatch", "traced subsystem labels must be unique"
            )
        labels = tuple(factor.label for factor in self.matrix.factors)
        if not set(self.traced_factor_labels) <= set(labels):
            raise _validation_error(
                "invariant_mismatch",
                "each traced subsystem label must occur in matrix.factors",
            )
        expected_order = tuple(
            label for label in labels if label in self.traced_factor_labels
        )
        if self.traced_factor_labels != expected_order:
            raise _validation_error(
                "invariant_mismatch",
                "traced subsystem labels must follow source factor order",
            )
        expected_entries = _require_trace_work_envelope(
            self.matrix, self.traced_factor_labels
        )
        _require_trace_result_envelope(expected_entries)
        _require_trace_transport_envelope(
            self.matrix, self.traced_factor_labels, expected_entries
        )
        return self


class SubsystemPartialTraceResult(StrictModel):
    """An exact partial trace retaining its untraced subsystem coordinates."""

    source_matrix: FactorizedHermitianMatrix
    traced_factor_labels: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SUBSYSTEM_FACTORS,
    )
    reduced_matrix: FactorizedHermitianMatrix

    @model_validator(mode="after")
    def require_trace_result_axes(self) -> Self:
        source_labels = tuple(factor.label for factor in self.source_matrix.factors)
        if len(set(self.traced_factor_labels)) != len(self.traced_factor_labels):
            raise _validation_error(
                "invariant_mismatch", "traced subsystem labels must be unique"
            )
        if not set(self.traced_factor_labels) <= set(source_labels):
            raise _validation_error(
                "invariant_mismatch",
                "traced subsystem labels must occur in the source matrix",
            )
        expected_order = tuple(
            label for label in source_labels if label in self.traced_factor_labels
        )
        if self.traced_factor_labels != expected_order:
            raise _validation_error(
                "invariant_mismatch",
                "traced subsystem labels must follow source factor order",
            )
        expected_entries = _require_trace_work_envelope(
            self.source_matrix, self.traced_factor_labels
        )
        _require_trace_result_envelope(expected_entries)
        _require_trace_transport_envelope(
            self.source_matrix, self.traced_factor_labels, expected_entries
        )
        expected = tuple(
            factor
            for factor in self.source_matrix.factors
            if factor.label not in self.traced_factor_labels
        )
        if self.reduced_matrix.factors != expected:
            raise _validation_error(
                "shape_mismatch",
                "partial trace must retain untraced factors in source order",
            )
        actual_entries = tuple(
            tuple(entry.as_fraction() for entry in row)
            for row in self.reduced_matrix.matrix.entries
        )
        if actual_entries != expected_entries:
            raise _validation_error(
                "shape_mismatch", "partial trace entries must replay against the source"
            )
        return self


class PsdInertia(StrictModel):
    """Exact inertia of the difference in a rational Loewner-order decision."""

    n_positive: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)
    n_negative: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)
    n_zero: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)

    @model_validator(mode="after")
    def require_order(self) -> Self:
        if self.n_positive + self.n_negative + self.n_zero > MAX_SUBSYSTEM_DIMENSION:
            raise _validation_error(
                "shape_mismatch", "inertia counts exceed the subsystem matrix dimension"
            )
        return self


class NegativeQuadraticWitness(StrictModel):
    """A rational vector proving that an exact symmetric matrix is not PSD."""

    vector: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_SUBSYSTEM_DIMENSION,
    )
    quadratic_value: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero_negative_claim(self) -> Self:
        if not any(value.num != "0" for value in self.vector):
            raise _validation_error(
                "field_mismatch", "negative quadratic witness vector must be nonzero"
            )
        if self.quadratic_value.as_fraction() >= Fraction(0):
            raise _validation_error(
                "budget_exceeded", "negative quadratic witness value must be negative"
            )
        return self


class PsdOrderRequest(StrictModel):
    """Decide whether ``left <= right`` for one factorized rational basis."""

    left: FactorizedHermitianMatrix = Field(
        description=(
            "First operand; admission couples both operands through the "
            "measured right-minus-left component bound (513 digits), the "
            "dimension-scaled witness bound, and the serialized result's "
            "canonical output budget, not a fixed per-operand ceiling. "
            "Identical operands measure the zero matrix and admit trivially."
        ),
    )
    right: FactorizedHermitianMatrix = Field(
        description=(
            "Second operand; admission couples both operands through the "
            "measured right-minus-left component bound (513 digits), the "
            "dimension-scaled witness bound, and the serialized result's "
            "canonical output budget, not a fixed per-operand ceiling. "
            "Identical operands measure the zero matrix and admit trivially."
        ),
    )

    @model_validator(mode="after")
    def require_common_axis_bound_source(self) -> Self:
        _require_psd_pair_admission(self.left, self.right)
        return self


class PsdOrderResult(StrictModel):
    """A source-bound exact decision of whether ``right - left`` is PSD."""

    left: FactorizedHermitianMatrix
    right: FactorizedHermitianMatrix
    difference: FactorizedHermitianMatrix
    inertia: PsdInertia
    is_less_or_equal: bool
    negative_witness: NegativeQuadraticWitness | None = None
    convention: Literal["RIGHT_MINUS_LEFT_POSITIVE_SEMIDEFINITE_OVER_QQ"] = (
        "RIGHT_MINUS_LEFT_POSITIVE_SEMIDEFINITE_OVER_QQ"
    )

    @model_validator(mode="after")
    def require_bound_decision_and_witness(self) -> Self:
        if (
            self.left.factors != self.right.factors
            or self.difference.factors != self.left.factors
        ):
            raise _validation_error(
                "budget_exceeded",
                "PSD-order result matrices must share exactly one axis bound",
            )
        _require_psd_pair_admission(self.left, self.right)
        dimension = len(self.left.matrix.entries)
        if (
            self.inertia.n_positive + self.inertia.n_negative + self.inertia.n_zero
            != dimension
        ):
            raise _validation_error(
                "shape_mismatch",
                "inertia counts must sum to the source matrix dimension",
            )
        expected_difference = tuple(
            tuple(
                right.as_fraction() - left.as_fraction()
                for left, right in zip(left_row, right_row, strict=True)
            )
            for left_row, right_row in zip(
                self.left.matrix.entries,
                self.right.matrix.entries,
                strict=True,
            )
        )
        actual_difference = tuple(
            tuple(entry.as_fraction() for entry in row)
            for row in self.difference.matrix.entries
        )
        if actual_difference != expected_difference:
            raise _validation_error(
                "invariant_mismatch",
                "PSD-order difference must equal right minus left exactly",
            )
        expected_inertia = symmetric_inertia(actual_difference)  # type: ignore[arg-type]
        if (
            self.inertia.n_positive,
            self.inertia.n_negative,
            self.inertia.n_zero,
        ) != expected_inertia:
            raise _validation_error(
                "invariant_mismatch",
                "PSD-order inertia must replay against the difference",
            )
        expected_order = self.inertia.n_negative == 0
        if self.is_less_or_equal != expected_order:
            raise _validation_error(
                "invariant_mismatch",
                "PSD-order decision must agree with the negative inertia",
            )
        if self.is_less_or_equal and self.negative_witness is not None:
            raise _validation_error(
                "shape_mismatch", "a PSD-order result must not carry a negative witness"
            )
        if not self.is_less_or_equal and self.negative_witness is None:
            raise _validation_error(
                "shape_mismatch",
                "a non-PSD difference requires a negative quadratic witness",
            )
        if self.negative_witness is not None:
            vector = tuple(
                value.as_fraction() for value in self.negative_witness.vector
            )
            if len(vector) != dimension:
                raise _validation_error(
                    "shape_mismatch",
                    "negative witness length must equal the matrix dimension",
                )
            quadratic_value = sum(
                (
                    vector[row] * actual_difference[row][column] * vector[column]
                    for row in range(dimension)
                    for column in range(dimension)
                ),
                Fraction(0),
            )
            if quadratic_value != self.negative_witness.quadratic_value.as_fraction():
                raise _validation_error(
                    "budget_exceeded",
                    "negative witness value must replay against the difference",
                )
        return self


__all__ = [
    "MAX_KRONECKER_RESULT_COMPONENT_DIGITS",
    "MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS",
    "MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS",
    "MAX_PSD_DIFFERENCE_COMPONENT_DIGITS",
    "NegativeQuadraticWitness",
    "PsdInertia",
    "PsdOrderRequest",
    "PsdOrderResult",
    "SubsystemKroneckerProductRequest",
    "SubsystemKroneckerProductResult",
    "SubsystemPartialTraceRequest",
    "SubsystemPartialTraceResult",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
