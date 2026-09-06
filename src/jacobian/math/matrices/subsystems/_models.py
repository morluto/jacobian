"""Wire contracts for exact subsystem-aware matrix operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.subsystems.values import (
    MAX_SUBSYSTEM_DIMENSION,
    MAX_SUBSYSTEM_FACTORS,
    FactorizedHermitianMatrix,
    partial_trace_measured_entries,
)

MAX_KRONECKER_RESULT_COMPONENT_DIGITS = 256
MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS = 4_098
MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS = 4 * MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS
MAX_PSD_DIFFERENCE_COMPONENT_DIGITS = 513


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

    Request admission derives the exact reduced entries first and admits
    those measured components rather than a
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


def _require_traceable_factors(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> None:
    if len(set(traced_factor_labels)) != len(traced_factor_labels):
        raise _validation_error(
            "status_mismatch", "traced subsystem labels must be unique"
        )
    labels = tuple(factor.label for factor in matrix.factors)
    if not set(traced_factor_labels) <= set(labels):
        raise _validation_error(
            "invariant_mismatch",
            "each traced subsystem label must occur in matrix.factors",
        )
    expected_order = tuple(label for label in labels if label in traced_factor_labels)
    if traced_factor_labels != expected_order:
        raise _validation_error(
            "invariant_mismatch",
            "traced subsystem labels must follow source factor order",
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
    fits. The result is bounded by the retained operands, reduced difference
    components, and witness dimension rather than by its JSON encoding.
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


class SubsystemKroneckerProductRequest(StrictModel):
    """Two factorized rational Hermitian matrices for one product."""

    left: FactorizedHermitianMatrix = Field(
        description=(
            "First operand; no fixed per-operand digit ceiling applies. "
            "Admission evaluates the exact product coefficients and admits "
            "the pair when every numerator and denominator stays within the "
            f"{MAX_KRONECKER_RESULT_COMPONENT_DIGITS}-digit product component envelope."
        ),
    )
    right: FactorizedHermitianMatrix = Field(
        description=(
            "Second operand; no fixed per-operand digit ceiling applies. "
            "Admission evaluates the exact product coefficients and admits "
            "the pair when every numerator and denominator stays within the "
            f"{MAX_KRONECKER_RESULT_COMPONENT_DIGITS}-digit product component envelope."
        ),
    )


class SubsystemKroneckerProductResult(StrictModel):
    """The factorized exact product of two rational Hermitian matrices."""

    left: FactorizedHermitianMatrix
    right: FactorizedHermitianMatrix
    product: FactorizedHermitianMatrix

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: FactorizedHermitianMatrix,
        right: FactorizedHermitianMatrix,
        product: FactorizedHermitianMatrix,
    ) -> Self:
        return cls.model_construct(left=left, right=right, product=product)


class SubsystemPartialTraceRequest(StrictModel):
    """One factorized matrix and the named factors to trace out."""

    matrix: FactorizedHermitianMatrix = Field(
        description=(
            "Source operand; no fixed per-operand digit ceiling applies. "
            "Admission measures contraction intermediates, cancelling "
            f"equal-denominator terms first, against the {MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS}-digit work "
            f"envelope, admits reduced coefficients within the {MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS}-digit "
            "result-component envelope."
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
        _require_traceable_factors(self.matrix, self.traced_factor_labels)
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_matrix: FactorizedHermitianMatrix,
        traced_factor_labels: tuple[str, ...],
        reduced_matrix: FactorizedHermitianMatrix,
    ) -> Self:
        """Construct a result after the owner-local trace kernel established it."""

        return cls.model_construct(
            source_matrix=source_matrix,
            traced_factor_labels=traced_factor_labels,
            reduced_matrix=reduced_matrix,
        )


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
        if not any(value.num != 0 for value in self.vector):
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
            "dimension-scaled retained witness bound, not a fixed per-operand ceiling. "
            "Identical operands measure the zero matrix and admit trivially."
        ),
    )
    right: FactorizedHermitianMatrix = Field(
        description=(
            "Second operand; admission couples both operands through the "
            "measured right-minus-left component bound (513 digits), the "
            "dimension-scaled retained witness bound, not a fixed per-operand ceiling. "
            "Identical operands measure the zero matrix and admit trivially."
        ),
    )


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
        dimension = len(self.left.matrix.entries)
        if (
            self.inertia.n_positive + self.inertia.n_negative + self.inertia.n_zero
            != dimension
        ):
            raise _validation_error(
                "shape_mismatch",
                "inertia counts must sum to the source matrix dimension",
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
        if (
            self.negative_witness is not None
            and len(self.negative_witness.vector) != dimension
        ):
            raise _validation_error(
                "shape_mismatch",
                "negative witness length must equal the matrix dimension",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: FactorizedHermitianMatrix,
        right: FactorizedHermitianMatrix,
        difference: FactorizedHermitianMatrix,
        inertia: PsdInertia,
        is_less_or_equal: bool,
        negative_witness: NegativeQuadraticWitness | None,
    ) -> Self:
        """Construct a result after the owner-local PSD kernel established it."""

        return cls.model_construct(
            left=left,
            right=right,
            difference=difference,
            inertia=inertia,
            is_less_or_equal=is_less_or_equal,
            negative_witness=negative_witness,
        )


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
