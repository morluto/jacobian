"""Wire contracts for exact subsystem-aware matrix operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math.matrices.subsystems.values import (
    MAX_SUBSYSTEM_DIMENSION,
    FactorizedHermitianMatrix,
    partial_trace_entries,
)

MAX_KRONECKER_RESULT_COMPONENT_DIGITS = 256
MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS = 4_098
MAX_PSD_DIFFERENCE_COMPONENT_DIGITS = 513


def _component_digits(matrix: FactorizedHermitianMatrix) -> tuple[int, int]:
    numerators = (
        len(entry.num.lstrip("-")) for row in matrix.matrix.entries for entry in row
    )
    denominators = (len(entry.den) for row in matrix.matrix.entries for entry in row)
    return max(numerators, default=1), max(denominators, default=1)


def _trace_component_digit_bound(
    matrix: FactorizedHermitianMatrix,
    *,
    traced_factor_labels: tuple[str, ...],
) -> int:
    """Bound one trace coordinate after a common-denominator exact sum."""

    numerator_digits, denominator_digits = _component_digits(matrix)
    trace_dimension = 1
    for factor in matrix.factors:
        if factor.label in traced_factor_labels:
            trace_dimension *= factor.dimension
    # A common denominator is the product of at most trace_dimension input
    # denominators.  Each numerator has one input numerator and the remaining
    # denominator factors; summing them adds at most decimal digits(trace_dimension).
    return max(numerator_digits, denominator_digits) * trace_dimension + len(
        str(trace_dimension)
    )


def _psd_witness_digit_bound(
    matrix: FactorizedHermitianMatrix,
    *,
    difference_component_digits: int,
) -> int:
    """Conservatively bound the rational congruence witness coordinates.

    A common denominator for an ``n`` by ``n`` difference is the product of
    at most ``n²`` input denominators.  Symmetric elimination transports one
    direction through ratios of bounded minors; charging two such scaled
    minors and their quadratic evaluation gives this safe integer-digit cap.
    """

    dimension = len(matrix.matrix.entries)
    return 3 * dimension * dimension * difference_component_digits


class SubsystemKroneckerProductRequest(StrictModel):
    """Two factorized rational Hermitian matrices for one product."""

    left: FactorizedHermitianMatrix
    right: FactorizedHermitianMatrix

    @model_validator(mode="after")
    def require_product_envelope(self) -> Self:
        left_numerator, left_denominator = _component_digits(self.left)
        right_numerator, right_denominator = _component_digits(self.right)
        if (
            max(
                left_numerator + right_numerator,
                left_denominator + right_denominator,
            )
            > MAX_KRONECKER_RESULT_COMPONENT_DIGITS
        ):
            raise ValueError(
                "Kronecker product coefficient growth exceeds the "
                f"{MAX_KRONECKER_RESULT_COMPONENT_DIGITS}-digit result bound"
            )
        product_dimension = len(self.left.matrix.entries) * len(
            self.right.matrix.entries
        )
        if product_dimension > MAX_SUBSYSTEM_DIMENSION:
            raise ValueError(
                "Kronecker product dimension exceeds the "
                f"{MAX_SUBSYSTEM_DIMENSION} bound"
            )
        labels = (*self.left.factors, *self.right.factors)
        if len(labels) > 4:
            raise ValueError("Kronecker product exceeds the subsystem-factor bound")
        if len({factor.label for factor in labels}) != len(labels):
            raise ValueError("Kronecker product subsystem labels must remain unique")
        return self


class SubsystemKroneckerProductResult(StrictModel):
    """The factorized exact product of two rational Hermitian matrices."""

    product: FactorizedHermitianMatrix


class SubsystemPartialTraceRequest(StrictModel):
    """One factorized matrix and the named factors to trace out."""

    matrix: FactorizedHermitianMatrix
    traced_factor_labels: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Distinct subsystem labels to trace out; every label must occur in "
            "matrix.factors, and remaining factors retain their source order."
        ),
    )

    @model_validator(mode="after")
    def require_traceable_factors(self) -> Self:
        if len(set(self.traced_factor_labels)) != len(self.traced_factor_labels):
            raise ValueError("traced subsystem labels must be unique")
        labels = tuple(factor.label for factor in self.matrix.factors)
        if not set(self.traced_factor_labels) <= set(labels):
            raise ValueError("each traced subsystem label must occur in matrix.factors")
        expected_order = tuple(
            label for label in labels if label in self.traced_factor_labels
        )
        if self.traced_factor_labels != expected_order:
            raise ValueError("traced subsystem labels must follow source factor order")
        if (
            _trace_component_digit_bound(
                self.matrix,
                traced_factor_labels=self.traced_factor_labels,
            )
            > MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS
        ):
            raise ValueError(
                "partial-trace coefficient growth exceeds the "
                f"{MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS}-digit result bound"
            )
        return self


class SubsystemPartialTraceResult(StrictModel):
    """An exact partial trace retaining its untraced subsystem coordinates."""

    source_matrix: FactorizedHermitianMatrix
    traced_factor_labels: tuple[str, ...] = Field(min_length=1)
    reduced_matrix: FactorizedHermitianMatrix

    @model_validator(mode="after")
    def require_trace_result_axes(self) -> Self:
        source_labels = tuple(factor.label for factor in self.source_matrix.factors)
        if len(set(self.traced_factor_labels)) != len(self.traced_factor_labels):
            raise ValueError("traced subsystem labels must be unique")
        if not set(self.traced_factor_labels) <= set(source_labels):
            raise ValueError("traced subsystem labels must occur in the source matrix")
        expected_order = tuple(
            label for label in source_labels if label in self.traced_factor_labels
        )
        if self.traced_factor_labels != expected_order:
            raise ValueError("traced subsystem labels must follow source factor order")
        expected = tuple(
            factor
            for factor in self.source_matrix.factors
            if factor.label not in self.traced_factor_labels
        )
        if self.reduced_matrix.factors != expected:
            raise ValueError(
                "partial trace must retain untraced factors in source order"
            )
        expected_entries = partial_trace_entries(
            self.source_matrix,
            self.traced_factor_labels,
        )
        actual_entries = tuple(
            tuple(entry.as_fraction() for entry in row)
            for row in self.reduced_matrix.matrix.entries
        )
        if actual_entries != expected_entries:
            raise ValueError("partial trace entries must replay against the source")
        return self


class PsdInertia(StrictModel):
    """Exact inertia of the difference in a rational Loewner-order decision."""

    n_positive: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)
    n_negative: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)
    n_zero: int = Field(ge=0, le=MAX_SUBSYSTEM_DIMENSION)

    @model_validator(mode="after")
    def require_order(self) -> Self:
        if self.n_positive + self.n_negative + self.n_zero > MAX_SUBSYSTEM_DIMENSION:
            raise ValueError("inertia counts exceed the subsystem matrix dimension")
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
            raise ValueError("negative quadratic witness vector must be nonzero")
        if self.quadratic_value.as_fraction() >= Fraction(0):
            raise ValueError("negative quadratic witness value must be negative")
        return self


class PsdOrderRequest(StrictModel):
    """Decide whether ``left <= right`` for one factorized rational basis."""

    left: FactorizedHermitianMatrix
    right: FactorizedHermitianMatrix

    @model_validator(mode="after")
    def require_common_axis_bound_source(self) -> Self:
        left_numerator, left_denominator = _component_digits(self.left)
        right_numerator, right_denominator = _component_digits(self.right)
        if self.left.factors != self.right.factors:
            raise ValueError(
                "PSD order requires exactly equal subsystem labels, dimensions, and "
                "basis linearization"
            )
        difference_component_digits = max(
            left_numerator + right_denominator + 1,
            right_numerator + left_denominator + 1,
            left_denominator + right_denominator,
        )
        if difference_component_digits > MAX_PSD_DIFFERENCE_COMPONENT_DIGITS:
            raise ValueError(
                "PSD-order difference growth exceeds the "
                f"{MAX_PSD_DIFFERENCE_COMPONENT_DIGITS}-digit result bound"
            )
        if (
            _psd_witness_digit_bound(
                self.left,
                difference_component_digits=difference_component_digits,
            )
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise ValueError(
                "PSD-order witness growth exceeds the canonical rational component bound"
            )
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
            raise ValueError(
                "PSD-order result matrices must share exactly one axis bound"
            )
        dimension = len(self.left.matrix.entries)
        if (
            self.inertia.n_positive + self.inertia.n_negative + self.inertia.n_zero
            != dimension
        ):
            raise ValueError("inertia counts must sum to the source matrix dimension")
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
            raise ValueError("PSD-order difference must equal right minus left exactly")
        expected_inertia = symmetric_inertia(actual_difference)  # type: ignore[arg-type]
        if (
            self.inertia.n_positive,
            self.inertia.n_negative,
            self.inertia.n_zero,
        ) != expected_inertia:
            raise ValueError("PSD-order inertia must replay against the difference")
        expected_order = self.inertia.n_negative == 0
        if self.is_less_or_equal != expected_order:
            raise ValueError("PSD-order decision must agree with the negative inertia")
        if self.is_less_or_equal and self.negative_witness is not None:
            raise ValueError("a PSD-order result must not carry a negative witness")
        if not self.is_less_or_equal and self.negative_witness is None:
            raise ValueError(
                "a non-PSD difference requires a negative quadratic witness"
            )
        if self.negative_witness is not None:
            vector = tuple(
                value.as_fraction() for value in self.negative_witness.vector
            )
            if len(vector) != dimension:
                raise ValueError(
                    "negative witness length must equal the matrix dimension"
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
                raise ValueError(
                    "negative witness value must replay against the difference"
                )
        return self


__all__ = [
    "MAX_KRONECKER_RESULT_COMPONENT_DIGITS",
    "MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS",
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
