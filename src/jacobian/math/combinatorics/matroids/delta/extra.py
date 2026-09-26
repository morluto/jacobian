from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
)
from jacobian.math.polynomials._models import IntegerPolynomial

MAX_BINARY_GROUND = 8
MAX_BINARY_LABEL_BYTES = 2_048
MAX_TWIST_POLYNOMIAL_STATES = 4_096
MAX_TWIST_POLYNOMIAL_WORK = 262_144
# Bound native label copying/validation independently of recognition's byte cap.
MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS = 1_000_000
MAX_TWIST_POLYNOMIAL_GROUND = MAX_TWIST_POLYNOMIAL_STATES.bit_length() - 1
MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES = MAX_TWIST_POLYNOMIAL_GROUND + 1
MAX_TWIST_POLYNOMIAL_SOURCE_ROWS = MAX_DELTA_MEMBERSHIPS + 1
MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS = len(str(MAX_TWIST_POLYNOMIAL_STATES))


class DeltaMatroidDualRequest(StrictModel):
    delta_matroid: FiniteDeltaMatroid


class DeltaMatroidDualResult(StrictModel):
    source: FiniteDeltaMatroid
    dual: FiniteDeltaMatroid


class DeltaMatroidMinorRequest(StrictModel):
    delta_matroid: FiniteDeltaMatroid
    delete: tuple[int, ...] = ()
    contract: tuple[int, ...] = ()

    @model_validator(mode="after")
    def axes(self) -> Self:
        n = len(self.delta_matroid.ground)
        if self.delete != tuple(sorted(set(self.delete))) or self.contract != tuple(
            sorted(set(self.contract))
        ):
            raise PydanticCustomError(
                "delta_matroid.minor_axis", "minor indices must be sorted and distinct"
            )
        if set(self.delete) & set(self.contract) or any(
            i < 0 or i >= n for i in (*self.delete, *self.contract)
        ):
            raise PydanticCustomError(
                "delta_matroid.minor_axis",
                "minor indices must be disjoint and in range",
            )
        return self


class DeltaMatroidMinorResult(StrictModel):
    source: FiniteDeltaMatroid
    delete: tuple[int, ...]
    contract: tuple[int, ...]
    minor: FiniteDeltaMatroid


class BinarySymmetricMatrix(StrictModel):
    ground: tuple[str, ...] = Field(max_length=MAX_BINARY_GROUND)
    entries: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.ground)
        if n > MAX_BINARY_GROUND:
            raise PydanticCustomError(
                "delta_matroid.binary_ground",
                f"binary principal-minor ground is limited to {MAX_BINARY_GROUND} labels",
            )
        if len(set(self.ground)) != n:
            raise PydanticCustomError(
                "delta_matroid.binary_labels", "binary ground labels must be unique"
            )
        try:
            label_bytes = sum(len(label.encode("utf-8")) for label in self.ground)
        except UnicodeEncodeError:
            raise PydanticCustomError(
                "delta_matroid.binary_labels", "binary labels must be UTF-8"
            ) from None
        if label_bytes > MAX_BINARY_LABEL_BYTES:
            raise PydanticCustomError(
                "delta_matroid.binary_labels", "binary labels exceed the byte envelope"
            )
        if len(self.entries) != n or any(len(r) != n for r in self.entries):
            raise PydanticCustomError(
                "delta_matroid.binary_shape",
                "binary matrix must be square on the ground axis",
            )
        if any(x not in (0, 1) for r in self.entries for x in r):
            raise PydanticCustomError(
                "delta_matroid.binary_entry", "entries must be GF(2) bits"
            )
        if any(
            self.entries[i][j] != self.entries[j][i] for i in range(n) for j in range(n)
        ):
            raise PydanticCustomError(
                "delta_matroid.binary_symmetric", "matrix must be symmetric"
            )
        return self


class BinaryMatrixRequest(StrictModel):
    matrix: BinarySymmetricMatrix


class BinaryMatrixResult(StrictModel):
    matrix: BinarySymmetricMatrix
    delta_matroid: FiniteDeltaMatroid

    @model_validator(mode="after")
    def _source_binding(self) -> Self:
        if self.delta_matroid.ground != self.matrix.ground:
            raise PydanticCustomError(
                "delta_matroid.binary_result_ground",
                "delta matroid ground must equal the source matrix ground",
            )
        return self


class DeltaMatroidTwistPolynomialRequest(StrictModel):
    """Compute the generating function of widths across all twists."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Return coefficients indexed by width for the exact polynomial "
                "sum over all A subset E of z^width(D*A), plus the complete "
                "width histogram. The polynomial uses descending-degree "
                "IntegerPolynomial coefficients. Admission permits at most "
                f"{MAX_TWIST_POLYNOMIAL_GROUND} ground elements, "
                f"{MAX_TWIST_POLYNOMIAL_STATES} twist masks, and "
                f"{MAX_TWIST_POLYNOMIAL_WORK} mask-feasible-set evaluations, "
                f"and {MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS} aggregate ground-label "
                "codepoints for native allocation and source validation. "
                f"The result has at most {MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES} "
                "histogram entries and coefficients with at most "
                f"{MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS} decimal digits. "
                "The recognition operation's 2,048-byte label envelope does "
                "not apply: labels do not affect the mask sweep."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_TWIST_POLYNOMIAL_GROUND,
                "max_ground_label_codepoints": MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS,
                "max_twist_masks": MAX_TWIST_POLYNOMIAL_STATES,
                "max_mask_feasible_set_evaluations": MAX_TWIST_POLYNOMIAL_WORK,
                "max_histogram_entries": MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES,
                "max_polynomial_coefficient_digits": MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS,
                "max_source_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_source_feasible_rows": MAX_TWIST_POLYNOMIAL_SOURCE_ROWS,
                "max_source_exchange_candidates": MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete canonical finite delta-matroid; every twist subset is "
            "included exactly once after subset-state, mask/feasible evaluation, "
            "source membership, and symmetric-exchange admission."
        )
    )


class DeltaMatroidTwistPolynomialResult(StrictModel):
    """The twist polynomial and its complete width histogram."""

    ground: tuple[str, ...] = Field(max_length=MAX_TWIST_POLYNOMIAL_GROUND)
    coefficients_by_width: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES
    )
    polynomial: IntegerPolynomial = Field(
        description=(
            "Exact integer polynomial in the formal variable z, stored in "
            "descending-degree order. The coefficient of z^k is the number "
            "of twists of width k."
        )
    )

    @model_validator(mode="after")
    def complete_width_axis(self) -> Self:
        # Check retained-label allocation before hashing the authored ground.
        # The operation's source preflight admits this same native budget.
        if sum(map(len, self.ground)) > MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS:
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_labels",
                "ground labels exceed the admitted native codepoint budget",
            )
        try:
            for label in self.ground:
                label.encode("utf-8")
        except UnicodeEncodeError:
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_utf8",
                "ground labels must be UTF-8-representable",
            ) from None
        if len(set(self.ground)) != len(self.ground):
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_ground",
                "delta-matroid ground labels must be unique",
            )
        if len(self.coefficients_by_width) != len(self.ground) + 1 or any(
            type(coefficient) is not int or coefficient < 0
            for coefficient in self.coefficients_by_width
        ):
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_shape",
                "polynomial must contain one nonnegative coefficient for each width 0 through |E|",
            )
        if sum(self.coefficients_by_width) != 1 << len(self.ground):
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_total",
                "twist-width coefficients must sum to the number of ground subsets",
            )
        canonical_coefficients = tuple(reversed(self.coefficients_by_width))
        while len(canonical_coefficients) > 1 and canonical_coefficients[0] == 0:
            canonical_coefficients = canonical_coefficients[1:]
        if tuple(self.polynomial.coefficients) != canonical_coefficients:
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_coefficients",
                "integer polynomial coefficients must equal the canonical width histogram",
            )
        return self


__all__ = [
    "MAX_BINARY_GROUND",
    "MAX_BINARY_LABEL_BYTES",
    "MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS",
    "MAX_TWIST_POLYNOMIAL_GROUND",
    "MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES",
    "MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS",
    "MAX_TWIST_POLYNOMIAL_SOURCE_ROWS",
    "MAX_TWIST_POLYNOMIAL_STATES",
    "MAX_TWIST_POLYNOMIAL_WORK",
    "BinaryMatrixRequest",
    "BinaryMatrixResult",
    "BinarySymmetricMatrix",
    "DeltaMatroidDualRequest",
    "DeltaMatroidDualResult",
    "DeltaMatroidMinorRequest",
    "DeltaMatroidMinorResult",
    "DeltaMatroidTwistPolynomialRequest",
    "DeltaMatroidTwistPolynomialResult",
]
