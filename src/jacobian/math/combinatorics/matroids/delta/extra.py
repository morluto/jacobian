from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self, cast

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
)
from jacobian.math.polynomials._models import IntegerPolynomial

MAX_BINARY_GROUND = 8
MAX_BINARY_LABEL_BYTES = 2_048
MAX_BINARY_PRINCIPAL_MINOR_WORK = 250_000
MAX_BINARY_TWIST_STATES = 1 << MAX_BINARY_GROUND
MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS = MAX_BINARY_GROUND * MAX_BINARY_TWIST_STATES // 2
MAX_BINARY_TWIST_TRANSPORT_WORK = MAX_BINARY_TWIST_STATES * (
    1 + 2 * MAX_BINARY_GROUND + MAX_BINARY_GROUND**2
)
MAX_BINARY_TWIST_OUTPUT_CELLS = (
    MAX_BINARY_GROUND**2
    + MAX_BINARY_TWIST_STATES
    + MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS
    + 4 * MAX_BINARY_GROUND
)
MAX_TWIST_WIDTH_STATES = 4_096
MAX_TWIST_WIDTH_WORK = 262_144
MAX_FEASIBLE_SIZE_PROFILE_ENTRIES = 4_096
MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS = 32_768
MAX_DIRECT_SUM_GROUND = 2_048
MAX_DIRECT_SUM_FEASIBLE_PAIRS = 250_000
MAX_TWIST_POLYNOMIAL_STATES = 4_096
MAX_TWIST_POLYNOMIAL_WORK = 262_144
# Bound native label copying/validation independently of recognition's byte cap.
MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS = 1_000_000
MAX_TWIST_POLYNOMIAL_GROUND = MAX_TWIST_POLYNOMIAL_STATES.bit_length() - 1
MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES = MAX_TWIST_POLYNOMIAL_GROUND + 1
MAX_TWIST_POLYNOMIAL_SOURCE_ROWS = MAX_DELTA_MEMBERSHIPS + 1
MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS = len(str(MAX_TWIST_POLYNOMIAL_STATES))


class DeltaMatroidDirectSumRequest(StrictModel):
    """Take the direct sum on disjoint labelled ground sets."""

    left: FiniteDeltaMatroid
    right: FiniteDeltaMatroid

    @model_validator(mode="after")
    def require_disjoint_ground(self) -> Self:
        if set(self.left.ground).intersection(self.right.ground):
            raise PydanticCustomError(
                "delta_matroid.direct_sum_ground_overlap",
                "direct-sum ground labels must be disjoint",
            )
        return self


class DeltaMatroidDirectSumResult(StrictModel):
    """Direct sum and its canonical source-index injections."""

    left: FiniteDeltaMatroid
    right: FiniteDeltaMatroid
    direct_sum: FiniteDeltaMatroid
    left_injection: tuple[int, ...]
    right_injection: tuple[int, ...]

    @model_validator(mode="after")
    def require_source_maps(self) -> Self:
        if self.left_injection != tuple(range(len(self.left.ground))):
            raise PydanticCustomError(
                "delta_matroid.direct_sum_map",
                "left injection must retain source indices",
            )
        offset = len(self.left.ground)
        if self.right_injection != tuple(
            offset + i for i in range(len(self.right.ground))
        ):
            raise PydanticCustomError(
                "delta_matroid.direct_sum_map",
                "right injection must follow the left ground axis",
            )
        if self.direct_sum.ground != self.left.ground + self.right.ground:
            raise PydanticCustomError(
                "delta_matroid.direct_sum_ground",
                "direct-sum ground must concatenate the disjoint source grounds",
            )
        return self


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


BinaryMatrixEntry = Annotated[StrictInt, Field(ge=0, le=1)]
BinaryMatrixRow = Annotated[
    tuple[BinaryMatrixEntry, ...], Field(max_length=MAX_BINARY_GROUND)
]


class BinarySymmetricMatrix(StrictModel):
    ground: tuple[str, ...] = Field(max_length=MAX_BINARY_GROUND)
    entries: tuple[BinaryMatrixRow, ...] = Field(max_length=MAX_BINARY_GROUND)

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
    twist: tuple[StrictInt, ...] = Field(default=(), max_length=MAX_BINARY_GROUND)

    @model_validator(mode="after")
    def _source_binding(self) -> Self:
        if self.delta_matroid.ground != self.matrix.ground:
            raise PydanticCustomError(
                "delta_matroid.binary_result_ground",
                "delta matroid ground must equal the source matrix ground",
            )
        if (
            any(type(index) is not int for index in self.twist)
            or self.twist != tuple(sorted(set(self.twist)))
            or any(
                index < 0 or index >= len(self.matrix.ground) for index in self.twist
            )
        ):
            raise PydanticCustomError(
                "delta_matroid.binary_result_twist",
                "twist indices must be sorted, distinct, and in range",
            )
        return self


class BinaryLoopComplementRequest(StrictModel):
    """Toggle loop complementation on a canonical subset of matrix axes."""

    matrix: BinarySymmetricMatrix
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_subset(self) -> Self:
        if self.subset != tuple(sorted(set(self.subset))) or any(
            type(i) is not int or i < 0 or i >= len(self.matrix.ground)
            for i in self.subset
        ):
            raise PydanticCustomError(
                "delta_matroid.loop_complement_subset",
                "loop-complement indices must be sorted, distinct, and in range",
            )
        return self


class BinaryLoopComplementResult(StrictModel):
    """A diagonal-toggled binary presentation and its principal-minor family."""

    source: BinarySymmetricMatrix
    subset: tuple[int, ...]
    result: BinaryMatrixResult

    @model_validator(mode="after")
    def source_binding(self) -> Self:
        if self.result.matrix.ground != self.source.ground:
            raise PydanticCustomError(
                "delta_matroid.loop_complement_ground",
                "result matrix must retain the source ground axis",
            )
        expected = tuple(
            tuple(bit ^ int(i == j and i in self.subset) for j, bit in enumerate(row))
            for i, row in enumerate(self.source.entries)
        )
        if self.result.matrix.entries != expected:
            raise PydanticCustomError(
                "delta_matroid.loop_complement_matrix",
                "result matrix must toggle exactly the requested diagonal entries",
            )
        return self


class DeltaMatroidTwistWidthProfileRequest(StrictModel):
    """Compute width after twisting by every subset of the ground axis."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the width for every twist of a finite delta-matroid. "
                "Admission permits at most "
                f"{MAX_TWIST_WIDTH_STATES} subset masks and "
                f"{MAX_TWIST_WIDTH_WORK} mask-feasible-set evaluations."
            ),
            "admission_limits": {
                "max_subset_masks": MAX_TWIST_WIDTH_STATES,
                "max_mask_feasible_set_evaluations": MAX_TWIST_WIDTH_WORK,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete canonical finite delta-matroid. Every twist mask is "
            "evaluated only after the complete profile state and work bounds "
            "have passed."
        )
    )


class DeltaMatroidTwistWidthProfile(StrictModel):
    """Widths indexed by the integer mask of the twisting subset."""

    ground: tuple[str, ...]
    widths_by_mask: tuple[int, ...]

    @model_validator(mode="after")
    def complete_mask_axis(self) -> Self:
        expected = 1 << len(self.ground)
        if expected > MAX_TWIST_WIDTH_STATES:
            raise PydanticCustomError(
                "delta_matroid.twist_width_profile_states",
                "profile ground exceeds the complete subset-mask envelope",
            )
        if len(self.widths_by_mask) != expected or any(
            width < 0 or width > len(self.ground) for width in self.widths_by_mask
        ):
            raise PydanticCustomError(
                "delta_matroid.twist_width_profile_shape",
                "profile must contain one nonnegative width for every ground-subset mask",
            )
        return self


class DeltaMatroidTwistWidthProfileResult(StrictModel):
    delta_matroid: FiniteDeltaMatroid
    profile: DeltaMatroidTwistWidthProfile

    @model_validator(mode="after")
    def source_axis(self) -> Self:
        if self.profile.ground != self.delta_matroid.ground:
            raise PydanticCustomError(
                "delta_matroid.twist_width_profile_ground",
                "profile ground must equal the source delta-matroid ground",
            )
        return self


class DeltaMatroidFeasibleSizeProfileRequest(StrictModel):
    """Compute the feasible-set cardinality histogram."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Return the exact number of feasible sets of each size from zero "
                "through the ground-set size. Admission bounds the profile axis "
                f"to {MAX_FEASIBLE_SIZE_PROFILE_ENTRIES} entries and the retained "
                "counts and ground labels to "
                f"{MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS} allocation units."
            ),
            "admission_limits": {
                "max_profile_entries": MAX_FEASIBLE_SIZE_PROFILE_ENTRIES,
                "max_retained_units": MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid


class DeltaMatroidFeasibleSizeProfile(StrictModel):
    """Feasible-set counts indexed by cardinality, retaining the ground axis."""

    ground: tuple[str, ...]
    counts_by_size: tuple[int, ...]

    @model_validator(mode="after")
    def complete_size_axis(self) -> Self:
        if len(self.counts_by_size) != len(self.ground) + 1 or any(
            count < 0 for count in self.counts_by_size
        ):
            raise PydanticCustomError(
                "delta_matroid.feasible_size_profile_shape",
                "profile must contain one nonnegative count for each size 0 through |E|",
            )
        return self


class DeltaMatroidTwistPolynomialRequest(StrictModel):
    """Compute the generating function of widths across all twists."""

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_delta_matroid(cls, value: object) -> object:
        """Reject oversized nested axes before Pydantic builds their value model."""

        if type(value) is not dict:
            return value
        raw_value = cast(dict[str, object], value)
        delta_value = raw_value.get("delta_matroid")
        if type(delta_value) is not dict:
            return value
        delta = cast(dict[str, object], delta_value)
        ground = delta.get("ground")
        if isinstance(ground, (tuple, list)):
            ground_items = cast(Sequence[object], ground)
            if len(ground_items) > MAX_TWIST_POLYNOMIAL_GROUND:
                raise PydanticCustomError(
                    "delta_matroid.twist_polynomial_work",
                    "complete twist polynomial exceeds its subset-state envelope",
                )
            if all(type(label) is str for label in ground_items):
                ground_labels = cast(Sequence[str], ground_items)
                if sum(len(label) for label in ground_labels) > (
                    MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS
                ):
                    raise PydanticCustomError(
                        "delta_matroid.twist_polynomial_labels",
                        "ground labels exceed the admitted native codepoint budget",
                    )
        feasible = delta.get("feasible")
        if not isinstance(feasible, (tuple, list)):
            return value
        feasible_rows = cast(tuple[object, ...] | list[object], feasible)
        if len(feasible_rows) > MAX_TWIST_POLYNOMIAL_SOURCE_ROWS:
            raise PydanticCustomError(
                "delta_matroid.memberships_exceeded",
                "source feasible family exceeds its admitted row envelope",
            )
        remaining = MAX_DELTA_MEMBERSHIPS
        for row in feasible_rows:
            if isinstance(row, (tuple, list)):
                if any(type(index) is not int for index in row):
                    raise PydanticCustomError(
                        "delta_matroid.membership_index_type",
                        "feasible-set memberships must be integer indices",
                    )
                remaining -= len(row)
            if remaining < 0:
                raise PydanticCustomError(
                    "delta_matroid.memberships_exceeded",
                    "source feasible-family memberships exceed the admitted envelope",
                )
        # This before-validator receives decoded Python containers, where strict
        # tuple fields would otherwise lose Pydantic's JSON-array conversion.
        # Convert only after the axes and row counts have passed raw admission.
        normalized_delta = dict(delta)
        if isinstance(ground, list):
            normalized_delta["ground"] = tuple(ground)
        if isinstance(feasible, list):
            normalized_delta["feasible"] = tuple(
                tuple(row) if isinstance(row, list) else row for row in feasible
            )
        normalized_value = dict(raw_value)
        normalized_value["delta_matroid"] = normalized_delta
        return canonicalize_json_containers(normalized_value)

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

    @field_validator("polynomial", mode="before")
    @classmethod
    def admit_polynomial_claim(cls, value: object) -> object:
        # Check a raw authored polynomial before the generic IntegerPolynomial
        # codec expands up to 4,096 coefficients of up to 32,768 digits each.
        coefficients: object
        if isinstance(value, IntegerPolynomial):
            coefficients = value.coefficients
        elif type(value) is dict:
            coefficients = value.get("coefficients")
        else:
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_polynomial_bound",
                "polynomial must have an admitted coefficient sequence",
            )
        if (
            not isinstance(coefficients, (tuple, list))
            or type(coefficients) not in (tuple, list)
            or not 1 <= len(coefficients) <= MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES
        ):
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_polynomial_bound",
                "polynomial exceeds the admitted term capacity",
            )
        for coefficient in coefficients:
            if type(coefficient) is str:
                # Allow one sign character; the canonical codec checks spelling.
                if len(coefficient) <= MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS + (
                    coefficient.startswith("-")
                ):
                    continue
            elif (
                type(coefficient) is int
                and coefficient.bit_length() <= 14
                and abs(coefficient) < 10_000
            ):
                continue
            raise PydanticCustomError(
                "delta_matroid.twist_polynomial_polynomial_bound",
                "polynomial coefficient exceeds its admitted four-digit envelope",
            )
        return value

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
    "MAX_FEASIBLE_SIZE_PROFILE_ENTRIES",
    "MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS",
    "MAX_TWIST_POLYNOMIAL_COEFFICIENT_DIGITS",
    "MAX_TWIST_POLYNOMIAL_GROUND",
    "MAX_TWIST_POLYNOMIAL_HISTOGRAM_ENTRIES",
    "MAX_TWIST_POLYNOMIAL_LABEL_CODEPOINTS",
    "MAX_TWIST_POLYNOMIAL_SOURCE_ROWS",
    "MAX_TWIST_POLYNOMIAL_STATES",
    "MAX_TWIST_POLYNOMIAL_WORK",
    "MAX_TWIST_WIDTH_STATES",
    "MAX_TWIST_WIDTH_WORK",
    "BinaryMatrixRequest",
    "BinaryMatrixResult",
    "BinarySymmetricMatrix",
    "DeltaMatroidDualRequest",
    "DeltaMatroidDualResult",
    "DeltaMatroidFeasibleSizeProfile",
    "DeltaMatroidFeasibleSizeProfileRequest",
    "DeltaMatroidMinorRequest",
    "DeltaMatroidMinorResult",
    "DeltaMatroidTwistPolynomialRequest",
    "DeltaMatroidTwistPolynomialResult",
    "DeltaMatroidTwistWidthProfile",
    "DeltaMatroidTwistWidthProfileRequest",
    "DeltaMatroidTwistWidthProfileResult",
]

class BinaryMatrixTwistRequest(StrictModel):
    """Present a binary delta-matroid with a supplied twist subset."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Construct the complete feasible family of D(A)*T. Principal "
                "submatrix work is bounded by "
                f"{MAX_BINARY_PRINCIPAL_MINOR_WORK} pivot units; the ground "
                "axis has at most eight elements. The output contains at most "
                f"{MAX_BINARY_TWIST_STATES} feasible rows, "
                f"{MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS} feasible memberships, "
                f"{MAX_BINARY_TWIST_OUTPUT_CELLS} matrix, row, membership, and "
                "twist cells, and 4,096 aggregate ground-label bytes across the "
                "retained matrix and delta-matroid axes."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_BINARY_GROUND,
                "max_principal_minor_elimination_work": MAX_BINARY_PRINCIPAL_MINOR_WORK,
                "max_feasible_rows": MAX_BINARY_TWIST_STATES,
                "max_feasible_set_memberships": MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS,
                "max_twist_transport_work_units": MAX_BINARY_TWIST_TRANSPORT_WORK,
                "max_output_cells": MAX_BINARY_TWIST_OUTPUT_CELLS,
                "max_output_ground_label_utf8_bytes": 2 * MAX_BINARY_LABEL_BYTES,
            },
        }
    )

    matrix: BinarySymmetricMatrix = Field(
        description=(
            "Symmetric GF(2) matrix on at most eight labelled elements. The "
            "principal-minor constructor admits at most "
            f"{MAX_BINARY_PRINCIPAL_MINOR_WORK} elimination "
            "work units before enumerating feasible sets."
        )
    )
    subset: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_BINARY_GROUND,
        description=(
            "Sorted unique matrix-axis indices defining the twist. Each index "
            f"must lie in 0..{MAX_BINARY_GROUND - 1}."
        ),
    )

    @model_validator(mode="after")
    def require_subset(self) -> Self:
        if (
            any(type(index) is not int for index in self.subset)
            or self.subset != tuple(sorted(set(self.subset)))
            or any(
                index < 0 or index >= len(self.matrix.ground) for index in self.subset
            )
        ):
            raise PydanticCustomError(
                "delta_matroid.binary_twist_subset",
                "twist indices must be sorted, distinct, and in range",
            )
        return self
