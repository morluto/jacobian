from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid

MAX_BINARY_GROUND = 8
MAX_BINARY_LABEL_BYTES = 2_048
MAX_TWIST_WIDTH_STATES = 4_096
MAX_TWIST_WIDTH_WORK = 262_144
MAX_FEASIBLE_SIZE_PROFILE_ENTRIES = 4_096
MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS = 32_768
MAX_DIRECT_SUM_GROUND = 2_048
MAX_DIRECT_SUM_FEASIBLE_PAIRS = 250_000


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


__all__ = [
    "MAX_BINARY_GROUND",
    "MAX_BINARY_LABEL_BYTES",
    "MAX_FEASIBLE_SIZE_PROFILE_ENTRIES",
    "MAX_FEASIBLE_SIZE_PROFILE_RETAINED_UNITS",
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
    "DeltaMatroidTwistWidthProfile",
    "DeltaMatroidTwistWidthProfileRequest",
    "DeltaMatroidTwistWidthProfileResult",
]
