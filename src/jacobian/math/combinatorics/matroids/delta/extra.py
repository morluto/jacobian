from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid

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
    + MAX_BINARY_GROUND
)


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
    subset: tuple[int, ...] = Field(
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


class BinaryMatrixResult(StrictModel):
    matrix: BinarySymmetricMatrix
    delta_matroid: FiniteDeltaMatroid
    twist: tuple[int, ...] = ()

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


__all__ = [
    "MAX_BINARY_GROUND",
    "MAX_BINARY_LABEL_BYTES",
    "MAX_BINARY_PRINCIPAL_MINOR_WORK",
    "MAX_BINARY_TWIST_OUTPUT_CELLS",
    "MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS",
    "MAX_BINARY_TWIST_STATES",
    "MAX_BINARY_TWIST_TRANSPORT_WORK",
    "BinaryMatrixRequest",
    "BinaryMatrixResult",
    "BinaryMatrixTwistRequest",
    "BinarySymmetricMatrix",
    "DeltaMatroidDualRequest",
    "DeltaMatroidDualResult",
    "DeltaMatroidMinorRequest",
    "DeltaMatroidMinorResult",
]
