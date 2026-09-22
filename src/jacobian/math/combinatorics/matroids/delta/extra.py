from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid

MAX_BINARY_GROUND = 12
MAX_BINARY_LABEL_BYTES = 2048


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
    ground: tuple[str, ...] = Field(
        description=(
            "Distinct labelled axes; at most "
            f"{MAX_BINARY_GROUND} labels because all principal subsets are materialized."
        )
    )
    entries: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.ground)
        if n > MAX_BINARY_GROUND:
            raise PydanticCustomError(
                "delta_matroid.binary_ground",
                f"binary principal-minor axes are limited to {MAX_BINARY_GROUND} labels",
            )
        if len(set(self.ground)) != n:
            raise PydanticCustomError(
                "delta_matroid.binary_ground", "binary ground labels must be unique"
            )
        if (
            sum(len(label.encode("utf-8")) for label in self.ground)
            > MAX_BINARY_LABEL_BYTES
        ):
            raise PydanticCustomError(
                "delta_matroid.binary_ground",
                f"binary ground labels are limited to {MAX_BINARY_LABEL_BYTES} UTF-8 bytes",
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


__all__ = [
    "BinaryMatrixRequest",
    "BinaryMatrixResult",
    "BinarySymmetricMatrix",
    "DeltaMatroidDualRequest",
    "DeltaMatroidDualResult",
    "DeltaMatroidMinorRequest",
    "DeltaMatroidMinorResult",
]
