from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid

MAX_BINARY_GROUND = 8
MAX_BINARY_LABEL_BYTES = 2_048


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

    @staticmethod
    def _principal_minor_is_nonsingular(
        entries: tuple[tuple[int, ...], ...], indices: tuple[int, ...]
    ) -> bool:
        if not indices:
            return True
        rows = [[entries[row][column] for column in indices] for row in indices]
        size = len(rows)
        pivot = 0
        for column in range(size):
            found = next((row for row in range(pivot, size) if rows[row][column]), None)
            if found is None:
                continue
            rows[pivot], rows[found] = rows[found], rows[pivot]
            for row in range(size):
                if row != pivot and rows[row][column]:
                    for cell in range(column, size):
                        rows[row][cell] ^= rows[pivot][cell]
            pivot += 1
        return pivot == size

    @model_validator(mode="after")
    def _source_binding(self) -> Self:
        if self.delta_matroid.ground != self.matrix.ground:
            raise PydanticCustomError(
                "delta_matroid.binary_result_ground",
                "delta matroid ground must equal the source matrix ground",
            )
        expected = tuple(
            sorted(
                indices
                for mask in range(1 << len(self.matrix.ground))
                for indices in [
                    tuple(
                        index
                        for index in range(len(self.matrix.ground))
                        if mask >> index & 1
                    )
                ]
                if self._principal_minor_is_nonsingular(self.matrix.entries, indices)
            )
        )
        if self.delta_matroid.feasible != expected:
            raise PydanticCustomError(
                "delta_matroid.binary_result_relation",
                "delta matroid feasible sets must equal the source principal minors",
            )
        return self


__all__ = [
    "MAX_BINARY_GROUND",
    "MAX_BINARY_LABEL_BYTES",
    "BinaryMatrixRequest",
    "BinaryMatrixResult",
    "BinarySymmetricMatrix",
    "DeltaMatroidDualRequest",
    "DeltaMatroidDualResult",
    "DeltaMatroidMinorRequest",
    "DeltaMatroidMinorResult",
]
