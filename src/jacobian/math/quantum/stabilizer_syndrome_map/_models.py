"""Typed values for the binary stabilizer syndrome linear map."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBITS,
    CheckSpaceValue,
    NormalizerResult,
)

MAX_SYNDROME_MAP_CHECK_ROWS = MAX_CHECK_ROWS
MAX_SYNDROME_MAP_WORK = 1_000_000
MAX_SYNDROME_MAP_RESULT_BYTES = 2_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quantum.syndrome_map.{reason}", message)


class SyndromeMapRequest(StrictModel):
    """Build the full syndrome map for one finite binary check space."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Construct the GF(2)-linear map from canonical phase-free Pauli "
                "coordinates [x|z] to the dual coordinates of a canonical check "
                "basis. The request supports at most 32 qubits and 64 input check "
                "rows; admitted work is at most 1,000,000 units and the result "
                "at most 2,000,000 compact JSON bytes."
            ),
            "admission_limits": {
                "max_qubits": MAX_QUBITS,
                "max_input_check_rows": MAX_SYNDROME_MAP_CHECK_ROWS,
                "max_work_units": MAX_SYNDROME_MAP_WORK,
                "max_result_compact_json_bytes": MAX_SYNDROME_MAP_RESULT_BYTES,
                "max_fiber_dimension": 2 * MAX_QUBITS,
            },
        }
    )

    check_space: CheckSpaceValue


class SyndromeMapResult(StrictModel):
    """A source-bound matrix for the map ``GF(2)^(2n) -> S*``.

    Matrix columns are ordered ``[x_0,...,x_(n-1),z_0,...,z_(n-1)]``.
    Matrix row ``i`` is the linear functional ``e |-> <g_i,e>`` for row ``g_i``
    in ``normalizer.check_space.basis``. Thus the rows give coordinates in the
    dual basis of the canonical check-space basis.
    """

    source_check_space: CheckSpaceValue
    normalizer: NormalizerResult
    matrix: tuple[tuple[int, ...], ...] = Field(max_length=MAX_SYNDROME_MAP_CHECK_ROWS)
    field_order: Literal[2] = 2
    domain_dimension: int = Field(ge=0, le=2 * MAX_QUBITS)
    codomain_dimension: int = Field(ge=0, le=MAX_SYNDROME_MAP_CHECK_ROWS)
    rank: int = Field(ge=0, le=MAX_SYNDROME_MAP_CHECK_ROWS)
    kernel_dimension: int = Field(ge=0, le=2 * MAX_QUBITS)
    fiber_dimension: int = Field(ge=0, le=2 * MAX_QUBITS)
    fiber_cardinality: StrictStr = Field(
        min_length=1,
        max_length=20,
        pattern=r"^(0|[1-9][0-9]*)$",
        description="Exact fiber cardinality as a canonical decimal string.",
    )

    @model_validator(mode="after")
    def require_bound_dimensions(self) -> Self:
        register = self.source_check_space.qubit_register
        if self.normalizer.check_space.qubit_register != register:
            raise _validation_error(
                "register_axis", "source and syndrome map must share one register"
            )
        width = 2 * len(register.qubit_ids)
        if (
            self.domain_dimension != width
            or self.codomain_dimension != self.rank
            or self.rank != self.normalizer.rank
            or self.rank != len(self.normalizer.check_space.basis)
            or self.kernel_dimension != self.normalizer.orthogonal_rank
            or self.kernel_dimension != width - self.rank
            or self.fiber_dimension != self.kernel_dimension
            or self.fiber_cardinality != str(1 << self.fiber_dimension)
            or len(self.matrix) != self.rank
            or any(len(row) != width for row in self.matrix)
            or any(bit not in (0, 1) for row in self.matrix for bit in row)
        ):
            raise _validation_error(
                "dimensions_or_matrix_shape",
                "syndrome map matrix and dimensions must match their bound axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_check_space: CheckSpaceValue,
        normalizer: NormalizerResult,
        matrix: tuple[tuple[int, ...], ...],
    ) -> Self:
        domain_dimension = 2 * len(normalizer.check_space.qubit_register.qubit_ids)
        rank = normalizer.rank
        kernel_dimension = normalizer.orthogonal_rank
        return cls.model_construct(
            source_check_space=source_check_space,
            normalizer=normalizer,
            matrix=matrix,
            field_order=2,
            domain_dimension=domain_dimension,
            codomain_dimension=rank,
            rank=rank,
            kernel_dimension=kernel_dimension,
            fiber_dimension=kernel_dimension,
            fiber_cardinality=str(1 << kernel_dimension),
        )


__all__ = [
    "MAX_SYNDROME_MAP_CHECK_ROWS",
    "MAX_SYNDROME_MAP_RESULT_BYTES",
    "MAX_SYNDROME_MAP_WORK",
    "SyndromeMapRequest",
    "SyndromeMapResult",
]
