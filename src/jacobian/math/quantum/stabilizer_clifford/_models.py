"""Typed contract for one-gate transport of exact stabilizer groups."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import (
    MAX_QUBIT_LABEL_LENGTH,
    ExactStabilizerGroup,
)


class StabilizerCliffordTransportRequest(StrictModel):
    """Conjugate one exact stabilizer group by a named elementary gate.

    Source relation checks, register and generator validation, gate
    transformation work, and the compact serialized result are bounded before
    the independent family is reduced or transformed rows are allocated. The
    current envelope supports at most 32 qubits, 64 source generators,
    1,000,000 work units, and 65,536 compact result bytes, independent of the
    exponentially larger group order.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "admission_limits": {
                "max_qubits": 32,
                "max_source_generators": 64,
                "max_gate_count": 1,
                "max_work_units": 1_000_000,
                "max_result_compact_json_bytes": 65_536,
            }
        }
    )

    group: ExactStabilizerGroup
    gate: Literal["H", "S", "CNOT"]
    qubits: tuple[StrictStr, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_gate_axes(self) -> Self:
        arity = 2 if self.gate == "CNOT" else 1
        if len(self.qubits) != arity or len(set(self.qubits)) != arity:
            raise PydanticCustomError(
                "quantum.stabilizer_clifford.gate_axes",
                "H and S require one qubit; CNOT requires distinct control and target qubits",
            )
        if any(
            not q
            or len(q) > MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in self.qubits
        ):
            raise PydanticCustomError(
                "quantum.stabilizer_clifford.qubit_label",
                "gate axes must be bounded Unicode scalar labels",
            )
        return self
