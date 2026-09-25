"""Typed contract for one-gate transport of exact stabilizer groups."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import (
    MAX_QUBIT_LABEL_LENGTH,
    ExactStabilizerGroup,
)


class StabilizerCliffordTransportRequest(StrictModel):
    """Conjugate one exact stabilizer group by a named elementary gate."""

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
