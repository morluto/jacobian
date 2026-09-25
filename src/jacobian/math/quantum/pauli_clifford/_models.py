"""Typed request for exact conjugation by one elementary Clifford gate."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import MAX_QUBIT_LABEL_LENGTH, ExactQubitPauli


class PauliCliffordConjugationRequest(StrictModel):
    """Conjugate one exact Pauli by H, S, or CNOT on its bound register.

    ``qubits`` contains the target for H/S and ``(control, target)`` for CNOT.
    This is a single gate action, not a circuit or gate-sequence language.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "admission_limits": {
                "max_qubits": 32,
                "max_gate_count": 1,
                "max_work_units": 4096,
                "max_result_compact_json_bytes": 16384,
            }
        }
    )

    pauli: ExactQubitPauli
    gate: Literal["H", "S", "CNOT"]
    qubits: tuple[StrictStr, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_gate_axes(self) -> Self:
        expected = 2 if self.gate == "CNOT" else 1
        if len(self.qubits) != expected:
            raise PydanticCustomError(
                "quantum.pauli_clifford.gate_arity",
                "H and S require one qubit; CNOT requires control and target",
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise PydanticCustomError(
                "quantum.pauli_clifford.distinct_qubits",
                "gate qubit roles must refer to distinct register elements",
            )
        if any(not q or len(q) > MAX_QUBIT_LABEL_LENGTH for q in self.qubits):
            raise PydanticCustomError(
                "quantum.pauli_clifford.qubit_label",
                "gate qubit labels must be nonempty and bounded",
            )
        if any(0xD800 <= ord(char) <= 0xDFFF for q in self.qubits for char in q):
            raise PydanticCustomError(
                "quantum.pauli_clifford.unicode_scalar",
                "gate qubit labels must contain Unicode scalar values",
            )
        return self
