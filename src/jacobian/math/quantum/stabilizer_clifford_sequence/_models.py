"""Typed finite Clifford sequences and operation requests."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import (
    MAX_QUBIT_LABEL_LENGTH,
    ExactStabilizerGroup,
    QubitRegister,
)

MAX_CLIFFORD_SEQUENCE_GATES = 128


class CliffordGate(StrictModel):
    """One named H, S, or directed CNOT gate on register axes."""

    gate: Literal["H", "S", "CNOT"]
    qubits: tuple[StrictStr, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_gate_axes(self) -> Self:
        arity = 2 if self.gate == "CNOT" else 1
        if len(self.qubits) != arity or len(set(self.qubits)) != arity:
            raise PydanticCustomError(
                "quantum.stabilizer_clifford_sequence.gate_axes",
                "H and S require one axis; CNOT requires distinct control and target axes",
            )
        if any(
            not q
            or len(q) > MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in self.qubits
        ):
            raise PydanticCustomError(
                "quantum.stabilizer_clifford_sequence.qubit_label",
                "gate axes must be bounded Unicode scalar labels",
            )
        return self


class StabilizerCliffordSequence(StrictModel):
    """An ordered finite sequence of elementary Clifford gates on one register.

    Gates are listed in application order: for ``(g1, g2)``, the resulting
    group is ``U2 U1 S U1† U2†``. The empty sequence is the identity action.
    """

    model_config = ConfigDict(populate_by_name=True)

    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    gates: tuple[CliffordGate, ...] = Field(max_length=MAX_CLIFFORD_SEQUENCE_GATES)

    @property
    def register(self) -> QubitRegister:
        return self.qubit_register

    @model_validator(mode="after")
    def require_register_axes(self) -> Self:
        axes = set(self.register.qubit_ids)
        if any(axis not in axes for gate in self.gates for axis in gate.qubits):
            raise PydanticCustomError(
                "quantum.stabilizer_clifford_sequence.register_axis",
                "every gate axis must name a qubit in the sequence register",
            )
        return self


class CliffordSequenceCompositionRequest(StrictModel):
    """Compose two sequences in the order left first, then right."""

    left: StabilizerCliffordSequence
    right: StabilizerCliffordSequence


class StabilizerCliffordSequenceApplyRequest(StrictModel):
    """Apply one finite Clifford sequence to an exact stabilizer group."""

    group: ExactStabilizerGroup
    sequence: StabilizerCliffordSequence
