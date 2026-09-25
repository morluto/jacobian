"""Request contract for exact qubit Pauli register relabelling."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quantum._models import (
    MAX_QUBITS,
    ExactQubitPauli,
    QubitId,
    QubitRegister,
)


class QubitRegisterRelabeling(StrictModel):
    """An explicit bijection between two ordered register axes."""

    source_register: QubitRegister
    target_register: QubitRegister
    target_ids_in_source_order: tuple[QubitId, ...] = Field(
        min_length=1, max_length=MAX_QUBITS
    )

    @model_validator(mode="after")
    def require_bijection(self) -> QubitRegisterRelabeling:
        source_ids = self.source_register.qubit_ids
        target_ids = self.target_register.qubit_ids
        if len(source_ids) != len(target_ids) or len(
            self.target_ids_in_source_order
        ) != len(source_ids):
            raise PydanticCustomError(
                "quantum.pauli.relabel.mapping_shape",
                "a register relabelling must map every source axis to a target axis of equal size",
            )
        if len(set(self.target_ids_in_source_order)) != len(source_ids) or set(
            self.target_ids_in_source_order
        ) != set(target_ids):
            raise PydanticCustomError(
                "quantum.pauli.relabel.mapping_not_bijection",
                "the source-to-target labels must define a bijection onto the target register",
            )
        return self


class PauliRelabelRequest(StrictModel):
    """Tool arguments selecting an exact Pauli and an explicit axis map."""

    pauli: ExactQubitPauli
    relabeling: QubitRegisterRelabeling
