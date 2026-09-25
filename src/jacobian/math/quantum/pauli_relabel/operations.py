"""Native exact qubit Pauli register relabelling."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    MAX_QUBIT_LABEL_LENGTH,
    MAX_QUBITS,
    ExactQubitPauli,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.pauli_relabel._models import QubitRegisterRelabeling


def _reject(field: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(location=(field,), code=code, message=message)


def relabel_pauli(
    pauli: ExactQubitPauli, relabeling: QubitRegisterRelabeling
) -> ExactQubitPauli:
    """Transport an exact Pauli along a bijection of ordered register axes.

    ``target_ids_in_source_order[i]`` is the target qubit corresponding to
    source register position ``i``. A pure relabelling preserves the phase
    under the fixed ``i^r X^x Z^z`` convention.
    """
    if not isinstance(pauli, ExactQubitPauli):
        _reject(
            "pauli",
            "quantum.pauli.relabel.invalid_pauli",
            "input must be a valid exact register-bound Pauli",
        )
    phase_free = getattr(pauli, "phase_free", None)
    if not isinstance(phase_free, PhaseFreeQubitPauli):
        _reject(
            "pauli",
            "quantum.pauli.relabel.invalid_pauli",
            "input must be a valid exact register-bound Pauli",
        )
    source = getattr(phase_free, "qubit_register", None)
    if not isinstance(source, QubitRegister) or not isinstance(
        relabeling, QubitRegisterRelabeling
    ):
        _reject(
            "register",
            "quantum.pauli.relabel.invalid_register",
            "source and target must be qubit registers",
        )
    if relabeling.source_register != source:
        _reject(
            "relabeling",
            "quantum.pauli.relabel.source_mismatch",
            "the register map must start at the Pauli's source register",
        )
    target = relabeling.target_register
    mapping = relabeling.target_ids_in_source_order
    source_ids = source.qubit_ids
    target_ids = target.qubit_ids
    if (
        type(source_ids) is not tuple
        or type(target_ids) is not tuple
        or not 1 <= len(source_ids) <= MAX_QUBITS
        or len(source_ids) != len(target_ids)
        or any(
            type(label) is not str
            or not 1 <= len(label) <= MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(char) <= 0xDFFF for char in label)
            for label in (*source_ids, *target_ids)
        )
        or len(set(source_ids)) != len(source_ids)
        or len(set(target_ids)) != len(target_ids)
    ):
        _reject(
            "register",
            "quantum.pauli.relabel.invalid_register",
            "register axes must be bounded unique qubit labels of equal size",
        )
    x_bits = phase_free.x_bits
    z_bits = phase_free.z_bits
    if (
        type(x_bits) is not tuple
        or type(z_bits) is not tuple
        or len(x_bits) != len(source_ids)
        or len(z_bits) != len(source_ids)
        or any(type(bit) is not int or bit not in (0, 1) for bit in (*x_bits, *z_bits))
        or type(pauli.phase) is not int
        or not 0 <= pauli.phase <= 3
    ):
        _reject(
            "pauli",
            "quantum.pauli.relabel.invalid_pauli",
            "input must be a valid exact register-bound Pauli",
        )
    if (
        type(mapping) is not tuple
        or len(mapping) != len(source_ids)
        or any(type(label) is not str for label in mapping)
        or len(set(mapping)) != len(mapping)
        or set(mapping) != set(target_ids)
    ):
        _reject(
            "relabeling.target_ids_in_source_order",
            "quantum.pauli.relabel.not_bijection",
            "the source-to-target labels must define a bijection onto the target register",
        )

    target_index = {label: index for index, label in enumerate(target_ids)}
    target_x = [0] * len(target_ids)
    target_z = [0] * len(target_ids)
    for source_index, target_label in enumerate(mapping):
        target_position = target_index[target_label]
        target_x[target_position] = x_bits[source_index]
        target_z[target_position] = z_bits[source_index]
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=target,
            x_bits=tuple(target_x),
            z_bits=tuple(target_z),
        ),
        phase=pauli.phase,
    )
