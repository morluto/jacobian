"""Native exact addition of phase-free qubit Pauli vectors."""

from __future__ import annotations

from typing import NoReturn, cast

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    MAX_QUBIT_LABEL_LENGTH,
    MAX_QUBITS,
    PhaseFreeQubitPauli,
    QubitRegister,
)


def _reject(field: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(location=(field,), code=code, message=message)


def add_phase_free_paulis(
    left: PhaseFreeQubitPauli, right: PhaseFreeQubitPauli
) -> PhaseFreeQubitPauli:
    """Add binary symplectic coordinates on one identical ordered register."""
    values = (left, right)
    if any(not isinstance(value, PhaseFreeQubitPauli) for value in values):
        _reject(
            "paulis",
            "quantum.pauli.phase_free.invalid_value",
            "both inputs must be typed phase-free Paulis",
        )
    registers = tuple(getattr(value, "qubit_register", None) for value in values)
    if any(not isinstance(register, QubitRegister) for register in registers):
        _reject(
            "paulis",
            "quantum.pauli.phase_free.invalid_register",
            "both Paulis must retain valid qubit registers",
        )
    register = cast(QubitRegister, registers[0])
    qubit_ids = register.qubit_ids
    if (
        type(qubit_ids) is not tuple
        or not 1 <= len(qubit_ids) <= MAX_QUBITS
        or any(
            type(label) is not str
            or not 1 <= len(label) <= MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(char) <= 0xDFFF for char in label)
            for label in qubit_ids
        )
        or len(set(qubit_ids)) != len(qubit_ids)
    ):
        _reject(
            "paulis",
            "quantum.pauli.phase_free.invalid_register",
            "register axes must be bounded unique qubit labels",
        )
    if registers[1] != register:
        _reject(
            "paulis",
            "quantum.pauli.phase_free.register_mismatch",
            "both Paulis must use the same ordered qubit register",
        )
    coordinates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for value in values:
        x_bits = getattr(value, "x_bits", None)
        z_bits = getattr(value, "z_bits", None)
        if (
            type(x_bits) is not tuple
            or type(z_bits) is not tuple
            or len(x_bits) != len(qubit_ids)
            or len(z_bits) != len(qubit_ids)
            or any(
                type(bit) is not int or bit not in (0, 1) for bit in (*x_bits, *z_bits)
            )
        ):
            _reject(
                "paulis",
                "quantum.pauli.phase_free.invalid_coordinates",
                "Pauli coordinates must be binary rows on the shared register",
            )
        coordinates.append((x_bits, z_bits))

    left_x, left_z = coordinates[0]
    right_x, right_z = coordinates[1]
    return PhaseFreeQubitPauli(
        register=register,
        x_bits=tuple(x ^ y for x, y in zip(left_x, right_x, strict=True)),
        z_bits=tuple(z ^ y for z, y in zip(left_z, right_z, strict=True)),
    )
