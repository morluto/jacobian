"""Exact conjugation of Pauli values by one elementary Clifford gate."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum._models import (
    MAX_QUBITS,
    ExactQubitPauli,
    PhaseFreeQubitPauli,
    QubitRegister,
)

MAX_CLIFFORD_WORK = 4096


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def conjugate_pauli(
    pauli: ExactQubitPauli, gate: str, qubits: tuple[str, ...]
) -> ExactQubitPauli:
    """Return ``U P U†`` for one H, S, or CNOT gate under ``i^r X^x Z^z``.

    The phase update follows the published operator convention ``Y=iXZ``.
    Complexity is linear in the register width, including structural admission.
    """
    phase_free = (
        getattr(pauli, "phase_free", None)
        if isinstance(pauli, ExactQubitPauli)
        else None
    )
    if not isinstance(phase_free, PhaseFreeQubitPauli):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "input must be an exact qubit Pauli",
        )
    register = getattr(phase_free, "qubit_register", None)
    if not isinstance(register, QubitRegister):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "Pauli register must be a typed qubit register",
        )
    ids = getattr(register, "qubit_ids", None)
    x_bits = getattr(phase_free, "x_bits", None)
    z_bits = getattr(phase_free, "z_bits", None)
    width = len(ids) if isinstance(ids, tuple) else 0
    if (
        type(ids) is not tuple
        or not 1 <= width <= MAX_QUBITS
        or any(
            type(q) is not str
            or not q
            or len(q) > 64
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in ids
        )
        or len(set(ids)) != width
        or not isinstance(x_bits, tuple)
        or not isinstance(z_bits, tuple)
        or len(x_bits) != width
        or len(z_bits) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in (*x_bits, *z_bits))
        or type(getattr(pauli, "phase", None)) is not int
        or not 0 <= getattr(pauli, "phase", -1) <= 3
    ):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "Pauli coordinates or register are outside the admitted form",
        )
    if gate not in ("H", "S", "CNOT"):
        _reject(
            "gate", "quantum.pauli_clifford.invalid_gate", "gate must be H, S, or CNOT"
        )
    if (
        type(qubits) is not tuple
        or len(qubits) != (2 if gate == "CNOT" else 1)
        or any(
            type(q) is not str
            or not q
            or len(q) > 64
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in qubits
        )
    ):
        _reject(
            "qubits",
            "quantum.pauli_clifford.invalid_axes",
            "gate axes must be a tuple of register labels",
        )
    expected_arity = 2 if gate == "CNOT" else 1
    if len(set(qubits)) != expected_arity or any(q not in ids for q in qubits):
        _reject(
            "qubits",
            "quantum.pauli_clifford.invalid_axes",
            "gate roles must identify the required distinct register elements",
        )

    # Admission is based on intrinsic linear scans and allocation cells. The
    # domain admits at most 32 coordinates in each of x and z.
    work = 12 * width + 32
    if work > MAX_CLIFFORD_WORK:
        raise OperationResourceAdmissionError(
            location=("pauli",),
            code="quantum.pauli_clifford.over_envelope",
            message="Pauli gate work exceeds its envelope",
        )

    x = list(x_bits)
    z = list(z_bits)
    phase = pauli.phase
    first = ids.index(qubits[0])
    if gate == "H":
        phase = (phase + 2 * x[first] * z[first]) % 4
        x[first], z[first] = z[first], x[first]
    elif gate == "S":
        phase = (phase + x[first]) % 4
        z[first] ^= x[first]
    else:
        control = first
        target = ids.index(qubits[1])
        old_x_control = x[control]
        old_z_target = z[target]
        x[target] ^= old_x_control
        z[control] ^= old_z_target

    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=register, x_bits=tuple(x), z_bits=tuple(z)
        ),
        phase=phase,
    )


__all__ = ["conjugate_pauli"]
