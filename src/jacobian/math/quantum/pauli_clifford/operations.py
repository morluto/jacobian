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
from jacobian.math.quantum.pauli_clifford._models import (
    PauliCliffordConjugationRequest,
)

MAX_CLIFFORD_WORK = 4096
MAX_CLIFFORD_RESULT_BYTES = 16_384


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def conjugate_pauli(request: PauliCliffordConjugationRequest) -> ExactQubitPauli:
    """Return ``U P U†`` for one H, S, or CNOT gate under ``i^r X^x Z^z``.

    The phase update follows the published operator convention ``Y=iXZ``.
    Complexity is linear in the register width, including structural admission.
    """
    if not isinstance(request, PauliCliffordConjugationRequest):
        _reject(
            "request",
            "quantum.pauli_clifford.invalid_request",
            "request must use the typed gate action",
        )
    pauli = request.pauli
    if not isinstance(pauli, ExactQubitPauli) or not isinstance(
        getattr(pauli, "phase_free", None), PhaseFreeQubitPauli
    ):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "input must be an exact qubit Pauli",
        )
    register = pauli.register
    if not isinstance(register, QubitRegister):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "Pauli register must be a typed qubit register",
        )
    ids = getattr(register, "qubit_ids", None)
    x_bits = getattr(pauli.phase_free, "x_bits", None)
    z_bits = getattr(pauli.phase_free, "z_bits", None)
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
        or type(pauli.phase) is not int
        or not 0 <= pauli.phase <= 3
    ):
        _reject(
            "pauli",
            "quantum.pauli_clifford.invalid_pauli",
            "Pauli coordinates or register are outside the admitted form",
        )
    if request.gate not in ("H", "S", "CNOT"):
        _reject(
            "gate", "quantum.pauli_clifford.invalid_gate", "gate must be H, S, or CNOT"
        )
    if (
        type(request.qubits) is not tuple
        or len(request.qubits) != (2 if request.gate == "CNOT" else 1)
        or any(
            type(q) is not str
            or not q
            or len(q) > 64
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in request.qubits
        )
    ):
        _reject(
            "qubits",
            "quantum.pauli_clifford.invalid_axes",
            "gate axes must be a tuple of register labels",
        )
    expected_arity = 2 if request.gate == "CNOT" else 1
    if len(set(request.qubits)) != expected_arity or any(
        q not in ids for q in request.qubits
    ):
        _reject(
            "qubits",
            "quantum.pauli_clifford.invalid_axes",
            "gate roles must identify the required distinct register elements",
        )

    # Count worst-case JSON escaping so output admission precedes coordinate copies.
    label_bytes = sum(6 * len(q) + 4 for q in ids)
    work = 12 * width + label_bytes + 32
    output_bound = label_bytes + 24 * width + 256
    if work > MAX_CLIFFORD_WORK or output_bound > MAX_CLIFFORD_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("pauli",),
            code="quantum.pauli_clifford.over_envelope",
            message="Pauli gate work or exact result size exceeds its envelope",
        )

    x = list(x_bits)
    z = list(z_bits)
    phase = pauli.phase
    first = ids.index(request.qubits[0])
    if request.gate == "H":
        phase = (phase + 2 * x[first] * z[first]) % 4
        x[first], z[first] = z[first], x[first]
    elif request.gate == "S":
        phase = (phase + x[first]) % 4
        z[first] ^= x[first]
    else:
        control = first
        target = ids.index(request.qubits[1])
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
