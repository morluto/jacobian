"""Exact single-gate Clifford action on an exact stabilizer group."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBITS,
    ExactQubitPauli,
    ExactStabilizerGroup,
    ExactStabilizerGroupRequest,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.operations import stabilizer_group_from_generators
from jacobian.math.quantum.stabilizer_clifford._models import (
    StabilizerCliffordTransportRequest,
)

MAX_STABILIZER_CLIFFORD_WORK = 1_000_000
MAX_STABILIZER_CLIFFORD_RESULT_BYTES = 65_536


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _transport_pauli(
    pauli: ExactQubitPauli,
    gate: str,
    axes: tuple[str, ...],
) -> ExactQubitPauli:
    """Apply the H/S/CNOT conjugation formulas in the shared Pauli convention."""
    register = pauli.register
    ids = register.qubit_ids
    x = list(pauli.phase_free.x_bits)
    z = list(pauli.phase_free.z_bits)
    phase = pauli.phase
    first = ids.index(axes[0])
    if gate == "H":
        phase = (phase + 2 * x[first] * z[first]) % 4
        x[first], z[first] = z[first], x[first]
    elif gate == "S":
        phase = (phase + x[first]) % 4
        z[first] ^= x[first]
    else:
        target = ids.index(axes[1])
        old_x_control = x[first]
        old_z_target = z[target]
        x[target] ^= old_x_control
        z[first] ^= old_z_target
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=register, x_bits=tuple(x), z_bits=tuple(z)
        ),
        phase=phase,
    )


def conjugate_stabilizer_group(
    request: StabilizerCliffordTransportRequest,
) -> ExactStabilizerGroup:
    """Return the exact group ``U S U†`` for one H, S, or directed CNOT gate.

    Input group claims are independently checked and canonicalized before the
    automorphism is applied. Clifford conjugation preserves Hermiticity,
    commutation, and independence, so the canonical result remains a
    phase-consistent exact stabilizer group.
    """
    if not isinstance(request, StabilizerCliffordTransportRequest):
        _reject(
            "request",
            "quantum.stabilizer_clifford.invalid_request",
            "request must contain a typed group and one elementary gate",
        )
    group = request.group
    if not isinstance(group, ExactStabilizerGroup):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_group",
            "input must be an exact stabilizer group",
        )
    register = group.qubit_register
    if not isinstance(register, QubitRegister):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_register",
            "group register must be typed",
        )
    ids = register.qubit_ids
    width = len(ids) if isinstance(ids, tuple) else 0
    generators = group.generators
    count = len(generators) if isinstance(generators, tuple) else -1
    if not 1 <= width <= MAX_QUBITS or not 0 <= count <= MAX_CHECK_ROWS:
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_shape",
            "register or generator family exceeds the admitted shape",
        )
    if (
        any(
            type(q) is not str
            or not q
            or len(q) > 64
            or any(0xD800 <= ord(char) <= 0xDFFF for char in q)
            for q in ids
        )
        or len(set(ids)) != width
        or any(
            not isinstance(g, ExactQubitPauli)
            or not isinstance(g.phase_free, PhaseFreeQubitPauli)
            or g.register != register
            or type(g.phase) is not int
            or not 0 <= g.phase < 4
            or len(g.phase_free.x_bits) != width
            or len(g.phase_free.z_bits) != width
            or any(
                type(bit) is not int or bit not in (0, 1)
                for bit in (*g.phase_free.x_bits, *g.phase_free.z_bits)
            )
            for g in generators
        )
    ):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_group",
            "group generators must be exact Paulis on the identical register",
        )
    if request.gate not in ("H", "S", "CNOT"):
        _reject(
            "gate",
            "quantum.stabilizer_clifford.invalid_gate",
            "gate must be H, S, or CNOT",
        )
    arity = 2 if request.gate == "CNOT" else 1
    if (
        type(request.qubits) is not tuple
        or len(request.qubits) != arity
        or len(set(request.qubits)) != arity
        or any(q not in ids for q in request.qubits)
    ):
        _reject(
            "qubits",
            "quantum.stabilizer_clifford.invalid_axes",
            "gate axes must name distinct qubits in the group's register",
        )

    # Bound complete source semantic validation plus all gate transformations
    # and compact output before canonicalization or transformed row allocation.
    pair_count = count * (count - 1) // 2
    source_validation = (
        2 * pair_count * width
        + 8 * count * min(count, width) * width
        + width * 68
        + 2 * count * width * 68
        + count * width
    )
    label_bytes = sum(6 * len(q) + 4 for q in ids)
    gate_work = count * (12 * width + label_bytes + 32)
    output_bound = (count + 1) * label_bytes + count * (24 * width + 128) + 256
    if (
        source_validation + gate_work > MAX_STABILIZER_CLIFFORD_WORK
        or output_bound > MAX_STABILIZER_CLIFFORD_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("group",),
            code="quantum.stabilizer_clifford.over_envelope",
            message="group validation or exact transported result exceeds its envelope",
        )

    canonical = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(
            qubit_register=register,
            generators=generators,
        )
    )
    transported = tuple(
        _transport_pauli(pauli, request.gate, request.qubits)
        for pauli in canonical.generators
    )
    return ExactStabilizerGroup(qubit_register=register, generators=transported)


__all__ = ["conjugate_stabilizer_group"]
