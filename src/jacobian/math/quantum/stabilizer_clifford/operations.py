"""Exact single-gate Clifford action on an exact stabilizer group."""

from __future__ import annotations

from typing import Any, NoReturn, cast

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


def _is_admitted_generator(value: object, register: QubitRegister, width: int) -> bool:
    """Check one nested generator without dereferencing forged model fields."""

    if not isinstance(value, ExactQubitPauli):
        return False
    phase_free = getattr(value, "phase_free", None)
    if not isinstance(phase_free, PhaseFreeQubitPauli):
        return False
    x_bits = getattr(phase_free, "x_bits", None)
    z_bits = getattr(phase_free, "z_bits", None)
    phase = getattr(value, "phase", None)
    return (
        getattr(phase_free, "qubit_register", None) == register
        and type(phase) is int
        and 0 <= phase < 4
        and isinstance(x_bits, tuple)
        and isinstance(z_bits, tuple)
        and len(x_bits) == width
        and len(z_bits) == width
        and all(type(bit) is int and bit in (0, 1) for bit in (*x_bits, *z_bits))
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

    The supplied generators are independently re-admitted and reduced to an
    independent family in their given order before the automorphism is applied
    to each retained generator. Clifford conjugation preserves Hermiticity,
    commutation, and independence, so the result is again a phase-consistent
    exact stabilizer group. The retained family is a valid presentation, not a
    canonical one: equal subgroups may serialize with different generator
    order, so callers must not use serialization as group identity.
    """
    if not isinstance(request, StabilizerCliffordTransportRequest):
        _reject(
            "request",
            "quantum.stabilizer_clifford.invalid_request",
            "request must contain a typed group and one elementary gate",
        )
    group = getattr(request, "group", None)
    if not isinstance(group, ExactStabilizerGroup):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_group",
            "input must be an exact stabilizer group",
        )
    register_field: object = getattr(group, "qubit_register", None)
    if not isinstance(register_field, QubitRegister):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_register",
            "group register must be typed",
        )
    register = register_field
    ids: tuple[Any, ...] = ()
    ids_field: object = getattr(register, "qubit_ids", None)
    if isinstance(ids_field, tuple):
        ids = ids_field
    width = len(ids)
    generators: tuple[Any, ...] = ()
    generators_field: object = getattr(group, "generators", None)
    if isinstance(generators_field, tuple):
        generators = generators_field
        count = len(generators)
    else:
        count = -1
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
        or any(not _is_admitted_generator(g, register, width) for g in generators)
    ):
        _reject(
            "group",
            "quantum.stabilizer_clifford.invalid_group",
            "group generators must be exact Paulis on the identical register",
        )
    gate_field: object = getattr(request, "gate", None)
    if gate_field not in ("H", "S", "CNOT"):
        _reject(
            "gate",
            "quantum.stabilizer_clifford.invalid_gate",
            "gate must be H, S, or CNOT",
        )
    gate = gate_field
    arity = 2 if gate == "CNOT" else 1
    axes_field: object = getattr(request, "qubits", None)
    if type(axes_field) is not tuple:
        _reject(
            "qubits",
            "quantum.stabilizer_clifford.invalid_axes",
            "gate axes must name distinct qubits in the group's register",
        )
    axes = cast("tuple[str, ...]", axes_field)
    if (
        len(axes) != arity
        or any(type(q) is not str for q in axes)
        or len(set(axes)) != arity
        or any(q not in ids for q in axes)
    ):
        _reject(
            "qubits",
            "quantum.stabilizer_clifford.invalid_axes",
            "gate axes must name distinct qubits in the group's register",
        )

    # Bound full source validation before canonicalizing caller-supplied rows.
    pair_count = count * (count - 1) // 2
    source_validation = (
        2 * pair_count * width
        + 8 * count * min(count, width) * width
        + width * 68
        + 2 * count * width * 68
        + count * width
    )
    label_bytes = sum(6 * len(q) + 4 for q in ids)
    if source_validation > MAX_STABILIZER_CLIFFORD_WORK:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="quantum.stabilizer_clifford.over_envelope",
            message="source group validation exceeds its exact work envelope",
        )

    canonical = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(
            register=register,
            generators=generators,
        )
    )
    canonical_count = len(canonical.generators)
    gate_work = canonical_count * (12 * width + label_bytes + 32)
    output_bound = (
        (canonical_count + 1) * label_bytes + canonical_count * (24 * width + 128) + 256
    )
    if (
        source_validation + gate_work > MAX_STABILIZER_CLIFFORD_WORK
        or output_bound > MAX_STABILIZER_CLIFFORD_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("group",),
            code="quantum.stabilizer_clifford.over_envelope",
            message="group validation or exact transported result exceeds its envelope",
        )
    transported = tuple(
        _transport_pauli(pauli, gate, axes) for pauli in canonical.generators
    )
    return ExactStabilizerGroup(register=register, generators=transported)


__all__ = ["conjugate_stabilizer_group"]
