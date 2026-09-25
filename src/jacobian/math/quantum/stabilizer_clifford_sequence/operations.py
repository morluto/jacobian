"""Composition and exact stabilizer-group action for finite Clifford sequences."""

from __future__ import annotations

from itertools import chain
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
from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    MAX_CLIFFORD_SEQUENCE_GATES,
    CliffordGate,
    CliffordSequenceCompositionRequest,
    StabilizerCliffordSequence,
    StabilizerCliffordSequenceApplyRequest,
)

MAX_SEQUENCE_WORK = 1_500_000
MAX_SEQUENCE_RESULT_BYTES = 65_536
MAX_SEQUENCE_COMPOSITION_WORK = 300_000
MAX_SEQUENCE_COMPOSITION_RESULT_BYTES = 512_000


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _register_label_bytes(register: QubitRegister) -> int:
    return sum(6 * len(qubit) + 4 for qubit in register.qubit_ids)


def _admit_register(value: object, location: str) -> QubitRegister:
    if not isinstance(value, QubitRegister):
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_register",
            "register must be typed",
        )
    ids = value.qubit_ids
    if (
        type(ids) is not tuple
        or not 1 <= len(ids) <= MAX_QUBITS
        or any(
            type(qubit) is not str
            or not qubit
            or len(qubit) > 64
            or any(0xD800 <= ord(char) <= 0xDFFF for char in qubit)
            for qubit in ids
        )
        or len(set(ids)) != len(ids)
    ):
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_register",
            "register axes must be unique bounded Unicode scalar labels",
        )
    return value


def _admit_sequence_shape(
    value: object,
    location: str,
    expected_register: QubitRegister | None = None,
) -> tuple[StabilizerCliffordSequence, int]:
    if not isinstance(value, StabilizerCliffordSequence):
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_sequence",
            "value must be a typed finite Clifford sequence",
        )
    register = _admit_register(value.register, f"{location}.register")
    if expected_register is not None and register != expected_register:
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.register_mismatch",
            "sequence and target must use the identical ordered register",
        )
    gates = value.gates
    count = len(gates) if isinstance(gates, tuple) else -1
    if not 0 <= count <= MAX_CLIFFORD_SEQUENCE_GATES:
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_size",
            "sequence exceeds its admitted gate count",
        )
    ids = register.qubit_ids
    for index, gate in enumerate(gates):
        arity = 2 if isinstance(gate, CliffordGate) and gate.gate == "CNOT" else 1
        axes = getattr(gate, "qubits", None)
        if (
            not isinstance(gate, CliffordGate)
            or gate.gate not in ("H", "S", "CNOT")
            or type(axes) is not tuple
            or len(axes) != arity
            or any(type(axis) is not str for axis in axes)
            or len(set(axes)) != arity
            or any(
                type(axis) is not str
                or not axis
                or len(axis) > 64
                or any(0xD800 <= ord(char) <= 0xDFFF for char in axis)
                or axis not in ids
                for axis in axes
            )
        ):
            _reject(
                f"{location}.gates[{index}]",
                "quantum.stabilizer_clifford_sequence.invalid_gate",
                "gate axes must be valid distinct labels in the sequence register",
            )
    return value, count


def _resolve_gates(
    sequence: StabilizerCliffordSequence,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    ids = sequence.register.qubit_ids
    return tuple(
        (gate.gate, tuple(ids.index(axis) for axis in gate.qubits))
        for gate in sequence.gates
    )


def _admit_group(value: object) -> tuple[ExactStabilizerGroup, int, int]:
    if not isinstance(value, ExactStabilizerGroup):
        _reject(
            "group",
            "quantum.stabilizer_clifford_sequence.invalid_group",
            "input must be an exact stabilizer group",
        )
    register = _admit_register(value.qubit_register, "group.register")
    generators = value.generators
    count = len(generators) if isinstance(generators, tuple) else -1
    width = len(register.qubit_ids)
    if not 0 <= count <= MAX_CHECK_ROWS:
        _reject(
            "group.generators",
            "quantum.stabilizer_clifford_sequence.invalid_group_size",
            "generator family exceeds its admitted row count",
        )
    if any(
        not isinstance(generator, ExactQubitPauli)
        or not isinstance(generator.phase_free, PhaseFreeQubitPauli)
        or generator.register != register
        or type(generator.phase) is not int
        or not 0 <= generator.phase < 4
        or type(generator.phase_free.x_bits) is not tuple
        or type(generator.phase_free.z_bits) is not tuple
        or len(generator.phase_free.x_bits) != width
        or len(generator.phase_free.z_bits) != width
        or any(
            type(bit) is not int or bit not in (0, 1)
            for bit in (*generator.phase_free.x_bits, *generator.phase_free.z_bits)
        )
        for generator in generators
    ):
        _reject(
            "group.generators",
            "quantum.stabilizer_clifford_sequence.invalid_group",
            "generators must be exact Pauli values on the identical register",
        )
    pair_count = count * (count - 1) // 2
    source_validation = (
        2 * pair_count * width
        + 8 * count * min(count, width) * width
        + width * 68
        + 2 * count * width * 68
        + count * width
    )
    return value, count, source_validation


def compose_stabilizer_clifford_sequences(
    request: CliffordSequenceCompositionRequest,
) -> StabilizerCliffordSequence:
    """Compose two exact gate sequences, applying ``left`` before ``right``."""
    if not isinstance(request, CliffordSequenceCompositionRequest):
        _reject(
            "request",
            "quantum.stabilizer_clifford_sequence.invalid_request",
            "request must contain two typed Clifford sequences",
        )
    left, left_count = _admit_sequence_shape(request.left, "left")
    right, right_count = _admit_sequence_shape(request.right, "right", left.register)
    total_count = left_count + right_count
    if total_count > MAX_CLIFFORD_SEQUENCE_GATES:
        raise OperationResourceAdmissionError(
            location=("right", "gates"),
            code="quantum.stabilizer_clifford_sequence.composition_too_long",
            message="composed gate count exceeds the finite sequence limit",
        )
    labels = _register_label_bytes(left.register) + sum(
        6 * len(axis) + 4
        for gate in chain(left.gates, right.gates)
        for axis in gate.qubits
    )
    work = (
        (left_count + right_count) * len(left.register.qubit_ids)
        + sum(
            len(axis) for gate in chain(left.gates, right.gates) for axis in gate.qubits
        )
        + total_count * 8
    )
    output_bound = labels + total_count * 80 + 256
    if (
        work > MAX_SEQUENCE_COMPOSITION_WORK
        or output_bound > MAX_SEQUENCE_COMPOSITION_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="quantum.stabilizer_clifford_sequence.composition_over_envelope",
            message="sequence composition work or exact output size exceeds its envelope",
        )
    return StabilizerCliffordSequence(
        register=left.register,
        gates=(*left.gates, *right.gates),
    )


def _conjugate_bits(
    x: list[int],
    z: list[int],
    phase: int,
    actions: tuple[tuple[str, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    for gate, axes in actions:
        first = axes[0]
        if gate == "H":
            phase = (phase + 2 * x[first] * z[first]) % 4
            x[first], z[first] = z[first], x[first]
        elif gate == "S":
            phase = (phase + x[first]) % 4
            z[first] ^= x[first]
        else:
            target = axes[1]
            old_x_control = x[first]
            old_z_target = z[target]
            x[target] ^= old_x_control
            z[first] ^= old_z_target
    return tuple(x), tuple(z), phase


def apply_stabilizer_clifford_sequence(
    request: StabilizerCliffordSequenceApplyRequest,
) -> ExactStabilizerGroup:
    """Apply a bounded finite Clifford sequence by exact group conjugation."""
    if not isinstance(request, StabilizerCliffordSequenceApplyRequest):
        _reject(
            "request",
            "quantum.stabilizer_clifford_sequence.invalid_request",
            "request must contain an exact group and typed finite sequence",
        )
    group, _generator_count, source_validation = _admit_group(request.group)
    register = group.register
    sequence, gate_count = _admit_sequence_shape(request.sequence, "sequence", register)
    width = len(register.qubit_ids)
    register_bytes = _register_label_bytes(register)
    gate_axis_bytes = sum(
        6 * len(axis) + 4 for gate in sequence.gates for axis in gate.qubits
    )
    axis_resolution_work = (
        gate_count * (4 * width + 4 * register_bytes) + gate_axis_bytes
    )
    if (
        source_validation > 1_000_000
        or source_validation + axis_resolution_work > MAX_SEQUENCE_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="quantum.stabilizer_clifford_sequence.over_envelope",
            message="group validation, sequence action, or result exceeds its envelope",
        )

    # Resolve each gate axis once and mutate each canonical generator in place.
    actions = _resolve_gates(sequence)
    canonical = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(
            qubit_register=register,
            generators=group.generators,
        )
    )
    canonical_count = len(canonical.generators)
    transformation_work = canonical_count * (3 * width + 8 * gate_count)
    output_bound = (
        (canonical_count + 1) * register_bytes
        + canonical_count * (24 * width + 128)
        + 256
    )
    if (
        source_validation + axis_resolution_work + transformation_work
        > MAX_SEQUENCE_WORK
        or output_bound > MAX_SEQUENCE_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="quantum.stabilizer_clifford_sequence.over_envelope",
            message="group validation, sequence action, or result exceeds its envelope",
        )
    result_generators = []
    for generator in canonical.generators:
        x, z, phase = _conjugate_bits(
            list(generator.phase_free.x_bits),
            list(generator.phase_free.z_bits),
            generator.phase,
            actions,
        )
        result_generators.append(
            ExactQubitPauli(
                phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
                phase=phase,
            )
        )
    return ExactStabilizerGroup(
        qubit_register=register,
        generators=tuple(result_generators),
    )


__all__ = [
    "apply_stabilizer_clifford_sequence",
    "compose_stabilizer_clifford_sequences",
]
