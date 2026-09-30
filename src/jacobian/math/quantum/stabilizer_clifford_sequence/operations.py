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
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.operations import stabilizer_group_from_generators
from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    MAX_CLIFFORD_SEQUENCE_GATES,
    CliffordGate,
    StabilizerCliffordSequence,
)

MAX_SEQUENCE_WORK = 1_500_000
MAX_SEQUENCE_RESULT_CELLS = 65_536
MAX_SEQUENCE_COMPOSITION_WORK = 300_000
MAX_SEQUENCE_COMPOSITION_RESULT_CELLS = 512_000


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _register_width(register: QubitRegister) -> int:
    """Count register axes. Label widths are a transport measure and are
    already bounded by the register model; admission uses the axis count."""
    return len(register.qubit_ids)


def _admit_register(value: object, location: str) -> QubitRegister:
    if not isinstance(value, QubitRegister):
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_register",
            "register must be typed",
        )
    ids = getattr(value, "qubit_ids", None)
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
    register = _admit_register(
        getattr(value, "qubit_register", None), f"{location}.register"
    )
    if expected_register is not None and register != expected_register:
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.register_mismatch",
            "sequence and target must use the identical ordered register",
        )
    gates = getattr(value, "gates", None)
    count = len(gates) if isinstance(gates, tuple) else -1
    if not 0 <= count <= MAX_CLIFFORD_SEQUENCE_GATES:
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_size",
            "sequence exceeds its admitted gate count",
        )
    ids = register.qubit_ids
    if not isinstance(gates, tuple):
        _reject(
            location,
            "quantum.stabilizer_clifford_sequence.invalid_gates",
            "gates must be a canonical tuple",
        )
    for index, gate in enumerate(gates):
        gate_name = getattr(gate, "gate", None)
        axes = getattr(gate, "qubits", None)
        if not isinstance(axes, tuple):
            _reject(
                f"{location}[{index}].qubits",
                "quantum.stabilizer_clifford_sequence.invalid_gate_axes",
                "gate axes must be a canonical tuple",
            )
        arity = 2 if gate_name == "CNOT" else 1
        if (
            not isinstance(gate, CliffordGate)
            or gate_name not in ("H", "S", "CNOT")
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
    id_to_index = {label: index for index, label in enumerate(ids)}
    return tuple(
        (gate.gate, tuple(id_to_index[axis] for axis in gate.qubits))
        for gate in sequence.gates
    )


def _admitted_pauli_bits(phase_free: PhaseFreeQubitPauli, width: int) -> bool:
    """Check binary Pauli coordinates on already-typed values.

    The coordinates are read with ``getattr`` because a native caller can build
    a typed value through ``model_construct`` without them. A missing
    coordinate is a malformed value, not an implementation error, so it is
    reported as an admission failure rather than an ``AttributeError``.
    """
    x_bits = getattr(phase_free, "x_bits", None)
    z_bits = getattr(phase_free, "z_bits", None)
    if type(x_bits) is not tuple or type(z_bits) is not tuple:
        return False
    return (
        len(x_bits) == width
        and len(z_bits) == width
        and all(type(bit) is int and bit in (0, 1) for bit in (*x_bits, *z_bits))
    )


def _admit_group(value: object) -> tuple[ExactStabilizerGroup, int, int]:
    if not isinstance(value, ExactStabilizerGroup):
        _reject(
            "group",
            "quantum.stabilizer_clifford_sequence.invalid_group",
            "input must be an exact stabilizer group",
        )
    register = _admit_register(getattr(value, "qubit_register", None), "group.register")
    generators = getattr(value, "generators", None)
    if not isinstance(generators, tuple):
        _reject(
            "group.generators",
            "quantum.stabilizer_clifford_sequence.invalid_group",
            "group generators must be a canonical tuple",
        )
    count = len(generators)
    width = len(register.qubit_ids)
    if not 0 <= count <= MAX_CHECK_ROWS:
        _reject(
            "group.generators",
            "quantum.stabilizer_clifford_sequence.invalid_group_size",
            "generator family exceeds its admitted row count",
        )

    def is_admitted_generator(generator: object) -> bool:
        if not isinstance(generator, ExactQubitPauli):
            return False
        phase_free = getattr(generator, "phase_free", None)
        if not isinstance(phase_free, PhaseFreeQubitPauli):
            return False
        phase = getattr(generator, "phase", None)
        if (
            getattr(phase_free, "qubit_register", None) != register
            or type(phase) is not int
            or not 0 <= phase < 4
        ):
            return False
        return _admitted_pauli_bits(phase_free, width)

    if any(not is_admitted_generator(generator) for generator in generators):
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
    left_value: StabilizerCliffordSequence,
    right_value: StabilizerCliffordSequence,
) -> StabilizerCliffordSequence:
    """Compose two exact gate sequences, applying ``left`` before ``right``."""
    left, left_count = _admit_sequence_shape(left_value, "left")
    right, right_count = _admit_sequence_shape(right_value, "right", left.register)
    total_count = left_count + right_count
    if total_count > MAX_CLIFFORD_SEQUENCE_GATES:
        raise OperationResourceAdmissionError(
            location=("right", "gates"),
            code="quantum.stabilizer_clifford_sequence.composition_too_long",
            message="composed gate count exceeds the finite sequence limit",
        )
    work = (
        (left_count + right_count) * len(left.register.qubit_ids)
        + sum(
            len(axis) for gate in chain(left.gates, right.gates) for axis in gate.qubits
        )
        + total_count * 8
    )
    output_cells = total_count
    if (
        work > MAX_SEQUENCE_COMPOSITION_WORK
        or output_cells > MAX_SEQUENCE_COMPOSITION_RESULT_CELLS
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
    group_value: ExactStabilizerGroup,
    sequence_value: StabilizerCliffordSequence,
) -> ExactStabilizerGroup:
    """Apply a bounded finite Clifford sequence by exact group conjugation."""
    group, _generator_count, source_validation = _admit_group(group_value)
    register = group.register
    sequence, gate_count = _admit_sequence_shape(sequence_value, "sequence", register)
    width = len(register.qubit_ids)
    register_cells = _register_width(register)
    gate_axis_cells = sum(len(axis) for gate in sequence.gates for axis in gate.qubits)
    axis_resolution_work = register_cells + gate_axis_cells + 8 * gate_count
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
    canonical = stabilizer_group_from_generators(register, group.generators)
    canonical_count = len(canonical.generators)
    transformation_work = canonical_count * (3 * width + 8 * gate_count)
    output_cells = (canonical_count + 1) * register_cells + canonical_count * width
    if (
        source_validation + axis_resolution_work + transformation_work
        > MAX_SEQUENCE_WORK
        or output_cells > MAX_SEQUENCE_RESULT_CELLS
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
        register=register,
        generators=tuple(result_generators),
    )


__all__ = [
    "apply_stabilizer_clifford_sequence",
    "compose_stabilizer_clifford_sequences",
]
