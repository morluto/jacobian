"""Finite Clifford sequence composition and action against exact matrices."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    CliffordGate,
    CliffordSequenceCompositionRequest,
    StabilizerCliffordSequence,
    StabilizerCliffordSequenceApplyRequest,
)
from jacobian.math.quantum.stabilizer_clifford_sequence._tools import TOOLS
from jacobian.math.quantum.stabilizer_clifford_sequence.operations import (
    apply_stabilizer_clifford_sequence,
    compose_stabilizer_clifford_sequences,
)

Gaussian = tuple[int, int]
Matrix = tuple[tuple[Gaussian, ...], ...]
ZERO: Gaussian = (0, 0)
ONE: Gaussian = (1, 0)
IDENTITY: Matrix = (((1, 0), (0, 0)), ((0, 0), (1, 0)))
X: Matrix = (((0, 0), (1, 0)), ((1, 0), (0, 0)))
XZ: Matrix = (((0, 0), (-1, 0)), ((1, 0), (0, 0)))
Z: Matrix = (((1, 0), (0, 0)), ((0, 0), (-1, 0)))
H_SCALED: Matrix = (((1, 0), (1, 0)), ((1, 0), (-1, 0)))
PHASE: Matrix = (((1, 0), (0, 0)), ((0, 0), (0, 1)))


def _add(left: Gaussian, right: Gaussian) -> Gaussian:
    return left[0] + right[0], left[1] + right[1]


def _mul(left: Gaussian, right: Gaussian) -> Gaussian:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _scale(matrix: Matrix, scalar: Gaussian) -> Matrix:
    return tuple(tuple(_mul(scalar, value) for value in row) for row in matrix)


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            _sum(_mul(left[i][k], right[k][j]) for k in range(size))
            for j in range(size)
        )
        for i in range(size)
    )


def _sum(values) -> Gaussian:
    result = ZERO
    for value in values:
        result = _add(result, value)
    return result


def _dagger(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(
            (matrix[row][column][0], -matrix[row][column][1])
            for row in range(len(matrix))
        )
        for column in range(len(matrix[0]))
    )


def _tensor(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            _mul(left[i][j], right[k][column2])
            for j in range(len(left[0]))
            for column2 in range(len(right[0]))
        )
        for i in range(len(left))
        for k in range(len(right))
    )


def _pauli_matrix(pauli: ExactQubitPauli) -> Matrix:
    factors = []
    for x, z in zip(pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True):
        factors.append(((IDENTITY, Z), (X, XZ))[x][z])
    result = _tensor(factors[0], factors[1])
    phase = ((1, 0), (0, 1), (-1, 0), (0, -1))[pauli.phase]
    return _scale(result, phase)


def _gate_matrix(gate: CliffordGate, register: QubitRegister) -> tuple[Matrix, int]:
    axes = tuple(register.qubit_ids.index(axis) for axis in gate.qubits)
    if gate.gate == "CNOT":
        matrix = [[ZERO for _ in range(4)] for _ in range(4)]
        for column in range(4):
            bits = [(column >> 1) & 1, column & 1]
            bits[axes[1]] ^= bits[axes[0]]
            row = (bits[0] << 1) | bits[1]
            matrix[row][column] = ONE
        return tuple(tuple(row) for row in matrix), 0
    factors = [IDENTITY, IDENTITY]
    factors[axes[0]] = H_SCALED if gate.gate == "H" else PHASE
    return _tensor(factors[0], factors[1]), int(gate.gate == "H")


def _sequence_unitary(
    sequence: StabilizerCliffordSequence,
) -> tuple[Matrix, int]:
    identity = _tensor(IDENTITY, IDENTITY)
    result = identity
    h_count = 0
    for gate in sequence.gates:
        unitary, h_scale = _gate_matrix(gate, sequence.register)
        result = _matmul(unitary, result)
        h_count += h_scale
    return result, h_count


def _generated_group(generators: tuple[Matrix, ...]) -> frozenset[Matrix]:
    values = {_tensor(IDENTITY, IDENTITY)}
    for generator in generators:
        values |= {_matmul(value, generator) for value in tuple(values)}
    return frozenset(values)


def _pauli(
    register: QubitRegister,
    x: tuple[int, int],
    z: tuple[int, int],
    phase: int = 0,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
        phase=phase,
    )


def test_sequence_action_matches_independent_exact_matrix_conjugation():
    register = QubitRegister(qubit_ids=("q0", "q1"))
    source_generators = (
        _pauli(register, (0, 0), (1, 0)),
        _pauli(register, (0, 1), (0, 0)),
    )
    group = ExactStabilizerGroup(qubit_register=register, generators=source_generators)
    sequence = StabilizerCliffordSequence(
        register=register,
        gates=(
            CliffordGate(gate="H", qubits=("q0",)),
            CliffordGate(gate="S", qubits=("q1",)),
            CliffordGate(gate="CNOT", qubits=("q1", "q0")),
            CliffordGate(gate="H", qubits=("q1",)),
            CliffordGate(gate="S", qubits=("q0",)),
        ),
    )
    result = apply_stabilizer_clifford_sequence(
        StabilizerCliffordSequenceApplyRequest(group=group, sequence=sequence)
    )

    unitary, hadamard_count = _sequence_unitary(sequence)
    normalization = 1 << hadamard_count
    # Remove the unnormalized Hadamard scale exactly by integer division.
    scaled_expected_generators = tuple(
        _matmul(_matmul(unitary, _pauli_matrix(source)), _dagger(unitary))
        for source in source_generators
    )
    assert all(
        value[0] % normalization == value[1] % normalization == 0
        for matrix in scaled_expected_generators
        for row in matrix
        for value in row
    )
    expected_generators = tuple(
        tuple(
            tuple(
                (value[0] // normalization, value[1] // normalization) for value in row
            )
            for row in matrix
        )
        for matrix in scaled_expected_generators
    )
    assert _generated_group(tuple(_pauli_matrix(row) for row in result.generators)) == (
        _generated_group(expected_generators)
    )
    assert ExactStabilizerGroup.model_validate_json(result.model_dump_json()) == result


def test_sequence_composition_has_empty_identity_and_application_order():
    register = QubitRegister(qubit_ids=("q0", "q1"))
    empty = StabilizerCliffordSequence(register=register, gates=())
    left = StabilizerCliffordSequence(
        register=register,
        gates=(CliffordGate(gate="H", qubits=("q0",)),),
    )
    right = StabilizerCliffordSequence(
        register=register,
        gates=(CliffordGate(gate="S", qubits=("q1",)),),
    )
    compose = compose_stabilizer_clifford_sequences
    assert compose(CliffordSequenceCompositionRequest(left=empty, right=left)) == left
    assert compose(CliffordSequenceCompositionRequest(left=left, right=empty)) == left
    combined = compose(CliffordSequenceCompositionRequest(left=left, right=right))
    assert combined.gates == (*left.gates, *right.gates)
    assert (
        StabilizerCliffordSequence.model_validate_json(combined.model_dump_json())
        == combined
    )
    third = StabilizerCliffordSequence(
        register=register,
        gates=(CliffordGate(gate="CNOT", qubits=("q0", "q1")),),
    )
    left_associated = compose(
        CliffordSequenceCompositionRequest(left=combined, right=third)
    )
    right_associated = compose(
        CliffordSequenceCompositionRequest(
            left=left,
            right=compose(CliffordSequenceCompositionRequest(left=right, right=third)),
        )
    )
    assert left_associated == right_associated


def test_composed_action_matches_sequential_group_action():
    register = QubitRegister(qubit_ids=("q0", "q1"))
    source = ExactStabilizerGroup(
        qubit_register=register,
        generators=(
            _pauli(register, (0, 0), (1, 0)),
            _pauli(register, (0, 1), (0, 0)),
        ),
    )
    left = StabilizerCliffordSequence(
        register=register,
        gates=(CliffordGate(gate="H", qubits=("q0",)),),
    )
    right = StabilizerCliffordSequence(
        register=register,
        gates=(CliffordGate(gate="CNOT", qubits=("q0", "q1")),),
    )
    combined = compose_stabilizer_clifford_sequences(
        CliffordSequenceCompositionRequest(left=left, right=right)
    )
    combined_group = apply_stabilizer_clifford_sequence(
        StabilizerCliffordSequenceApplyRequest(group=source, sequence=combined)
    )
    after_left = apply_stabilizer_clifford_sequence(
        StabilizerCliffordSequenceApplyRequest(group=source, sequence=left)
    )
    after_both = apply_stabilizer_clifford_sequence(
        StabilizerCliffordSequenceApplyRequest(group=after_left, sequence=right)
    )
    assert _generated_group(
        tuple(_pauli_matrix(row) for row in combined_group.generators)
    ) == (_generated_group(tuple(_pauli_matrix(row) for row in after_both.generators)))


def test_empty_sequence_is_identity_on_a_stabilizer_group():
    register = QubitRegister(qubit_ids=("q0", "q1"))
    group = ExactStabilizerGroup(
        qubit_register=register,
        generators=(
            _pauli(register, (0, 0), (1, 0)),
            _pauli(register, (0, 1), (0, 0)),
        ),
    )
    identity = StabilizerCliffordSequence(register=register, gates=())
    result = apply_stabilizer_clifford_sequence(
        StabilizerCliffordSequenceApplyRequest(group=group, sequence=identity)
    )
    assert _generated_group(tuple(_pauli_matrix(row) for row in result.generators)) == (
        _generated_group(tuple(_pauli_matrix(row) for row in group.generators))
    )


def test_composition_rejects_sequence_longer_than_admitted_bound():
    register = QubitRegister(qubit_ids=("q0",))
    h = CliffordGate(gate="H", qubits=("q0",))
    left = StabilizerCliffordSequence(register=register, gates=(h,) * 65)
    right = StabilizerCliffordSequence(register=register, gates=(h,) * 64)
    with pytest.raises(OperationResourceAdmissionError):
        compose_stabilizer_clifford_sequences(
            CliffordSequenceCompositionRequest(left=left, right=right)
        )


def test_owner_local_manifest_publishes_both_sequence_operations():
    assert {tool.operation_id for tool in TOOLS} == {
        "quantum.stabilizer.clifford_sequence.compose.compute",
        "quantum.stabilizer.clifford_sequence.apply.compute",
    }
