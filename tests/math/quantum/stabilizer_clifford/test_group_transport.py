"""Exact Clifford transport of stabilizer groups against tiny matrix arithmetic."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.stabilizer_clifford._models import (
    StabilizerCliffordTransportRequest,
)
from jacobian.math.quantum.stabilizer_clifford._tools import TOOLS
from jacobian.math.quantum.stabilizer_clifford.operations import (
    conjugate_stabilizer_group,
)

Gaussian = tuple[int, int]
Matrix = tuple[tuple[Gaussian, ...], ...]
ZERO: Gaussian = (0, 0)
ONE: Gaussian = (1, 0)
IMAGINARY: Gaussian = (0, 1)
IDENTITY: Matrix = (((1, 0), (0, 0)), ((0, 0), (1, 0)))
X: Matrix = (((0, 0), (1, 0)), ((1, 0), (0, 0)))
XZ: Matrix = (((0, 0), (-1, 0)), ((1, 0), (0, 0)))
Z: Matrix = (((1, 0), (0, 0)), ((0, 0), (-1, 0)))
H_SCALED: Matrix = (((1, 0), (1, 0)), ((1, 0), (-1, 0)))
S: Matrix = (((1, 0), (0, 0)), ((0, 0), (0, 1)))


def _gadd(left: Gaussian, right: Gaussian) -> Gaussian:
    return left[0] + right[0], left[1] + right[1]


def _gmul(left: Gaussian, right: Gaussian) -> Gaussian:
    return left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[
        1
    ] * right[0]


def _gconjugate(value: Gaussian) -> Gaussian:
    return value[0], -value[1]


def _scale(matrix: Matrix, scalar: Gaussian) -> Matrix:
    return tuple(tuple(_gmul(scalar, value) for value in row) for row in matrix)


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            tuple_sum(_gmul(left[i][k], right[k][j]) for k in range(len(right)))
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


def tuple_sum(values: Iterable[Gaussian]) -> Gaussian:
    result = ZERO
    for value in values:
        result = _gadd(result, value)
    return result


def _dagger(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(_gconjugate(matrix[row][column]) for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _tensor(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            _gmul(left[i][j], right[k][column2])
            for j in range(len(left[0]))
            for column2 in range(len(right[0]))
        )
        for i in range(len(left))
        for k in range(len(right))
    )


def _pauli_matrix(pauli: ExactQubitPauli) -> Matrix:
    local = []
    for x, z in zip(pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True):
        local.append(((IDENTITY, Z), (X, XZ))[x][z])
    result = local[0]
    for factor in local[1:]:
        result = _tensor(result, factor)
    phase = (ONE, IMAGINARY, (-1, 0), (0, -1))[pauli.phase]
    return _scale(result, phase)


def _gate_matrix(gate: str, axes: tuple[str, ...], register: QubitRegister) -> Matrix:
    n = len(register.qubit_ids)
    size = 1 << n
    if gate == "CNOT":
        control, target = (register.qubit_ids.index(axis) for axis in axes)
        rows = [[ZERO for _ in range(size)] for _ in range(size)]
        for column in range(size):
            bits = [(column >> (n - 1 - i)) & 1 for i in range(n)]
            bits[target] ^= bits[control]
            row = sum(bit << (n - 1 - i) for i, bit in enumerate(bits))
            rows[row][column] = ONE
        return tuple(tuple(row) for row in rows)
    axis = register.qubit_ids.index(axes[0])
    local_gate = H_SCALED if gate == "H" else S
    factors = [IDENTITY] * n
    factors[axis] = local_gate
    result = factors[0]
    for factor in factors[1:]:
        result = _tensor(result, factor)
    return result


def _pauli(
    register: QubitRegister,
    x: tuple[int, ...],
    z: tuple[int, ...],
    phase: int = 0,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
        phase=phase,
    )


@pytest.mark.parametrize(
    ("gate", "axes", "normalization"),
    (
        ("H", ("q0",), 2),
        ("S", ("q1",), 1),
        ("CNOT", ("q1", "q0"), 1),
    ),
)
def test_group_transport_matches_independent_dense_matrix_conjugation(
    gate: Literal["H", "S", "CNOT"], axes: tuple[str, ...], normalization: int
) -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    # Independent Z(q0), X(q1) checks span a commuting exact group.
    source_generators = (
        _pauli(register, (0, 0), (1, 0)),
        _pauli(register, (0, 1), (0, 0)),
    )
    group = ExactStabilizerGroup(register=register, generators=source_generators)
    result = conjugate_stabilizer_group(group, gate, axes)
    unitary = _gate_matrix(gate, axes, register)
    assert len(result.generators) == len(source_generators)
    for source, transformed in zip(source_generators, result.generators, strict=True):
        expected = _matmul(_matmul(unitary, _pauli_matrix(source)), _dagger(unitary))
        assert _scale(_pauli_matrix(transformed), (normalization, 0)) == expected


def test_empty_group_transports_as_trivial_group_and_roundtrips() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    group = ExactStabilizerGroup(register=register, generators=())
    result = conjugate_stabilizer_group(group, "H", ("q0",))
    assert result.generators == ()
    assert ExactStabilizerGroup.model_validate_json(result.model_dump_json()) == result


def test_owner_local_manifest_publishes_the_transport_operation() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "quantum.stabilizer.clifford_gate.conjugate.compute",
    )


def test_redundant_source_family_is_reduced_before_transport() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    z = _pauli(register, (0,), (1,))
    group = ExactStabilizerGroup(register=register, generators=(z, z))
    result = conjugate_stabilizer_group(group, "H", ("q0",))
    assert result.generators == (_pauli(register, (1,), (0,)),)


def test_independent_family_order_follows_the_source_presentation() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    z0 = _pauli(register, (0, 0), (1, 0))
    z1 = _pauli(register, (0, 0), (0, 1))
    first_group = ExactStabilizerGroup(register=register, generators=(z0, z1))
    first = conjugate_stabilizer_group(first_group, "H", ("q0",))
    second_group = ExactStabilizerGroup(register=register, generators=(z1, z0))
    second = conjugate_stabilizer_group(second_group, "H", ("q0",))
    # H Z(q0) H† = X(q0) while Z(q1) is untouched. Both results present the
    # same subgroup, but the retained family is not canonicalized, so the
    # serialization follows the source order instead of acting as identity.
    assert first.generators == (_pauli(register, (1, 0), (0, 0)), z1)
    assert second.generators == (z1, _pauli(register, (1, 0), (0, 0)))
    assert first.generators != second.generators


def test_request_schema_publishes_admission_limits() -> None:
    limits = StabilizerCliffordTransportRequest.model_json_schema()["admission_limits"]
    assert limits == {
        "max_qubits": 32,
        "max_source_generators": 64,
        "max_gate_count": 1,
        "max_work_units": 1_000_000,
        "max_result_compact_json_bytes": 65_536,
    }


def test_forged_group_register_is_rejected_before_native_dereference() -> None:
    forged = ExactStabilizerGroup.model_construct(generators=())
    request = StabilizerCliffordTransportRequest.model_construct(
        group=forged, gate="H", qubits=("q0",)
    )
    with pytest.raises(OperationDomainValidationError) as raised:
        conjugate_stabilizer_group(request.group, request.gate, request.qubits)
    assert raised.value.errors()[0]["type"] == (
        "quantum.stabilizer_clifford.invalid_register"
    )


def test_forged_generator_fields_are_rejected_before_native_dereference() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    forged = ExactQubitPauli.model_construct(
        phase_free=PhaseFreeQubitPauli.model_construct(qubit_register=register),
        phase=0,
    )
    group = ExactStabilizerGroup.model_construct(
        qubit_register=register, generators=(forged,)
    )
    request = StabilizerCliffordTransportRequest.model_construct(
        group=group, gate="H", qubits=("q0",)
    )
    with pytest.raises(OperationDomainValidationError) as raised:
        conjugate_stabilizer_group(request.group, request.gate, request.qubits)
    assert raised.value.errors()[0]["type"] == (
        "quantum.stabilizer_clifford.invalid_group"
    )


def test_forged_request_axes_are_rejected_before_native_dereference() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    z = _pauli(register, (0,), (1,))
    group = ExactStabilizerGroup(register=register, generators=(z,))
    with pytest.raises(OperationDomainValidationError) as raised:
        conjugate_stabilizer_group(group, None, ("q0",))
    assert raised.value.errors()[0]["type"] == (
        "quantum.stabilizer_clifford.invalid_gate"
    )
    with pytest.raises(OperationDomainValidationError) as raised:
        conjugate_stabilizer_group(group, "H", ["q0"])
    assert raised.value.errors()[0]["type"] == (
        "quantum.stabilizer_clifford.invalid_axes"
    )


def test_dependent_generators_with_minus_identity_are_rejected() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    z = _pauli(register, (0,), (1,))
    minus_z = _pauli(register, (0,), (1,), phase=2)
    group = ExactStabilizerGroup(register=register, generators=(z, minus_z))
    with pytest.raises(OperationDomainValidationError):
        conjugate_stabilizer_group(group, "H", ("q0",))


def test_cnot_axis_roles_are_directional() -> None:
    register = QubitRegister(qubit_ids=("control", "target"))
    source = _pauli(register, (1, 0), (0, 0))
    group = ExactStabilizerGroup(register=register, generators=(source,))
    result = conjugate_stabilizer_group(group, "CNOT", ("control", "target"))
    assert result.generators == (_pauli(register, (1, 1), (0, 0)),)
