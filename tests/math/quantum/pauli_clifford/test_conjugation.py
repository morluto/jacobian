"""Exact dense-matrix oracle for elementary Pauli Clifford conjugation."""

from itertools import product

import pytest
import sympy as sp

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.pauli_clifford._models import (
    PauliCliffordConjugationRequest,
)
from jacobian.math.quantum.pauli_clifford.operations import conjugate_pauli

_I = sp.eye(2)
_X = sp.Matrix([[0, 1], [1, 0]])
_Z = sp.Matrix([[1, 0], [0, -1]])
_H = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
_S = sp.diag(1, sp.I)


def _dense(pauli: ExactQubitPauli) -> sp.Matrix:
    result = sp.Matrix([[sp.I**pauli.phase]])
    for x, z in zip(pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True):
        result = sp.kronecker_product(result, (_X if x else _I) * (_Z if z else _I))
    return result


def _gate_matrix(gate: str, qubits: tuple[str, ...], ids: tuple[str, ...]) -> sp.Matrix:
    dimension = 1 << len(ids)
    if gate in ("H", "S"):
        local = _H if gate == "H" else _S
        factors = [local if q == qubits[0] else _I for q in ids]
        result = factors[0]
        for factor in factors[1:]:
            result = sp.kronecker_product(result, factor)
        return result
    control = ids.index(qubits[0])
    target = ids.index(qubits[1])
    result = sp.zeros(dimension)
    for source in range(dimension):
        control_bit = (source >> (len(ids) - 1 - control)) & 1
        target_mask = 1 << (len(ids) - 1 - target)
        destination = source ^ target_mask if control_bit else source
        result[destination, source] = 1
    return result


def test_elementary_gates_match_exact_dense_conjugation() -> None:
    ids = ("a", "b")
    register = QubitRegister(qubit_ids=ids)
    gate_specs = (
        ("H", ("a",)),
        ("H", ("b",)),
        ("S", ("a",)),
        ("S", ("b",)),
        ("CNOT", ("a", "b")),
        ("CNOT", ("b", "a")),
    )
    for bits in product((0, 1), repeat=4):
        x, z = bits[:2], bits[2:]
        for phase in range(4):
            pauli = ExactQubitPauli(
                phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
                phase=phase,
            )
            for gate, qubits in gate_specs:
                actual = conjugate_pauli(pauli, gate, qubits)
                unitary = _gate_matrix(gate, qubits, ids)
                expected = unitary * _dense(pauli) * unitary.conjugate().T
                assert (expected - _dense(actual)).applyfunc(sp.simplify) == sp.zeros(
                    4
                ), (bits, phase, gate, qubits, actual)


def test_gate_roles_must_bind_to_distinct_register_qubits() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(0,)),
        phase=0,
    )
    try:
        PauliCliffordConjugationRequest(pauli=pauli, gate="CNOT", qubits=("q0", "q0"))
    except Exception as exc:
        assert "distinct" in str(exc)
    else:
        raise AssertionError("CNOT control and target cannot be the same qubit")


def test_register_boundary_is_accepted_and_work_overflow_is_refused() -> None:
    ids = tuple(f"q{index}" for index in range(32))
    register = QubitRegister(qubit_ids=ids)
    pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=register, x_bits=(0,) * 32, z_bits=(0,) * 32
        ),
        phase=0,
    )
    result = conjugate_pauli(pauli, "H", (ids[-1],))
    assert result == pauli

    long_ids = tuple(f"{index:02d}" + "x" * 62 for index in range(32))
    long_register = QubitRegister(qubit_ids=long_ids)
    long_pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=long_register, x_bits=(0,) * 32, z_bits=(0,) * 32
        ),
        phase=0,
    )
    assert conjugate_pauli(long_pauli, "H", (long_ids[-1],)) == long_pauli


def test_elementary_clifford_generators_compose_with_exact_inverses() -> None:
    register = QubitRegister(qubit_ids=("control", "target"))
    source = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=(1, 1), z_bits=(1, 0)),
        phase=3,
    )

    h_twice = source
    for _ in range(2):
        h_twice = conjugate_pauli(h_twice, "H", ("control",))
    assert h_twice == source

    s_four_times = source
    for _ in range(4):
        s_four_times = conjugate_pauli(s_four_times, "S", ("target",))
    assert s_four_times == source

    cnot_twice = source
    for _ in range(2):
        cnot_twice = conjugate_pauli(
            cnot_twice,
            "CNOT",
            ("target", "control"),
        )
    assert cnot_twice == source


def test_gate_axes_are_bound_to_the_exact_register() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,)),
        phase=0,
    )
    request = PauliCliffordConjugationRequest(pauli=pauli, gate="H", qubits=("q1",))
    with pytest.raises(OperationDomainValidationError, match="register elements"):
        conjugate_pauli(request.pauli, request.gate, request.qubits)
