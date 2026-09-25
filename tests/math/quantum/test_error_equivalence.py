"""Exact stabilizer cosets checked against exhaustive finite enumeration."""

from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    stabilizer_error_equivalence,
)


def _pauli(register: QubitRegister, bits: tuple[int, ...]) -> PhaseFreeQubitPauli:
    n = len(register.qubit_ids)
    return PhaseFreeQubitPauli(register=register, x_bits=bits[:n], z_bits=bits[n:])


def test_error_cosets_match_exhaustive_stabilizer_span() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    zz = _pauli(register, (0, 0, 1, 1))
    checks = CheckSpaceValue(register=register, basis=(zz,))
    errors = tuple(_pauli(register, bits) for bits in product((0, 1), repeat=4))
    stabilizer_span = {(0, 0, 0, 0), (0, 0, 1, 1)}
    for left in errors:
        for right in errors:
            delta = tuple(
                (a + b) % 2
                for a, b in zip(
                    (*left.x_bits, *left.z_bits),
                    (*right.x_bits, *right.z_bits),
                    strict=True,
                )
            )
            result = stabilizer_error_equivalence(checks, left, right)
            assert result.difference == _pauli(register, delta)
            assert result.equivalent_mod_stabilizers == (delta in stabilizer_span)


def test_same_syndrome_can_be_distinct_logical_error_cosets() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    zz = _pauli(register, (0, 0, 1, 1))
    checks = CheckSpaceValue(register=register, basis=(zz,))
    x0 = _pauli(register, (1, 0, 0, 0))
    x1 = _pauli(register, (0, 1, 0, 0))
    assert not stabilizer_error_equivalence(checks, x0, x1).equivalent_mod_stabilizers
    assert stabilizer_error_equivalence(checks, x0, x0).equivalent_mod_stabilizers


def test_rejects_nonisotropic_checks_and_foreign_register() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_equivalence(
            CheckSpaceValue(register=register, basis=(x, z)), x, z
        )

    other = QubitRegister(qubit_ids=("other",))
    foreign = _pauli(other, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_equivalence(
            CheckSpaceValue(register=register, basis=(z,)), x, foreign
        )


def test_error_equivalence_rejects_model_constructed_check_spaces() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    forged_values = (
        CheckSpaceValue.model_construct(),
        CheckSpaceValue.model_construct(qubit_register=register),
        CheckSpaceValue.model_construct(qubit_register=register, basis="ab"),
        CheckSpaceValue.model_construct(qubit_register=register, basis=(None, "x") * 4),
    )
    for forged in forged_values:
        with pytest.raises(OperationDomainValidationError):
            stabilizer_error_equivalence(forged, x, x)
