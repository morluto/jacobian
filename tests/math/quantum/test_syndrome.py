"""Exact syndrome values on bounded binary stabilizer check spaces."""

from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    stabilizer_syndrome,
)


def _pauli(register: QubitRegister, x: tuple[int, ...], z: tuple[int, ...]):
    return PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z)


def _pairing(left: tuple[int, ...], right: tuple[int, ...], n: int) -> int:
    return sum(left[i] * right[n + i] + left[n + i] * right[i] for i in range(n)) % 2


def test_syndromes_match_direct_symplectic_oracle_for_every_two_qubit_error() -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    # A dependent presentation must produce one canonical check coordinate.
    zz = _pauli(register, (0, 0), (1, 1))
    checks = CheckSpaceValue(register=register, basis=(zz, zz))
    for bits in product((0, 1), repeat=4):
        error = _pauli(register, bits[:2], bits[2:])
        result = stabilizer_syndrome(checks, error)
        expected = _pairing((0, 0, 1, 1), bits, 2)
        assert result.syndrome == (expected,)
        assert result.zero_syndrome is (expected == 0)
        assert result.check_space.basis == (zz,)
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored == result


def test_zero_syndrome_does_not_claim_stabilizer_equivalence() -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    zz = _pauli(register, (0, 0), (1, 1))
    check_space = CheckSpaceValue(register=register, basis=(zz,))
    identity = _pauli(register, (0, 0), (0, 0))
    logical_z = _pauli(register, (0, 0), (1, 0))
    identity_result = stabilizer_syndrome(check_space, identity)
    logical_result = stabilizer_syndrome(check_space, logical_z)
    assert identity_result.syndrome == logical_result.syndrome == (0,)
    assert logical_z != zz  # The zero-syndrome bit is not a coset classifier.


def test_empty_check_space_has_empty_zero_syndrome() -> None:
    register = QubitRegister(qubit_ids=("q",))
    error = _pauli(register, (1,), (1,))
    result = stabilizer_syndrome(CheckSpaceValue(register=register, basis=()), error)
    assert result.syndrome == ()
    assert result.zero_syndrome is True


def test_syndrome_rejects_nonisotropic_or_foreign_register_inputs() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1,), (0,))
    z = _pauli(register, (0,), (1,))
    with pytest.raises(OperationDomainValidationError) as raised:
        stabilizer_syndrome(CheckSpaceValue(register=register, basis=(x, z)), x)
    assert raised.value.errors()[0]["type"] == "quantum.stabilizer.not_isotropic"

    foreign = QubitRegister(qubit_ids=("other",))
    with pytest.raises(OperationDomainValidationError) as raised:
        stabilizer_syndrome(
            CheckSpaceValue(register=register, basis=(z,)),
            _pauli(foreign, (1,), (0,)),
        )
    assert raised.value.errors()[0]["type"] == "quantum.pauli.register_mismatch"
