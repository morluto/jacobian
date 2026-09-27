"""Phase-free Pauli addition checked by exhaustive binary vector addition."""

from __future__ import annotations

import itertools

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import PhaseFreeQubitPauli, QubitRegister
from jacobian.math.quantum.phase_free_add.operations import add_phase_free_paulis


def _value(
    register: QubitRegister,
    x_bits: tuple[int, ...],
    z_bits: tuple[int, ...],
) -> PhaseFreeQubitPauli:
    return PhaseFreeQubitPauli(register=register, x_bits=x_bits, z_bits=z_bits)


def test_addition_is_binary_vector_addition_and_preserves_the_register() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    rows = tuple(itertools.product((0, 1), repeat=4))
    for first in rows:
        for second in rows:
            result = add_phase_free_paulis(
                _value(register, first[:2], first[2:]),
                _value(register, second[:2], second[2:]),
            )
            assert result.qubit_register == register
            assert result.x_bits == tuple(
                a ^ b for a, b in zip(first[:2], second[:2], strict=True)
            )
            assert result.z_bits == tuple(
                a ^ b for a, b in zip(first[2:], second[2:], strict=True)
            )
            assert result.weight == sum(
                bool(x or z) for x, z in zip(result.x_bits, result.z_bits, strict=True)
            )


def test_addition_rejects_an_ordered_register_mismatch() -> None:
    left_register = QubitRegister(qubit_ids=("q0", "q1"))
    right_register = QubitRegister(qubit_ids=("q1", "q0"))
    left = _value(left_register, (1, 0), (0, 1))
    right = _value(right_register, (1, 0), (0, 1))

    with pytest.raises(OperationDomainValidationError) as error:
        add_phase_free_paulis(left, right)

    assert (
        error.value.errors()[0]["type"] == "quantum.pauli.phase_free.register_mismatch"
    )
