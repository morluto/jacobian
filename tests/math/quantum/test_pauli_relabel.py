"""Pauli register relabelling checked against an independent dense oracle."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError
from sympy import I, Matrix, kronecker_product, zeros

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.pauli_relabel._models import QubitRegisterRelabeling
from jacobian.math.quantum.pauli_relabel.operations import relabel_pauli

_X = Matrix([[0, 1], [1, 0]])
_Z = Matrix([[1, 0], [0, -1]])


def _operator(pauli: ExactQubitPauli) -> Matrix:
    local = tuple(
        _X**x * _Z**z
        for x, z in zip(pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True)
    )
    return I**pauli.phase * kronecker_product(*local)


def _permutation_unitary(
    source: QubitRegister,
    target: QubitRegister,
    target_ids_in_source_order: tuple[str, ...],
) -> Matrix:
    dimension = 2 ** len(source.qubit_ids)
    result = zeros(dimension)
    target_position = {
        identifier: index for index, identifier in enumerate(target.qubit_ids)
    }
    for source_integer in range(dimension):
        source_bits = tuple(
            (source_integer >> (len(source.qubit_ids) - 1 - i)) & 1
            for i in range(len(source.qubit_ids))
        )
        target_bits = [0] * len(target.qubit_ids)
        for bit, label in zip(source_bits, target_ids_in_source_order, strict=True):
            target_bits[target_position[label]] = bit
        target_integer = 0
        for bit in target_bits:
            target_integer = 2 * target_integer + bit
        result[target_integer, source_integer] = 1
    return result


def test_relabeling_agrees_with_dense_tensor_factor_permutation() -> None:
    source = QubitRegister(qubit_ids=("source-0", "source-1"))
    target = QubitRegister(qubit_ids=("target-1", "target-0"))
    mapping = ("target-0", "target-1")
    unitary = _permutation_unitary(source, target, mapping)

    for x_bits in itertools.product((0, 1), repeat=2):
        for z_bits in itertools.product((0, 1), repeat=2):
            for phase in range(4):
                source_pauli = ExactQubitPauli(
                    phase_free=PhaseFreeQubitPauli(
                        register=source, x_bits=x_bits, z_bits=z_bits
                    ),
                    phase=phase,
                )
                result = relabel_pauli(
                    source_pauli,
                    QubitRegisterRelabeling(
                        source_register=source,
                        target_register=target,
                        target_ids_in_source_order=mapping,
                    ),
                )

                assert (
                    _operator(result) == unitary * _operator(source_pauli) * unitary.H
                )
                assert result.register == target
                assert result.phase == phase


def test_relabeling_rejects_a_non_bijection() -> None:
    source = QubitRegister(qubit_ids=("a", "b"))
    pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=source, x_bits=(1, 0), z_bits=(0, 1)),
        phase=0,
    )
    with pytest.raises(ValidationError, match="bijection"):
        QubitRegisterRelabeling(
            source_register=source,
            target_register=QubitRegister(qubit_ids=("c", "d")),
            target_ids_in_source_order=("c", "c"),
        )

    wrong_source = QubitRegisterRelabeling(
        source_register=QubitRegister(qubit_ids=("other-0", "other-1")),
        target_register=QubitRegister(qubit_ids=("c", "d")),
        target_ids_in_source_order=("c", "d"),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        relabel_pauli(pauli, wrong_source)
    assert error.value.errors()[0]["type"] == "quantum.pauli.relabel.source_mismatch"
