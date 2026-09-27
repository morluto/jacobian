from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    QubitRegister,
    StabilizerCodeRequest,
)
from jacobian.math.quantum.operations import stabilizer_code_compute


def _pauli(
    register: QubitRegister,
    x: tuple[int, ...],
    z: tuple[int, ...],
    phase: int = 0,
) -> ExactQubitPauli:
    from jacobian.math.quantum._models import PhaseFreeQubitPauli

    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
        phase=phase,
    )


def _small_pauli_matrix(pauli: ExactQubitPauli) -> np.ndarray:
    """Independent computational-basis oracle for at most two qubits."""
    n = len(pauli.register.qubit_ids)
    dimension = 1 << n
    matrix = np.zeros((dimension, dimension), dtype=complex)
    flip_mask = sum(
        bit << (n - qubit - 1) for qubit, bit in enumerate(pauli.phase_free.x_bits)
    )
    for basis in range(dimension):
        sign = (-1) ** sum(
            pauli.phase_free.z_bits[qubit] * ((basis >> (n - qubit - 1)) & 1)
            for qubit in range(n)
        )
        matrix[basis ^ flip_mask, basis] = (1j**pauli.phase) * sign
    return matrix


def _eigenspace_projector(
    generators: tuple[ExactQubitPauli, ...], eigenvalues: tuple[int, ...]
) -> np.ndarray:
    dimension = 1 << len(generators[0].register.qubit_ids)
    identity = np.eye(dimension, dtype=complex)
    projector = identity.copy()
    for generator, eigenvalue in zip(generators, eigenvalues, strict=True):
        projector = (
            projector @ (identity + eigenvalue * _small_pauli_matrix(generator)) / 2
        )
    return projector


def test_code_character_selects_plus_and_minus_bell_eigenspaces() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    xx = _pauli(register, (1, 1), (0, 0))
    zz = _pauli(register, (0, 0), (1, 1))
    group = ExactStabilizerGroup(register=register, generators=(xx, zz))

    bell = stabilizer_code_compute(
        StabilizerCodeRequest(group=group, generator_eigenvalues=(1, 1))
    )
    orthogonal_bell = stabilizer_code_compute(
        StabilizerCodeRequest(group=group, generator_eigenvalues=(-1, 1))
    )

    assert bell.logical_qubits == 0
    assert orthogonal_bell.logical_qubits == 0
    assert bell.group.generators[0].phase == 0
    assert orthogonal_bell.group.generators[0].phase == 2
    assert bell.group.generators[1].phase == 0
    assert orthogonal_bell.group.generators[1].phase == 0


def test_equivalent_generator_bases_produce_the_same_canonical_code() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    xx = _pauli(register, (1, 1), (0, 0))
    zz = _pauli(register, (0, 0), (1, 1))
    minus_yy = _pauli(register, (1, 1), (1, 1), phase=0)
    original = stabilizer_code_compute(
        StabilizerCodeRequest(
            group=ExactStabilizerGroup(register=register, generators=(xx, zz)),
            generator_eigenvalues=(1, 1),
        )
    )
    changed_basis = stabilizer_code_compute(
        StabilizerCodeRequest(
            group=ExactStabilizerGroup(register=register, generators=(xx, minus_yy)),
            generator_eigenvalues=(1, 1),
        )
    )
    assert changed_basis == original
    canonical_projector = _eigenspace_projector(original.group.generators, (1, 1))
    source_projector = _eigenspace_projector((xx, zz), (1, 1))
    alternative_source_projector = _eigenspace_projector((xx, minus_yy), (1, 1))
    np.testing.assert_array_equal(canonical_projector, source_projector)
    np.testing.assert_array_equal(canonical_projector, alternative_source_projector)
    assert np.trace(canonical_projector) == 1


def test_negative_generator_and_character_are_canonicalized_together() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    negative_z = _pauli(register, (0,), (1,), phase=2)
    negative_group = ExactStabilizerGroup(register=register, generators=(negative_z,))
    value = stabilizer_code_compute(
        StabilizerCodeRequest(group=negative_group, generator_eigenvalues=(-1,))
    )

    assert value.group.generators == (_pauli(register, (0,), (1,)),)


def test_character_requires_one_strict_sign_per_generator() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    group = ExactStabilizerGroup(
        register=register, generators=(_pauli(register, (0,), (1,)),)
    )
    with pytest.raises(ValidationError):
        StabilizerCodeRequest(group=group, generator_eigenvalues=(0,))
    with pytest.raises(ValidationError):
        StabilizerCodeRequest(group=group, generator_eigenvalues=())


def test_no_checks_preserve_the_entire_register() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    value = stabilizer_code_compute(
        StabilizerCodeRequest(
            group=ExactStabilizerGroup(register=register, generators=()),
            generator_eigenvalues=(),
        )
    )
    assert value.logical_qubits == 2
    assert value.group.generators == ()


def test_maximum_register_with_full_rank_code_is_accepted() -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(32)))
    generators = tuple(
        _pauli(
            register,
            tuple(int(index == qubit) for index in range(32)),
            (0,) * 32,
        )
        for qubit in range(32)
    )
    value = stabilizer_code_compute(
        StabilizerCodeRequest(
            group=ExactStabilizerGroup(register=register, generators=generators),
            generator_eigenvalues=(1,) * 32,
        )
    )
    assert value.logical_qubits == 0
    assert len(value.group.generators) == 32


def test_dependent_group_with_inconsistent_character_is_rejected() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    z = _pauli(register, (0,), (1,))
    # A forged value can bypass the structural validator. The operation must
    # establish generator independence before interpreting the character.
    dependent_group = ExactStabilizerGroup.model_construct(
        qubit_register=register,
        generators=(z, z),
    )
    request = StabilizerCodeRequest.model_construct(
        group=dependent_group,
        generator_eigenvalues=(1, -1),
    )
    with pytest.raises(OperationDomainValidationError, match="independent generator"):
        stabilizer_code_compute(request)


def test_code_value_is_published_and_roundtrips_through_catalog() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.code.compute"
    assert catalog.operation(operation_id) is not None
    raw = invoke_operation(
        operation_id,
        {
            "group": {
                "register": {"qubit_ids": ["q"]},
                "generators": [
                    {
                        "phase_free": {
                            "register": {"qubit_ids": ["q"]},
                            "x_bits": [0],
                            "z_bits": [1],
                        },
                        "phase": 0,
                    }
                ],
            },
            "generator_eigenvalues": [-1],
        },
        catalog,
    )
    from jacobian.math.quantum._models import StabilizerCodeValue

    value = StabilizerCodeValue.model_validate(raw.output)
    assert value.logical_qubits == 0
    assert value.group.generators[0].phase == 2
