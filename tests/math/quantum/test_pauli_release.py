"""Exact phase-consistent Pauli and stabilizer normalizer slice."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    CheckSpaceValue,
    ExactQubitPauli,
    NormalizerResult,
    PhaseFreeQubitPauli,
    QubitRegister,
    pauli_multiply,
    pauli_pairing,
    stabilizer_normalizer,
)


def _pauli(
    register: QubitRegister, x: tuple[int, ...], z: tuple[int, ...], phase: int = 0
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
        phase=phase,
    )


def test_pauli_cocycle_and_associativity() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x, z = _pauli(register, (1,), (0,)), _pauli(register, (0,), (1,))
    product = pauli_multiply(x, z).product
    assert product.phase == 0  # XZ is the fixed phase-zero XZ basis element.
    assert product.phase_free.x_bits == (1,) and product.phase_free.z_bits == (1,)
    left = pauli_multiply(pauli_multiply(x, z).product, x).product
    right = pauli_multiply(x, pauli_multiply(z, x).product).product
    assert left == right


def test_pairing_commutation_and_register_parent_rejection() -> None:
    first = QubitRegister(qubit_ids=("q0", "q1"))
    second = QubitRegister(qubit_ids=("a", "b"))
    x = PhaseFreeQubitPauli(register=first, x_bits=(1, 0), z_bits=(0, 0))
    z = PhaseFreeQubitPauli(register=first, x_bits=(0, 0), z_bits=(1, 0))
    assert pauli_pairing(x, z).pairing == 1
    foreign = PhaseFreeQubitPauli(register=second, x_bits=(0, 0), z_bits=(1, 0))
    with pytest.raises(OperationDomainValidationError):
        pauli_pairing(x, foreign)


def test_normalizer_is_s_perp_and_quotient_dimension() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    z0 = PhaseFreeQubitPauli(register=register, x_bits=(0, 0), z_bits=(1, 0))
    value = CheckSpaceValue(register=register, basis=(z0,))
    result = stabilizer_normalizer(value)
    assert result.rank == 1
    assert result.orthogonal_rank == 3
    assert result.logical_dimension == 2
    assert all(pauli_pairing(z0, row).pairing == 0 for row in result.orthogonal_basis)


def test_native_pauli_admission_rejects_forged_shape_and_bits() -> None:
    register = QubitRegister(qubit_ids=("q",))
    forged_shape = PhaseFreeQubitPauli.model_construct(
        qubit_register=register, x_bits=(1, 0), z_bits=(0, 0)
    )
    with pytest.raises(OperationDomainValidationError):
        pauli_pairing(forged_shape, forged_shape)
    forged_bits = PhaseFreeQubitPauli.model_construct(
        qubit_register=register, x_bits=(2,), z_bits=(0,)
    )
    with pytest.raises(OperationDomainValidationError):
        pauli_pairing(forged_bits, forged_bits)


def test_forged_normalizer_rows_cannot_change_register_parent() -> None:
    payload = {
        "check_space": {
            "register": {"qubit_ids": ["q"]},
            "basis": [{"register": {"qubit_ids": ["q"]}, "x_bits": [1], "z_bits": [0]}],
        },
        "orthogonal_basis": [
            {"register": {"qubit_ids": ["foreign"]}, "x_bits": [1], "z_bits": [0]}
        ],
        "rank": 1,
        "orthogonal_rank": 1,
        "logical_dimension": 0,
    }
    with pytest.raises(ValidationError):
        NormalizerResult.model_validate(payload)


def test_normalizer_rejects_nonisotropic_authored_space() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,))
    z = PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(1,))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_normalizer(CheckSpaceValue(register=register, basis=(x, z)))


def test_pauli_coordinate_schema_and_parser_bound_materialization() -> None:
    schema = PhaseFreeQubitPauli.model_json_schema()
    assert schema["properties"]["x_bits"]["maxItems"] == 32
    assert schema["properties"]["z_bits"]["maxItems"] == 32
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(32)))
    value = PhaseFreeQubitPauli(
        register=register,
        x_bits=(0,) * 32,
        z_bits=(1,) * 32,
    )
    assert len(value.x_bits) == 32
    with pytest.raises(ValidationError):
        PhaseFreeQubitPauli.model_validate(
            {
                "register": {"qubit_ids": [f"q{i}" for i in range(32)]},
                "x_bits": [0] * 33,
                "z_bits": [1] * 32,
            }
        )
    normalizer_schema = NormalizerResult.model_json_schema()
    assert normalizer_schema["properties"]["orthogonal_basis"]["maxItems"] == 64
    assert normalizer_schema["properties"]["orthogonal_rank"]["maximum"] == 64
    with pytest.raises(ValidationError):
        NormalizerResult.model_validate(
            {
                "check_space": {
                    "register": {"qubit_ids": ["q"]},
                    "basis": [],
                },
                "orthogonal_basis": [
                    {"register": {"qubit_ids": ["q"]}, "x_bits": [0], "z_bits": [0]}
                ]
                * 65,
                "rank": 0,
                "orthogonal_rank": 65,
                "logical_dimension": 65,
            }
        )


def test_phaseful_pauli_json_round_trip_retains_register() -> None:
    register = QubitRegister(qubit_ids=("q",))
    value = _pauli(register, (1,), (1,), phase=1)
    restored = type(value).model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.register.qubit_ids == ("q",)
