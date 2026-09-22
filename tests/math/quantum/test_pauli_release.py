"""Exact phase-consistent Pauli and stabilizer normalizer slice."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    CheckSpaceValue,
    ExactQubitPauli,
    PhaseFreeQubitPauli,
    QubitRegister,
    pauli_multiply,
    pauli_pairing,
    stabilizer_normalizer,
)
from jacobian.math.quantum._models import (
    NormalizerResult,
    PauliInverseResult,
    PauliPairingResult,
    PauliProductResult,
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


def test_normalizer_rejects_nonisotropic_authored_space() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,))
    z = PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(1,))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_normalizer(CheckSpaceValue(register=register, basis=(x, z)))


def test_forged_native_pauli_bits_are_rejected_before_result_building() -> None:
    register = QubitRegister(qubit_ids=("q",))
    forged = PhaseFreeQubitPauli.model_construct(
        qubit_register=register, x_bits=(2,), z_bits=(0,)
    )
    with pytest.raises(OperationDomainValidationError) as raised:
        pauli_pairing(forged, forged)
    assert raised.value.errors()[0]["type"] == "quantum.pauli.invalid_bits"


def test_normalizer_result_rejects_a_foreign_register_row() -> None:
    first = QubitRegister(qubit_ids=("q",))
    second = QubitRegister(qubit_ids=("other",))
    check = CheckSpaceValue(
        register=first,
        basis=(PhaseFreeQubitPauli(register=first, x_bits=(0,), z_bits=(1,)),),
    )
    foreign = PhaseFreeQubitPauli(register=second, x_bits=(0,), z_bits=(1,))
    with pytest.raises(ValueError, match="normalizer_parent"):
        NormalizerResult.model_validate(
            {
                "check_space": check,
                "orthogonal_basis": [foreign],
                "rank": 1,
                "orthogonal_rank": 1,
                "logical_dimension": 0,
            }
        )

    forged_check = CheckSpaceValue.model_construct(
        qubit_register=first, basis=(foreign,)
    )
    with pytest.raises(ValueError):
        NormalizerResult.model_validate(
            {
                "check_space": forged_check,
                "orthogonal_basis": [foreign],
                "rank": 1,
                "orthogonal_rank": 1,
                "logical_dimension": 0,
            }
        )


def test_result_carriers_reject_foreign_registers_and_pairing_claims() -> None:
    first = QubitRegister(qubit_ids=("q",))
    second = QubitRegister(qubit_ids=("other",))
    left = _pauli(first, (1,), (0,))
    right = _pauli(second, (0,), (1,))
    foreign = _pauli(second, (0,), (0,))
    with pytest.raises(ValueError, match="register_binding"):
        PauliProductResult.model_validate(
            {"left": left, "right": right, "product": foreign}
        )
    with pytest.raises(ValueError, match="register_binding"):
        PauliInverseResult.model_validate({"source": left, "inverse": foreign})
    with pytest.raises(ValueError, match="register_binding"):
        PauliPairingResult.model_validate(
            {
                "left": left.phase_free,
                "right": right.phase_free,
                "pairing": 1,
                "commute": True,
            }
        )


def test_normalizer_result_bounds_orthogonal_carrier() -> None:
    register = QubitRegister(qubit_ids=("q",))
    row = PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(0,))
    check = CheckSpaceValue(register=register, basis=())
    with pytest.raises(ValueError):
        NormalizerResult.model_validate(
            {
                "check_space": check,
                "orthogonal_basis": [row] * (2 * 32 + 1),
                "rank": 0,
                "orthogonal_rank": 2 * 32 + 1,
                "logical_dimension": 2 * 32 + 1,
            }
        )


def test_phaseful_pauli_json_round_trip_retains_register() -> None:
    register = QubitRegister(qubit_ids=("q",))
    value = _pauli(register, (1,), (1,), phase=1)
    restored = type(value).model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.register.qubit_ids == ("q",)
