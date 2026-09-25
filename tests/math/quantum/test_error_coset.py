"""Exact canonical representatives for stabilizer error cosets."""

from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    StabilizerErrorCoset,
    StabilizerErrorCosetRequest,
    stabilizer_error_coset,
)
from jacobian.math.quantum._tools import TOOLS


def _pauli(register: QubitRegister, bits: tuple[int, ...]) -> PhaseFreeQubitPauli:
    n = len(register.qubit_ids)
    return PhaseFreeQubitPauli(register=register, x_bits=bits[:n], z_bits=bits[n:])


def test_canonical_coset_matches_exhaustive_span_and_is_unique() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    zz = _pauli(register, (0, 0, 1, 1))
    checks = CheckSpaceValue(register=register, basis=(zz,))
    errors = tuple(_pauli(register, bits) for bits in product((0, 1), repeat=4))
    span = {(0, 0, 0, 0), (0, 0, 1, 1)}
    values: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for error in errors:
        coset = stabilizer_error_coset(
            StabilizerErrorCosetRequest(check_space=checks, error=error)
        )
        reduced = (*coset.representative.x_bits, *coset.representative.z_bits)
        source = (*error.x_bits, *error.z_bits)
        assert tuple((a + b) % 2 for a, b in zip(source, reduced, strict=True)) in span
        values.setdefault(reduced, set()).add(source)
        assert (
            StabilizerErrorCoset.model_validate(coset.model_dump(mode="json")) == coset
        )
    assert len(values) == 8
    for sources in values.values():
        assert len(sources) == 2


def test_coset_is_not_syndrome_classification() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    checks = CheckSpaceValue(
        register=register,
        basis=(_pauli(register, (0, 0, 1, 1)),),
    )
    x0 = _pauli(register, (1, 0, 0, 0))
    x1 = _pauli(register, (0, 1, 0, 0))
    first = stabilizer_error_coset(
        StabilizerErrorCosetRequest(check_space=checks, error=x0)
    )
    second = stabilizer_error_coset(
        StabilizerErrorCosetRequest(check_space=checks, error=x1)
    )
    assert first != second


def test_coset_rejects_nonisotropic_space_and_foreign_register() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_coset(
            StabilizerErrorCosetRequest(
                check_space=CheckSpaceValue(register=register, basis=(x, z)),
                error=x,
            )
        )
    foreign_register = QubitRegister(qubit_ids=("other",))
    foreign = _pauli(foreign_register, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_coset(
            StabilizerErrorCosetRequest(
                check_space=CheckSpaceValue(register=register, basis=(z,)),
                error=foreign,
            )
        )


def test_serialized_coset_requires_isotropic_rref_and_reduced_representative() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    with pytest.raises(ValueError):
        StabilizerErrorCoset(
            check_space=CheckSpaceValue(register=register, basis=(x, z)),
            representative=_pauli(register, (0, 0)),
        )
    with pytest.raises(ValueError):
        StabilizerErrorCoset(
            check_space=CheckSpaceValue(register=register, basis=(z,)),
            representative=z,
        )


def test_catalog_publishes_coset_operation() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "quantum.stabilizer.error_coset.compute"
    )
    assert operation.request_type is StabilizerErrorCosetRequest
    assert operation.result_type is StabilizerErrorCoset
