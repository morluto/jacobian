"""Exact canonical representatives for stabilizer error cosets."""

from itertools import product

import pytest
from pydantic_core import PydanticCustomError

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
        coset = stabilizer_error_coset(checks, error)
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
    first = stabilizer_error_coset(checks, x0)
    second = stabilizer_error_coset(checks, x1)
    assert first != second


def test_coset_rejects_nonisotropic_space_and_foreign_register() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_coset(
            CheckSpaceValue(register=register, basis=(x, z)), x
        )
    foreign_register = QubitRegister(qubit_ids=("other",))
    foreign = _pauli(foreign_register, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_error_coset(
            CheckSpaceValue(register=register, basis=(z,)), foreign
        )


def test_serialized_coset_requires_canonical_rref_and_reduced_representative() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    with pytest.raises(ValueError):
        StabilizerErrorCoset(
            check_space=CheckSpaceValue(register=register, basis=(z, x)),
            representative=_pauli(register, (0, 0)),
        )
    with pytest.raises(ValueError):
        StabilizerErrorCoset(
            check_space=CheckSpaceValue(register=register, basis=(z,)),
            representative=z,
        )


def test_value_constructor_defers_isotropy_to_the_admitted_operation() -> None:
    register = QubitRegister(qubit_ids=("q",))
    x = _pauli(register, (1, 0))
    z = _pauli(register, (0, 1))
    # ``(x, z)`` is canonical RREF but not isotropic. The constructor stays
    # structural: isotropy is admitted once by the operation, and a consumer
    # re-admits a caller-authored claim only when its result relies on it.
    value = StabilizerErrorCoset(
        check_space=CheckSpaceValue(register=register, basis=(x, z)),
        representative=_pauli(register, (0, 0)),
    )
    assert value.check_space.basis == (x, z)


def test_coset_rejects_a_foreign_register_basis_row() -> None:
    first = QubitRegister(qubit_ids=("q",))
    second = QubitRegister(qubit_ids=("other",))
    row = _pauli(second, (1, 0))
    # A native caller can embed a model-constructed check space whose declared
    # register is A while a basis row lives on B of the same width. The coset
    # validator owns the retained-parent invariant, so it compares every
    # retained row against the declared register.
    forged_check = CheckSpaceValue.model_construct(qubit_register=first, basis=(row,))
    forged = StabilizerErrorCoset.model_construct(
        check_space=forged_check,
        representative=_pauli(first, (0, 0)),
    )
    with pytest.raises(PydanticCustomError) as raised:
        forged.require_canonical_coset()  # type: ignore[operator]
    assert raised.value.type == "stabilizer.error_coset_register"

    with pytest.raises(ValueError):
        StabilizerErrorCoset.model_validate(
            {
                "check_space": CheckSpaceValue.model_construct(
                    qubit_register=first, basis=(row,)
                ),
                "representative": _pauli(first, (0, 0)),
            }
        )


def test_coset_rejects_model_constructed_check_spaces() -> None:
    register = QubitRegister(qubit_ids=("q",))
    error = _pauli(register, (1, 0))
    malformed_spaces = (
        CheckSpaceValue.model_construct(),
        CheckSpaceValue.model_construct(qubit_register=register),
        CheckSpaceValue.model_construct(qubit_register=register, basis="ab"),
        CheckSpaceValue.model_construct(
            qubit_register=register, basis=(None, "x") * 4
        ),
    )
    for check_space in malformed_spaces:
        with pytest.raises(OperationDomainValidationError):
            stabilizer_error_coset(check_space, error)  # type: ignore[arg-type]


def test_catalog_publishes_coset_operation() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "quantum.stabilizer.error_coset.compute"
    )
    assert operation.request_type is StabilizerErrorCosetRequest
    assert operation.result_type is StabilizerErrorCoset
