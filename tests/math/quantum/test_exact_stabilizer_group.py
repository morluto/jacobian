"""Exact stabilizer group construction checked in the public Pauli convention."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    ExactStabilizerGroupRequest,
    PhaseFreeQubitPauli,
    QubitRegister,
    pauli_multiply,
    stabilizer_group_from_generators,
)


def _pauli(
    register: QubitRegister,
    x_bits: tuple[int, ...],
    z_bits: tuple[int, ...],
    phase: int,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x_bits, z_bits=z_bits),
        phase=phase,
    )


def test_reduces_redundant_bell_generators_without_losing_phase() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    xx = _pauli(register, (1, 1), (0, 0), 0)
    zz = _pauli(register, (0, 0), (1, 1), 0)
    yy_product = _pauli(register, (1, 1), (1, 1), 0)
    group = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(register=register, generators=(xx, zz, yy_product))
    )

    assert group.register == register
    assert group.generators == (xx, zz)
    # Under the fixed X^x Z^z convention, XX*ZZ is the exact row with
    # x=z=11 and phase zero, so this is a genuine +I dependency.
    assert pauli_multiply(xx, zz).product == yy_product


def test_accepts_positive_identity_dependency_and_empty_trivial_group() -> None:
    register = QubitRegister(qubit_ids=("q",))
    z = _pauli(register, (0,), (1,), 0)
    same_z = _pauli(register, (0,), (1,), 0)
    reduced = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(register=register, generators=(z, same_z))
    )
    trivial = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(register=register, generators=())
    )

    assert reduced.generators == (z,)
    assert trivial.generators == ()


def test_reduces_maximum_dependent_generator_family() -> None:
    register = QubitRegister(qubit_ids=("q",))
    z = _pauli(register, (0,), (1,), 0)
    reduced = stabilizer_group_from_generators(
        ExactStabilizerGroupRequest(register=register, generators=(z,) * 64)
    )
    assert reduced.generators == (z,)


@pytest.mark.parametrize(
    "generators",
    (
        # iX is not Hermitian.
        ((1, 0, 1),),
        # X and Z anticommute.
        ((1, 0, 0), (0, 1, 0)),
        # Z and -Z commute, but their dependency produces -I.
        ((0, 1, 0), (0, 1, 2)),
    ),
)
def test_rejects_nonstabilizer_generator_families(generators) -> None:
    register = QubitRegister(qubit_ids=("q",))
    values = tuple(_pauli(register, (x,), (z,), phase) for x, z, phase in generators)
    with pytest.raises(OperationDomainValidationError):
        stabilizer_group_from_generators(
            ExactStabilizerGroupRequest(register=register, generators=values)
        )


def test_rejects_exact_generators_from_another_ordered_register() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    other = QubitRegister(qubit_ids=("q1", "q0"))
    z0 = _pauli(register, (0, 0), (1, 0), 0)
    z_foreign = _pauli(other, (0, 0), (1, 0), 0)

    with pytest.raises(OperationDomainValidationError):
        stabilizer_group_from_generators(
            ExactStabilizerGroupRequest(register=register, generators=(z0, z_foreign))
        )


def test_exact_group_is_available_through_catalog_dispatch() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.exact_group.from_generators.compute"
    assert catalog.operation(operation_id) is not None
    operation_result = invoke_operation(
        operation_id,
        {
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
        catalog,
    )
    result = ExactStabilizerGroup.model_validate(operation_result.output)
    assert result.generators == (
        _pauli(QubitRegister(qubit_ids=("q",)), (0,), (1,), 0),
    )
