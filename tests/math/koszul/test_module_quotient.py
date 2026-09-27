"""Exact degree-zero quotient module and projection tests."""

import pytest

import jacobian.math.koszul.module_operations as koszul_operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_homology,
    module_koszul_quotient,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def dual_numbers_regular_module() -> tuple[FiniteCommutativeAlgebra, BasedFiniteModule]:
    # Independent presentation: 1*1=1, 1*e=e*1=e, and e*e=0.
    algebra = FiniteCommutativeAlgebra(
        basis=("1", "e"),
        multiplication=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(1)), (q(0), q(0))),
        ),
        unit=(q(1), q(0)),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    return algebra, module


def test_quotient_by_epsilon_matches_hand_computed_dual_number_quotient() -> None:
    algebra, module = dual_numbers_regular_module()
    request = ModuleKoszulRequest(
        algebra=algebra, module=module, sequence=((q(0), q(1)),)
    )

    quotient = module_koszul_quotient(request)

    # e*A is span{e}; reducing a+b*e modulo that subspace leaves a*1.
    assert quotient.relation_basis == ((q(0), q(1)),)
    assert quotient.quotient_basis_representatives == ((q(1), q(0)),)
    assert quotient.projection == ((q(1), q(0)),)
    assert quotient.quotient_module.basis == ("q0",)
    assert quotient.quotient_module.action == (
        ((q(1),),),
        ((q(0),),),
    )

    # H_0 is the same quotient dimension from the Koszul complex.
    homology = module_koszul_homology(module_koszul_complex(request))
    assert homology.dimensions[0] == len(quotient.quotient_module.basis) == 1
    assert homology.degrees[0].cycle_basis == ((q(1), q(0)), (q(0), q(1)))
    assert homology.degrees[0].boundary_basis == ((q(0), q(1)),)
    assert homology.degrees[0].homology_basis == quotient.quotient_basis_representatives
    assert homology.degrees[1].cycle_basis == ((q(0), q(1)),)
    assert homology.degrees[1].boundary_basis == ()
    assert homology.degrees[1].homology_basis == ((q(0), q(1)),)
    assert type(homology).model_validate_json(homology.model_dump_json()) == homology


def test_empty_sequence_and_unit_sequence_preserve_zero_and_identity_cases() -> None:
    algebra, module = dual_numbers_regular_module()
    empty = module_koszul_quotient(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=())
    )
    assert empty.relation_basis == ()
    assert empty.projection == ((q(1), q(0)), (q(0), q(1)))
    assert len(empty.quotient_module.basis) == len(module.basis)

    zero = module_koszul_quotient(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=((q(1), q(0)),))
    )
    assert len(zero.quotient_module.basis) == 0
    assert zero.relation_basis == ((q(1), q(0)), (q(0), q(1)))
    assert zero.projection == ()
    assert zero.quotient_module.action == ((), ())
    # The zero module remains composable through the existing Koszul consumer.
    zero_complex = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=zero.quotient_module, sequence=())
    )
    assert zero_complex.basis_sizes == (0,)
    assert module_koszul_homology(zero_complex).dimensions == (0,)


def test_quotient_value_round_trips_and_is_source_bound() -> None:
    algebra, module = dual_numbers_regular_module()
    result = module_koszul_quotient(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=((q(0), q(1)),))
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert result.source_module == module
    assert result.algebra == algebra


def test_quotient_rejects_predicted_scalar_growth_before_algebra_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("m",),
        action=(((q(10**4000),),),),
    )

    def arithmetic_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("quotient arithmetic ran before coefficient admission")

    monkeypatch.setattr(koszul_operations, "_admit", arithmetic_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as admission:
        module_koszul_quotient(
            ModuleKoszulRequest(algebra=algebra, module=module, sequence=())
        )
    assert admission.value.errors()[0]["type"] == (
        "koszul.module.quotient_coefficient_budget"
    )
