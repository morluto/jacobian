"""Package-native Koszul composition uses mathematical values, not requests."""

import pytest

import jacobian.math.koszul as koszul
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_native_map_homology_change_and_top_composition() -> None:
    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    chain = koszul.module_koszul_map(algebra, module, module, ((q(0),),), ((q(2),),))
    assert chain.degree_maps[0].entries == ((0, 0, q(2)),)
    assert chain.degree_maps[1].entries == ((0, 0, q(2)),)
    decoded = koszul.ModuleKoszulChainMap.model_validate_json(chain.model_dump_json())
    homology = koszul.koszul_homology_map(decoded)
    assert homology.degree_maps == (((q(2),),), ((q(2),),))
    source = koszul.ModuleKoszulComplex.model_validate_json(
        chain.source_complex.model_dump_json()
    )
    changed = koszul.module_koszul_sequence_linear_change(source, ((q(2),),))
    assert changed.target_complex.sequence == ((q(0),),)
    top = koszul.module_koszul_top_homology(changed.target_complex)
    assert top.annihilator_basis == ((q(1),),)


def test_native_map_reports_invalid_domain_axes() -> None:
    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    with pytest.raises(OperationDomainValidationError):
        koszul.module_koszul_map(algebra, module, module, ((q(0),),), ())
