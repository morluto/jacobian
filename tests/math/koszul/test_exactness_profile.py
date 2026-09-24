"""Positive-degree exactness profiles for finite-module Koszul complexes."""

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.koszul._tools import TOOLS
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulHomologyRequest,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_exactness_profile,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _complex(sequence):
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    return module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
    )


def test_unit_sequence_is_acyclic_above_degree_zero():
    result = module_koszul_exactness_profile(
        ModuleKoszulHomologyRequest(complex=_complex(((q(1),),)))
    )
    assert result.homology_dimensions == (0, 0)
    assert result.acyclic_above_zero
    assert result.first_nonzero_degree is None
    assert result.first_nonzero_class is None


def test_zero_sequence_returns_first_higher_homology_class():
    result = module_koszul_exactness_profile(
        ModuleKoszulHomologyRequest(complex=_complex(((q(0),),)))
    )
    assert result.homology_dimensions == (1, 1)
    assert not result.acyclic_above_zero
    assert result.first_nonzero_degree == 1
    assert result.first_nonzero_class == (q(1),)


def test_empty_sequence_has_no_positive_degree_and_does_not_require_h0_vanish():
    result = module_koszul_exactness_profile(
        ModuleKoszulHomologyRequest(complex=_complex(()))
    )
    assert result.homology_dimensions == (1,)
    assert result.acyclic_above_zero


def test_published_exactness_profile_example_runs():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "homological.koszul.exactness_profile.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.acyclic_above_zero
    assert result.homology_dimensions == (0, 0)
