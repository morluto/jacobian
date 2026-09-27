"""Exact direct-sum transport for finite-module Koszul complexes."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulDirectSumRequest,
    ModuleKoszulDirectSumValue,
)
from jacobian.math.koszul.module_operations import module_koszul_direct_sum


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_direct_sum_has_partitioning_chain_inclusions_and_exact_block_differential() -> (
    None
):
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    left = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    right = BasedFiniteModule(algebra=algebra, basis=("n",), action=(((q(1),),),))
    value = module_koszul_direct_sum(
        ModuleKoszulDirectSumRequest(
            algebra=algebra,
            left=left,
            right=right,
            sequence=((q(1),),),
        )
    )

    assert value.direct_sum_module.basis == ("L:m", "R:n")
    assert value.direct_sum_complex.basis_sizes == (2, 2)
    # d_1 is the direct sum of the two unit-action maps.
    assert value.direct_sum_complex.differentials[0].entries == (
        (0, 0, q(1)),
        (1, 1, q(1)),
    )
    assert value.left_inclusions == (
        ((q(1),), (q(0),)),
        ((q(1),), (q(0),)),
    )
    assert value.right_inclusions == (
        ((q(0),), (q(1),)),
        ((q(0),), (q(1),)),
    )
    for left_map, right_map in zip(
        value.left_inclusions, value.right_inclusions, strict=True
    ):
        # In concatenated source coordinates, the inclusions form identity.
        combined = tuple(
            (Fraction(lrow[0].num, lrow[0].den), Fraction(rrow[0].num, rrow[0].den))
            for lrow, rrow in zip(left_map, right_map, strict=True)
        )
        assert combined == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    assert ModuleKoszulDirectSumValue.model_validate(value.model_dump()) == value


def test_catalog_declares_the_reusable_direct_sum_postcondition() -> None:
    assert "homological.koszul.module_direct_sum.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }


def test_direct_sum_dimension_is_rejected_before_complex_expansion() -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=tuple(f"m{i}" for i in range(5)),
        action=(
            (
                (q(1), q(0), q(0), q(0), q(0)),
                (q(0), q(1), q(0), q(0), q(0)),
                (q(0), q(0), q(1), q(0), q(0)),
                (q(0), q(0), q(0), q(1), q(0)),
                (q(0), q(0), q(0), q(0), q(1)),
            ),
        ),
    )
    with pytest.raises(ValidationError):
        ModuleKoszulDirectSumRequest(
            algebra=algebra, left=module, right=module, sequence=()
        )
