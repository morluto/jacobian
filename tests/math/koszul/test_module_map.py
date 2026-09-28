"""Functorial transport of finite-module Koszul complexes."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulChainMap,
    ModuleKoszulMapRequest,
)
from jacobian.math.koszul.module_operations import module_koszul_map


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _dual_numbers() -> tuple[FiniteCommutativeAlgebra, BasedFiniteModule]:
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


def test_multiplication_by_nilpotent_induces_exact_chain_map() -> None:
    algebra, module = _dual_numbers()
    request = ModuleKoszulMapRequest(
        algebra=algebra,
        source=module,
        target=module,
        sequence=((q(0), q(1)),),
        map_matrix=((q(0), q(0)), (q(1), q(0))),
    )

    result = module_koszul_map(request)
    assert result.source_complex.differentials[0].entries == ((1, 0, q(1)),)
    assert result.degree_maps[0].entries == ((1, 0, q(1)),)
    assert result.degree_maps[1].entries == ((1, 0, q(1)),)

    # The defining square is checked independently in exact coordinates.
    differential = {
        (row, column): Fraction(value.num, value.den)
        for row, column, value in result.source_complex.differentials[0].entries
    }
    degree_map = {
        (row, column): Fraction(value.num, value.den)
        for row, column, value in result.degree_maps[0].entries
    }
    degree_map_1 = {
        (row, column): Fraction(value.num, value.den)
        for row, column, value in result.degree_maps[1].entries
    }
    assert differential == degree_map

    def product(
        left: dict[tuple[int, int], Fraction],
        right: dict[tuple[int, int], Fraction],
    ) -> dict[tuple[int, int], Fraction]:
        result: dict[tuple[int, int], Fraction] = {}
        for (row, middle), left_value in left.items():
            for (right_middle, column), right_value in right.items():
                if middle == right_middle:
                    result[(row, column)] = (
                        result.get((row, column), Fraction(0))
                        + left_value * right_value
                    )
        return {key: value for key, value in result.items() if value}

    # Both sides are the square of multiplication by e and vanish exactly.
    assert (
        product(differential, degree_map_1) == product(degree_map, differential) == {}
    )
    assert (
        ModuleKoszulChainMap.model_validate_json(
            encode_strict_json(result.model_dump(mode="json")), strict=True
        )
        == result
    )


def test_rectangular_map_between_distinct_modules_induces_chain_maps() -> None:
    algebra, target = _dual_numbers()
    source = BasedFiniteModule(
        algebra=algebra,
        basis=("one_mod_e",),
        action=(((q(1),),), ((q(0),),)),
    )
    # A/(e) -> A sends 1 to e. It is nonzero and A-linear, but rectangular.
    result = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=source,
            target=target,
            sequence=((q(0), q(1)),),
            map_matrix=((q(0),), (q(1),)),
        )
    )

    assert result.source != result.target
    assert result.degree_maps[0].row_count == 2
    assert result.degree_maps[0].column_count == 1
    assert result.degree_maps[0].entries == ((1, 0, q(1)),)
    assert result.degree_maps[1].row_count == 2
    assert result.degree_maps[1].column_count == 1
    assert result.degree_maps[1].entries == ((1, 0, q(1)),)
    assert not result.source_complex.differentials[0].entries
    assert result.target_complex.differentials[0].entries == ((1, 0, q(1)),)


def test_non_module_linear_map_is_rejected() -> None:
    algebra, module = _dual_numbers()
    request = ModuleKoszulMapRequest(
        algebra=algebra,
        source=module,
        target=module,
        sequence=((q(0), q(1)),),
        map_matrix=((q(1), q(0)), (q(0), q(0))),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        module_koszul_map(request)
    assert error.value.errors()[0]["type"] == "koszul.module.map_not_linear"


def test_map_to_zero_module_retains_zero_chain_axes() -> None:
    algebra, source = _dual_numbers()
    target = BasedFiniteModule(
        algebra=algebra,
        basis=(),
        action=((), ()),
    )
    result = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=source,
            target=target,
            sequence=((q(0), q(1)),),
            map_matrix=(),
        )
    )
    assert result.target_complex.basis_sizes == (0, 0)
    assert tuple(
        (matrix.row_count, matrix.column_count) for matrix in result.degree_maps
    ) == (
        (0, 2),
        (0, 2),
    )
    assert all(not matrix.entries for matrix in result.degree_maps)


def test_catalog_declares_module_map_transport() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "homological.koszul.module_map.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.source == result.target
    assert result.degree_maps[0].entries == ((0, 0, q(1)),)
