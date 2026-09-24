"""Exact finite-algebra Koszul DGA products and differential identities."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul import dga_operations
from jacobian.math.koszul._tools import TOOLS
from jacobian.math.koszul.module_models import (
    FiniteCommutativeAlgebra,
    ModuleKoszulDGA,
    ModuleKoszulDGARequest,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _dual_numbers() -> FiniteCommutativeAlgebra:
    return FiniteCommutativeAlgebra(
        basis=("1", "e"),
        multiplication=(
            ((_q(1), _q(0)), (_q(0), _q(1))),
            ((_q(0), _q(1)), (_q(0), _q(0))),
        ),
        unit=(_q(1), _q(0)),
    )


def _products(value, degree_left, index_left, degree_right, index_right):
    return {
        entry.result_index: entry.coefficient.as_fraction()
        for entry in value.products
        if (
            entry.left_degree,
            entry.left_index,
            entry.right_degree,
            entry.right_index,
        )
        == (degree_left, index_left, degree_right, index_right)
    }


def _differential(value, degree, index):
    if degree == 0:
        return {}
    matrix = value.complex.differentials[degree - 1]
    return {
        row: coefficient.as_fraction()
        for row, column, coefficient in matrix.entries
        if column == index
    }


def _linear_product(value, left_degree, left, right_degree, right):
    result = {}
    for left_index, left_coefficient in left.items():
        for right_index, right_coefficient in right.items():
            for target, coefficient in _products(
                value, left_degree, left_index, right_degree, right_index
            ).items():
                result[target] = result.get(target, Fraction(0)) + (
                    left_coefficient * right_coefficient * coefficient
                )
    return {index: coefficient for index, coefficient in result.items() if coefficient}


def _replay_dga_identities(value):
    basis_sizes = value.complex.basis_sizes
    unit = {
        index: coefficient.as_fraction()
        for index, coefficient in enumerate(value.unit_coordinates)
        if coefficient.num
    }
    for left_degree, left_size in enumerate(basis_sizes):
        for left_index in range(left_size):
            left = {left_index: Fraction(1)}
            assert _linear_product(value, 0, unit, left_degree, left) == left
            assert _linear_product(value, left_degree, left, 0, unit) == left
            for right_degree, right_size in enumerate(basis_sizes):
                for right_index in range(right_size):
                    right = {right_index: Fraction(1)}
                    product_xy = _products(
                        value, left_degree, left_index, right_degree, right_index
                    )
                    product_yx = _products(
                        value, right_degree, right_index, left_degree, left_index
                    )
                    sign = -1 if left_degree * right_degree % 2 else 1
                    assert product_xy == {
                        target: sign * coefficient
                        for target, coefficient in product_yx.items()
                    }

                    for third_degree, third_size in enumerate(basis_sizes):
                        for third_index in range(third_size):
                            third = {third_index: Fraction(1)}
                            left_associated = _linear_product(
                                value,
                                left_degree + right_degree,
                                product_xy,
                                third_degree,
                                third,
                            )
                            right_associated = _linear_product(
                                value,
                                left_degree,
                                left,
                                right_degree + third_degree,
                                _products(
                                    value,
                                    right_degree,
                                    right_index,
                                    third_degree,
                                    third_index,
                                ),
                            )
                            assert left_associated == right_associated

                    if left_degree + right_degree <= len(value.sequence) + 1:
                        differential_product = {}
                        if left_degree + right_degree <= len(value.sequence):
                            for product_index, coefficient in product_xy.items():
                                for target, differential_coefficient in _differential(
                                    value, left_degree + right_degree, product_index
                                ).items():
                                    differential_product[target] = (
                                        differential_product.get(target, Fraction(0))
                                        + coefficient * differential_coefficient
                                    )
                        dleft = _differential(value, left_degree, left_index)
                        dright = _differential(value, right_degree, right_index)
                        first_term = (
                            _linear_product(
                                value,
                                left_degree - 1,
                                dleft,
                                right_degree,
                                right,
                            )
                            if left_degree
                            else {}
                        )
                        second_term = (
                            _linear_product(
                                value,
                                left_degree,
                                left,
                                right_degree - 1,
                                dright,
                            )
                            if right_degree
                            else {}
                        )
                        leibniz = first_term.copy()
                        sign = -1 if left_degree % 2 else 1
                        for target, coefficient in second_term.items():
                            leibniz[target] = (
                                leibniz.get(target, Fraction(0)) + sign * coefficient
                            )
                        assert {
                            index: coefficient
                            for index, coefficient in differential_product.items()
                            if coefficient
                        } == {
                            index: coefficient
                            for index, coefficient in leibniz.items()
                            if coefficient
                        }


def test_dual_number_koszul_dga_has_exact_product_and_leibniz_identity():
    value = dga_operations.module_koszul_dga(
        ModuleKoszulDGARequest(algebra=_dual_numbers(), sequence=((_q(0), _q(1)),))
    )
    assert value.complex.basis_sizes == (2, 2)
    assert value.unit_coordinates == (_q(1), _q(0))
    assert value.complex.differentials[0].entries == ((1, 0, _q(1)),)
    _replay_dga_identities(value)
    assert ModuleKoszulDGA.model_validate_json(value.model_dump_json()) == value


def test_exterior_signs_and_repeated_generators_are_exact():
    rational = FiniteCommutativeAlgebra(
        basis=("1",),
        multiplication=(((_q(1),),),),
        unit=(_q(1),),
    )
    zero = ((_q(0),), (_q(0),))
    value = dga_operations.module_koszul_dga(
        ModuleKoszulDGARequest(algebra=rational, sequence=zero)
    )
    assert value.complex.basis_sizes == (1, 2, 1)
    assert _products(value, 1, 0, 1, 0) == {}
    assert _products(value, 1, 0, 1, 1) == {0: Fraction(1)}
    assert _products(value, 1, 1, 1, 0) == {0: Fraction(-1)}
    _replay_dga_identities(value)


def test_two_generator_differential_satisfies_graded_leibniz_at_top_degree():
    rational = FiniteCommutativeAlgebra(
        basis=("1",),
        multiplication=(((_q(1),),),),
        unit=(_q(1),),
    )
    value = dga_operations.module_koszul_dga(
        ModuleKoszulDGARequest(
            algebra=rational,
            sequence=((_q(1),), (_q(1),)),
        )
    )
    assert value.complex.basis_sizes == (1, 2, 1)
    _replay_dga_identities(value)


def test_missing_or_false_unit_is_rejected():
    nonunital = FiniteCommutativeAlgebra(
        basis=("e",), multiplication=(((_q(0),),),), unit=None
    )
    with pytest.raises(OperationDomainValidationError) as missing:
        dga_operations.module_koszul_dga(
            ModuleKoszulDGARequest(algebra=nonunital, sequence=())
        )
    assert missing.value.errors()[0]["type"] == "koszul.module.dga_unit_required"

    false_unit = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((_q(1),),),), unit=(_q(2),)
    )
    with pytest.raises(OperationDomainValidationError) as invalid:
        dga_operations.module_koszul_dga(
            ModuleKoszulDGARequest(algebra=false_unit, sequence=())
        )
    assert invalid.value.errors()[0]["type"] == "koszul.module.dga_invalid_unit"


def test_output_budget_rejects_before_complex_expansion(monkeypatch):
    def fail_if_built(_request):
        raise AssertionError("chain construction ran before DGA admission")

    monkeypatch.setattr(dga_operations, "_build_module_koszul_complex", fail_if_built)
    monkeypatch.setattr(dga_operations, "MAX_KOSZUL_DGA_OUTPUT_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        dga_operations.module_koszul_dga(
            ModuleKoszulDGARequest(algebra=_dual_numbers(), sequence=((_q(0), _q(1)),))
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.dga_output_budget"


def test_published_operation_runs_its_example():
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "homological.koszul.dga.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    value = tool.run(request)
    assert isinstance(value, ModuleKoszulDGA)
    assert _products(value, 1, 0, 1, 1) == {0: Fraction(1)}
