"""Exact chain splitting for a zero appended to a Koszul sequence."""

import json
from fractions import Fraction

import pytest

import jacobian.math.koszul.module_operations as operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.koszul._tools import TOOLS
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulRequest,
    ModuleKoszulZeroExtensionRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_append_zero,
    module_koszul_complex,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _source(sequence):
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    return module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
    )


def _apply(mapping, vector):
    result = {}
    for row, column, coefficient in mapping.entries:
        value = Fraction(coefficient.num, coefficient.den) * vector.get(
            column, Fraction(0)
        )
        if value:
            result[row] = result.get(row, Fraction(0)) + value
    return {index: value for index, value in result.items() if value}


def _check_splitting(result):
    source_length = len(result.source_complex.sequence)
    for degree, size in enumerate(result.target_complex.basis_sizes):
        for column in range(size):
            basis = {column: Fraction(1)}
            image = {}
            if degree <= source_length:
                image.update(
                    _apply(
                        result.unshifted_inclusions[degree],
                        _apply(result.unshifted_projections[degree], basis),
                    )
                )
            if degree:
                shifted = _apply(
                    result.shifted_inclusions[degree - 1],
                    _apply(result.shifted_projections[degree], basis),
                )
                for row, coefficient in shifted.items():
                    image[row] = image.get(row, Fraction(0)) + coefficient
            assert {row: value for row, value in image.items() if value} == basis

    for degree in range(1, source_length + 1):
        source_d = result.source_complex.differentials[degree - 1]
        target_d = result.target_complex.differentials[degree - 1]
        for column in range(result.source_complex.basis_sizes[degree]):
            basis = {column: Fraction(1)}
            assert _apply(
                result.unshifted_inclusions[degree - 1], _apply(source_d, basis)
            ) == _apply(target_d, _apply(result.unshifted_inclusions[degree], basis))
            assert _apply(
                result.unshifted_projections[degree - 1],
                _apply(target_d, _apply(result.unshifted_inclusions[degree], basis)),
            ) == _apply(source_d, basis)
            shifted_boundary = _apply(
                result.target_complex.differentials[degree],
                _apply(result.shifted_inclusions[degree], basis),
            )
            expected = {
                index: -value
                for index, value in _apply(
                    result.shifted_inclusions[degree - 1], _apply(source_d, basis)
                ).items()
            }
            assert shifted_boundary == expected


def test_appending_zero_returns_the_exact_shifted_direct_sum():
    result = module_koszul_append_zero(
        ModuleKoszulZeroExtensionRequest(complex=_source(((q(2),),)))
    )
    assert result.target_complex.sequence == ((q(2),), (q(0),))
    assert result.unshifted_inclusions[1].entries == ((0, 0, q(1)),)
    assert result.shifted_inclusions[0].entries == ((1, 0, q(1)),)
    assert result.shifted_inclusions[1].entries == ((0, 0, q(-1)),)
    assert result.shifted_projections[2].entries == ((0, 0, q(-1)),)
    assert result.target_complex.differentials[1].entries == ((1, 0, q(2)),)
    _check_splitting(result)


def test_empty_sequence_and_public_example():
    empty = module_koszul_append_zero(
        ModuleKoszulZeroExtensionRequest(complex=_source(()))
    )
    assert empty.target_complex.sequence == ((q(0),),)
    _check_splitting(empty)
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.koszul.append_zero.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    _check_splitting(tool.run(request))


def test_length_three_uses_lexicographic_wedge_lookup():
    result = module_koszul_append_zero(
        ModuleKoszulZeroExtensionRequest(complex=_source(((q(2),), (q(3),), (q(4),))))
    )
    assert result.unshifted_inclusions[2].entries == (
        (0, 0, q(1)),
        (1, 1, q(1)),
        (3, 2, q(1)),
    )
    assert result.shifted_inclusions[2].entries == (
        (1, 0, q(1)),
        (2, 1, q(1)),
        (3, 2, q(1)),
    )
    _check_splitting(result)


def test_zero_extension_admission_precedes_reconstruction(monkeypatch):
    source = _source(((q(1),),))

    def must_not_build(*_args, **_kwargs):
        raise AssertionError("complex reconstruction ran before output admission")

    monkeypatch.setattr(operations, "_build_module_koszul_complex", must_not_build)
    monkeypatch.setattr(operations, "MAX_KOSZUL_ZERO_EXTENSION_OUTPUT_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        module_koszul_append_zero(ModuleKoszulZeroExtensionRequest(complex=source))
    assert caught.value.errors()[0]["type"] == "koszul.module.zero_extension_budget"
