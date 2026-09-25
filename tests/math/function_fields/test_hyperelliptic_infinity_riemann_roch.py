from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldPlace,
    HyperellipticInfinityPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_genus,
    function_field_hyperelliptic_infinity_valuation,
    function_field_riemann_roch_space,
)


def _rf(
    characteristic: int, numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=numerator
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=denominator
        ),
    )


def _hyperelliptic_field(
    characteristic: int, branch_coefficients: tuple[int, ...]
) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rf(
                characteristic,
                tuple(
                    (-coefficient) % characteristic
                    for coefficient in branch_coefficients
                ),
            ),
            _rf(characteristic, (0,)),
            _rf(characteristic, (1,)),
        ),
    )


def _infinity(field: FiniteFunctionField) -> FunctionFieldPlace:
    return FunctionFieldPlace(field=field, kind="INFINITE", degree=1)


def _divisor(
    field: FiniteFunctionField, multiplicity: int | None
) -> FunctionFieldDivisor:
    if multiplicity is None or multiplicity == 0:
        return FunctionFieldDivisor(field=field, terms=())
    return FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(place=_infinity(field), multiplicity=multiplicity),
        ),
    )


def _valuation_place(field: FiniteFunctionField) -> HyperellipticInfinityPlace:
    return HyperellipticInfinityPlace(
        field=field,
        residue_field=FiniteFieldPresentation(
            characteristic=field.characteristic,
            modulus_coefficients=(0, 1),
            generator="z",
        ),
    )


def test_genus_one_basis_has_expected_valuations_and_rr_dimensions() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))  # y^2=x^3-x
    place = _valuation_place(field)
    assert function_field_genus(field).genus == 1
    expected_dimensions = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}

    for multiplicity, expected_dimension in expected_dimensions.items():
        result = function_field_riemann_roch_space(_divisor(field, multiplicity))
        assert result.dimension == expected_dimension
        assert len(result.basis) == expected_dimension
        assert result.divisor == _divisor(field, multiplicity)
        for basis_element in result.basis:
            valuation_result = function_field_hyperelliptic_infinity_valuation(
                place, basis_element
            )
            assert valuation_result.valuation.kind == "FINITE"
            assert valuation_result.valuation.value + multiplicity >= 0
        assert type(result).model_validate_json(result.model_dump_json()) == result


def test_genus_two_special_and_large_degree_spaces_match_closed_form() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 0, 0, 1))  # y^2=x^5-x
    place = _valuation_place(field)
    assert function_field_genus(field).genus == 2
    expected_dimensions = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5}

    for multiplicity, expected_dimension in expected_dimensions.items():
        result = function_field_riemann_roch_space(_divisor(field, multiplicity))
        assert result.dimension == expected_dimension
        assert len(result.basis) == expected_dimension
        # For m >= 2g-1, the independent Riemann-Roch theorem gives l(mP)=m+1-g.
        if multiplicity >= 3:
            assert (
                result.dimension == multiplicity + 1 - function_field_genus(field).genus
            )
        valuations = []
        for element in result.basis:
            value = function_field_hyperelliptic_infinity_valuation(
                place, element
            ).valuation
            assert value.kind == "FINITE"
            valuations.append(value.value)
        if multiplicity == 5:
            assert valuations == [0, -2, -4, -5]
            assert [
                (
                    element.coordinates[0].numerator.coefficients,
                    element.coordinates[1].numerator.coefficients,
                )
                for element in result.basis
            ] == [((1,), (0,)), ((0, 1), (0,)), ((0, 0, 1), (0,)), ((0,), (1,))]
        else:
            assert all(value >= -multiplicity for value in valuations)


def test_zero_and_negative_infinity_divisors_match_constant_or_empty_space() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))

    constant_space = function_field_riemann_roch_space(_divisor(field, None))
    assert constant_space.dimension == 1
    assert constant_space.basis[0].coordinates[0].numerator.is_one()
    assert constant_space.basis[0].coordinates[1].numerator.is_zero()

    negative_space = function_field_riemann_roch_space(_divisor(field, -1))
    assert negative_space.dimension == 0
    assert negative_space.basis == ()


def test_basis_growth_is_rejected_before_construction() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))

    boundary = function_field_riemann_roch_space(_divisor(field, 13))
    assert boundary.dimension == 13

    genus_two = _hyperelliptic_field(5, (0, 4, 0, 0, 0, 1))
    assert function_field_riemann_roch_space(_divisor(genus_two, 14)).dimension == 13
    with pytest.raises(OperationResourceAdmissionError, match="basis exceeds"):
        function_field_riemann_roch_space(_divisor(genus_two, 15))
    with pytest.raises(OperationResourceAdmissionError, match="basis exceeds"):
        function_field_riemann_roch_space(_divisor(field, 100_000))


def test_request_keeps_signed_divisor_coefficient_as_canonical_decimal() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    term = FunctionFieldDivisorTerm(place=_infinity(field), multiplicity=-3)

    assert term.multiplicity == -3
    assert '"multiplicity":"-3"' in term.model_dump_json()
    assert FunctionFieldDivisorTerm.model_validate_json(term.model_dump_json()) == term
    with pytest.raises(ValidationError):
        FunctionFieldDivisorTerm(place=_infinity(field), multiplicity=True)


def test_multiplicity_bound_is_checked_before_place_admission() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    forged = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=None, multiplicity=1 << 4096
            ),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_riemann_roch_space(forged)

    assert error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_multiplicity_exceeds_envelope"
    )


def test_parent_mismatch_is_rejected_by_the_private_path() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    other = _hyperelliptic_field(5, (0, 4, 0, 0, 0, 1))
    mismatched = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=_infinity(other), multiplicity=1
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_riemann_roch_space(mismatched)

    assert error.value.errors()[0]["type"] == "function_field.divisor_parent"


def test_even_degree_and_finite_support_are_rejected_by_the_private_path() -> None:
    even_field = _hyperelliptic_field(5, (1, 0, 0, 0, 1))  # y^2=x^4+1
    with pytest.raises(OperationDomainValidationError) as odd_model:
        function_field_riemann_roch_space(_divisor(even_field, 1))
    assert odd_model.value.errors()[0]["type"] == (
        "function_field.hyperelliptic_riemann_roch_requires_odd_model"
    )

    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    finite = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    finite_divisor = FunctionFieldDivisor(
        field=field,
        terms=(FunctionFieldDivisorTerm(place=finite, multiplicity=1),),
    )
    with pytest.raises(OperationDomainValidationError) as finite_support:
        function_field_riemann_roch_space(finite_divisor)
    assert finite_support.value.errors()[0]["type"] == (
        "function_field.hyperelliptic_riemann_roch_requires_infinity_support"
    )


def test_unsupported_extension_model_is_rejected() -> None:
    extension = FiniteFunctionField(
        characteristic=2,
        defining_polynomial=(
            _rf(2, (0, 1)),
            _rf(2, (1,)),
            _rf(2, (1,)),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_riemann_roch_space(
            FunctionFieldDivisor(field=extension, terms=())
        )

    assert error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_requires_supported_model"
    )


def test_hyperelliptic_slice_is_part_of_the_canonical_operation() -> None:
    operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert "function_field.riemann_roch_space.compute" in operation_ids
    assert (
        "function_field.hyperelliptic_infinity_riemann_roch_space.compute"
        not in operation_ids
    )

    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "function_field.riemann_roch_space.compute"
    )
    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    request = tool.request_type.model_validate_json(
        json.dumps(
            {
                "divisor": {
                    "field": field.model_dump(),
                    "terms": [
                        {
                            "place": {
                                "field": field.model_dump(),
                                "kind": "INFINITE",
                                "degree": 1,
                            },
                            "multiplicity": "3",
                        }
                    ],
                }
            }
        )
    )
    result = tool.run(request)
    assert result.dimension == 3
    assert any("hyperelliptic infinity place" in term for term in tool.discovery_terms)
