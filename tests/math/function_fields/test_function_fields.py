"""Tests for function_field.element.multiply.compute."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._gfpx import (
    _primitive_polynomial_terms,
    is_irreducible_over_rational_function,
    rf_normalize,
)
from jacobian.math.function_fields._models import (
    MAX_EXTENSION_DEGREE,
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldElementMultiplyRequest,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields._tools import TOOLS
from jacobian.math.function_fields.operations import (
    _element_inverse,
    _to_internal_rational_function,
    function_field_element_multiply,
)

OPERATION_ID = "function_field.element.multiply.compute"


def _poly(characteristic: int, coefficients: tuple[int, ...]) -> PrimeFieldPolynomial:
    return PrimeFieldPolynomial(
        characteristic=characteristic, coefficients=coefficients
    )


def _rf(
    characteristic: int, numerator: tuple[int, ...], denominator: tuple[int, ...]
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=_poly(characteristic, numerator),
        denominator=_poly(characteristic, denominator),
    )


def _field(
    characteristic: int, defining: tuple[PrimeFieldRationalFunction, ...]
) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        defining_polynomial=defining,
    )


def _element(
    field: FiniteFunctionField, coordinates: tuple[PrimeFieldRationalFunction, ...]
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=coordinates)


def _zero(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (0,), (1,))


def _one(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (1,), (1,))


def _x(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (0, 1), (1,))


# GF(2)(x)[y]/(y^2 + y + x): the standard elliptic-like quadratic model.
GF2_FIELD = _field(
    2,
    (
        _x(2),
        _one(2),
        _one(2),
    ),
)
GF2_Y = _element(GF2_FIELD, (_zero(2), _one(2)))
GF2_ONE = _element(GF2_FIELD, (_one(2), _zero(2)))

# GF(3)(x)[y]/(y^2 - x).
GF3_FIELD = _field(
    3,
    (
        _rf(3, (0, 2), (1,)),  # -x
        _zero(3),
        _one(3),
    ),
)
GF3_Y = _element(GF3_FIELD, (_zero(3), _one(3)))

# Every GF(3)-point specializes this polynomial to y^2, while the rational
# function x^3-x has odd valuations and is therefore not a square in GF(3)(x).
GF3_FALLBACK_FIELD = _field(
    3,
    (
        _rf(3, (0, 1, 0, 2), (1, 0, 1)),
        _zero(3),
        _one(3),
    ),
)
GF3_FALLBACK_Y = _element(GF3_FALLBACK_FIELD, (_zero(3), _one(3)))


def _multiply(
    left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement
) -> FiniteFunctionFieldElement:
    return function_field_element_multiply(left, right).product


def _equal(left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement) -> bool:
    return left.model_dump_json() == right.model_dump_json()


class TestKnownAnswers:
    def test_gf2_generator_squared_reduces_to_y_plus_x(self) -> None:
        product = _multiply(GF2_Y, GF2_Y)
        assert product.coordinates[0] == _x(2)
        assert product.coordinates[1] == _one(2)

    def test_gf3_generator_squared_reduces_to_x(self) -> None:
        product = _multiply(GF3_Y, GF3_Y)
        assert product.coordinates[0] == _x(3)
        assert product.coordinates[1] == _zero(3)

    def test_gf2_mixed_product(self) -> None:
        y_plus_x = _element(GF2_FIELD, (_x(2), _one(2)))
        y_plus_one = _element(GF2_FIELD, (_one(2), _one(2)))
        # (y + x)(y + 1) = x*y after reducing y^2 = y + x.
        product = _multiply(y_plus_x, y_plus_one)
        assert product.coordinates[0] == _zero(2)
        assert product.coordinates[1] == _x(2)

    def test_gf3_subfield_product_is_ordinary_rational_function_product(self) -> None:
        x_squared = _element(GF3_FIELD, (_rf(3, (0, 0, 1), (1,)), _zero(3)))
        two = _element(GF3_FIELD, (_rf(3, (2,), (1,)), _zero(3)))
        product = _multiply(x_squared, two)
        assert product.coordinates[0] == _rf(3, (0, 0, 2), (1,))
        assert product.coordinates[1] == _zero(3)

    def test_degree_one_extension_product(self) -> None:
        field = _field(3, (_x(3), _one(3)))  # y + x
        # In the degree-one quotient y = -x, so the element y is the constant
        # coordinate -x and y^2 is x^2.
        y = _element(field, (_rf(3, (0, 2), (1,)),))
        product = _multiply(y, y)
        assert product.coordinates[0] == _rf(3, (0, 0, 1), (1,))


class TestDefiningInvariants:
    def test_identity(self) -> None:
        assert _equal(_multiply(GF2_Y, GF2_ONE), GF2_Y)
        assert _equal(_multiply(GF2_ONE, GF2_Y), GF2_Y)

    def test_commutativity(self) -> None:
        a = _element(GF2_FIELD, (_x(2), _one(2)))
        b = _element(GF2_FIELD, (_one(2), _x(2)))
        assert _equal(_multiply(a, b), _multiply(b, a))

    def test_associativity(self) -> None:
        a = _element(GF2_FIELD, (_x(2), _one(2)))
        b = _element(GF2_FIELD, (_one(2), _x(2)))
        c = _element(GF2_FIELD, (_one(2), _one(2)))
        assert _equal(_multiply(_multiply(a, b), c), _multiply(a, _multiply(b, c)))

    def test_left_and_right_inverse(self) -> None:
        a = _element(GF3_FIELD, (_x(3), _one(3)))
        inverse = _element_inverse(a)
        assert _equal(_multiply(a, inverse), _element(GF3_FIELD, (_one(3), _zero(3))))
        assert _equal(_multiply(inverse, a), _element(GF3_FIELD, (_one(3), _zero(3))))

    def test_zero_is_absorbing(self) -> None:
        zero = _element(GF2_FIELD, (_zero(2), _zero(2)))
        product = _multiply(GF2_Y, zero)
        assert _equal(product, _element(GF2_FIELD, (_zero(2), _zero(2))))

    def test_valuation_identity_on_the_subfield(self) -> None:
        # v_0 is the order of vanishing at the rational place x = 0.
        x_squared = _element(GF3_FIELD, (_rf(3, (0, 0, 1), (1,)), _zero(3)))
        x_inverse = _element(GF3_FIELD, (_rf(3, (1,), (0, 1)), _zero(3)))
        product = _multiply(x_squared, x_inverse)
        left = _valuation(x_squared.coordinates[0], 3)
        right = _valuation(x_inverse.coordinates[0], 3)
        result = _valuation(product.coordinates[0], 3)
        assert left == 2
        assert right == -1
        assert result == left + right

    def test_irreducible_fallback_admits_a_polynomial_without_irreducible_specialization(
        self,
    ) -> None:
        product = _multiply(GF3_FALLBACK_Y, GF3_FALLBACK_Y)
        assert product.coordinates[0] == _rf(3, (0, 2, 0, 1), (1, 0, 1))
        assert product.coordinates[1] == _zero(3)


def _polynomial_order(coefficients: tuple[int, ...], characteristic: int) -> int | None:
    for index, value in enumerate(coefficients):
        if value % characteristic:
            return index
    return None


def _valuation(value: PrimeFieldRationalFunction, characteristic: int) -> int:
    numerator = _polynomial_order(value.numerator.coefficients, characteristic)
    denominator = _polynomial_order(value.denominator.coefficients, characteristic)
    assert numerator is not None and denominator is not None
    return numerator - denominator


class TestIndependentBackendCrossCheck:
    @staticmethod
    def _reduce_with_sympy(
        prime: int,
        defining_numerators: tuple[tuple[int, ...], ...],
        left_numerators: tuple[tuple[int, ...], ...],
        right_numerators: tuple[tuple[int, ...], ...],
    ) -> list:
        from sympy import GF, Poly, symbols
        from sympy.polys.fields import FractionField

        x_symbol, y_symbol = symbols("x y")
        fraction_field = FractionField(GF(prime), [x_symbol])

        def as_poly(numerators: tuple[tuple[int, ...], ...]):
            return Poly(
                sum(
                    _poly_expression(coefficient, x_symbol) * y_symbol**power
                    for power, coefficient in enumerate(numerators)
                ),
                y_symbol,
                domain=fraction_field,
            )

        defining = as_poly(defining_numerators)
        product = Poly(
            (as_poly(left_numerators) * as_poly(right_numerators)).as_expr(),
            y_symbol,
            domain=fraction_field,
        )
        return product.rem(defining).all_coeffs()

    @staticmethod
    def _coordinates_descending(
        element: FiniteFunctionFieldElement,
    ) -> list:
        from sympy import GF, symbols
        from sympy.polys.fields import FractionField

        x_symbol = symbols("x")
        fraction_field = FractionField(GF(element.field.characteristic), [x_symbol])

        def convert(coordinate: PrimeFieldRationalFunction):
            numerator = fraction_field(
                _poly_expression(coordinate.numerator.coefficients, x_symbol)
            )
            denominator = fraction_field(
                _poly_expression(coordinate.denominator.coefficients, x_symbol)
            )
            return numerator / denominator

        descending = [
            convert(coordinate) for coordinate in reversed(element.coordinates)
        ]
        while descending and str(descending[0]) == "0":
            descending.pop(0)
        return [str(value) for value in descending]

    def test_gf2_generator_squared_matches_sympy(self) -> None:
        result = function_field_element_multiply(GF2_Y, GF2_Y)
        expected = self._reduce_with_sympy(
            2, ((0, 1), (1,), (1,)), ((), (1,)), ((), (1,))
        )
        assert [str(value) for value in expected] == self._coordinates_descending(
            result.product
        )

    def test_gf3_generator_squared_matches_sympy(self) -> None:
        result = function_field_element_multiply(GF3_Y, GF3_Y)
        expected = self._reduce_with_sympy(
            3, ((0, 2), (), (1,)), ((), (1,)), ((), (1,))
        )
        assert [str(value) for value in expected] == self._coordinates_descending(
            result.product
        )

    def test_gf2_mixed_product_matches_sympy(self) -> None:
        left = _element(GF2_FIELD, (_x(2), _one(2)))
        right = _element(GF2_FIELD, (_one(2), _one(2)))
        result = function_field_element_multiply(left, right)
        expected = self._reduce_with_sympy(
            2, ((0, 1), (1,), (1,)), ((0, 1), (1,)), ((1,), (1,))
        )
        assert [str(value) for value in expected] == self._coordinates_descending(
            result.product
        )

    def test_flint_primitive_factorization_reconstructs_the_source(self) -> None:
        from flint import nmod_mpoly_ctx

        kpoly = tuple(
            _to_internal_rational_function(coefficient)
            for coefficient in GF3_FALLBACK_FIELD.defining_polynomial
        )
        terms = _primitive_polynomial_terms(kpoly, 3)
        context = nmod_mpoly_ctx.get(("x", "y"), 3, "lex")
        source = context.from_dict(terms)
        unit, factors = source.factor()
        reconstructed = context.constant(unit)
        for factor, exponent in factors:
            reconstructed *= factor ** int(exponent)
        assert reconstructed == source


def _poly_expression(coefficients: tuple[int, ...], x_symbol) -> object:
    if coefficients == ():
        return 0
    return sum(
        coefficient * x_symbol**power for power, coefficient in enumerate(coefficients)
    )


class TestBoundariesAndAdversarial:
    def test_reducible_defining_polynomial_is_rejected(self) -> None:
        reducible = _field(
            3,
            (_rf(3, (2,), (1,)), _zero(3), _one(3)),  # y^2 - 1
        )
        element = _element(reducible, (_zero(3), _one(3)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _multiply(element, element)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.extension_not_admitted_irreducible"
        )

    def test_rational_function_irreducibility_fallback_rejects_reducible_input(
        self,
    ) -> None:
        reducible = _field(
            3,
            (_rf(3, (2,), (1,)), _zero(3), _one(3)),  # y^2 - 1
        )
        kpoly = tuple(
            _to_internal_rational_function(coefficient)
            for coefficient in reducible.defining_polynomial
        )
        assert not is_irreducible_over_rational_function(kpoly, 3)

    def test_product_of_y_minus_x_and_y_plus_x_is_rejected(self) -> None:
        reducible = _field(
            3,
            (_rf(3, (0, 0, 2), (1,)), _zero(3), _one(3)),
        )
        element = _element(reducible, (_zero(3), _one(3)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _multiply(element, element)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.extension_not_admitted_irreducible"
        )

    def test_primitive_lift_removes_x_content_without_changing_y_degree(self) -> None:
        x = rf_normalize((0, 1), (1,), 3)
        kpoly = (x, rf_normalize((), (1,), 3), x)

        terms = _primitive_polynomial_terms(kpoly, 3)

        assert terms == {(0, 0): 1, (0, 2): 1}

    def test_exact_backend_unavailability_is_not_a_reducibility_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        original_import = builtins.__import__

        def deny_flint(name, *args, **kwargs):
            if name == "flint":
                raise ImportError("test backend unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", deny_flint)
        with pytest.raises(OperationBackendError) as exc_info:
            _multiply(GF3_FALLBACK_Y, GF3_FALLBACK_Y)
        assert exc_info.value.reason == BackendFailureReason.INITIALIZATION

    def test_invalid_backend_factorization_is_not_a_reducibility_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        from types import SimpleNamespace

        class FakeSource:
            def factor(self):
                return 1, []

            def __eq__(self, _other):
                return False

        class FakeContext:
            def from_dict(self, _terms):
                return FakeSource()

            def constant(self, value):
                return value

        fake_flint = SimpleNamespace(
            nmod_mpoly_ctx=SimpleNamespace(get=lambda *_args, **_kwargs: FakeContext())
        )
        monkeypatch.setitem(sys.modules, "flint", fake_flint)

        kpoly = tuple(
            _to_internal_rational_function(coefficient)
            for coefficient in GF3_FALLBACK_FIELD.defining_polynomial
        )
        with pytest.raises(OperationBackendError) as exc_info:
            is_irreducible_over_rational_function(kpoly, 3)
        assert exc_info.value.reason == BackendFailureReason.INVALID_OUTPUT

    def test_inseparable_defining_polynomial_is_rejected(self) -> None:
        inseparable = _field(
            2,
            (_x(2), _zero(2), _one(2)),  # y^2 + x, derivative zero in char 2
        )
        element = _element(inseparable, (_zero(2), _one(2)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _multiply(element, element)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.extension_not_separable"
        )

    def test_elements_from_different_fields_are_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _multiply(GF2_Y, GF3_Y)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.element_field_mismatch"
        )

    def test_non_monic_defining_polynomial_is_structurally_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _field(3, (_zero(3), _rf(3, (0, 2), (1,))))

    def test_zero_denominator_is_structurally_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _rf(2, (1,), (0,))

    def test_coefficient_growth_above_envelope_is_a_resource_rejection(self) -> None:
        high_degree = _rf(
            2, tuple(1 if index == 12 else 0 for index in range(13)), (1,)
        )
        element = _element(GF2_FIELD, (high_degree, _zero(2)))
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            _multiply(element, element)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.coefficient_growth_exceeds_envelope"
        )

    def test_extension_degree_above_envelope_is_a_resource_rejection(self) -> None:
        degree = MAX_EXTENSION_DEGREE + 1
        coefficients = (*tuple(_zero(2) for _ in range(degree)), _one(2))
        field = FiniteFunctionField.model_construct(
            characteristic=2,
            variable="x",
            generator="y",
            defining_polynomial=coefficients,
        )
        element = FiniteFunctionFieldElement.model_construct(
            field=field,
            coordinates=tuple(_zero(2) for _ in range(degree)),
        )
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            function_field_element_multiply(element, element)
        assert (
            exc_info.value.errors()[0]["type"]
            == "function_field.extension_degree_exceeds_envelope"
        )


class TestParityAndSerialization:
    def test_native_matches_catalog_tool(self) -> None:
        tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
        request = FunctionFieldElementMultiplyRequest(left=GF2_Y, right=GF2_Y)
        assert (
            tool.run(request).model_dump()
            == function_field_element_multiply(request.left, request.right).model_dump()
        )

    def test_request_round_trip_through_json(self) -> None:
        request = FunctionFieldElementMultiplyRequest(left=GF2_Y, right=GF2_Y)
        encoded = request.model_dump_json()
        assert (
            FunctionFieldElementMultiplyRequest.model_validate_json(encoded) == request
        )

    def test_result_round_trip_through_json(self) -> None:
        result = function_field_element_multiply(GF2_Y, GF2_Y)
        encoded = result.model_dump_json()
        parsed = type(result).model_validate_json(encoded)
        assert parsed.model_dump() == result.model_dump()

    def test_catalog_example_is_executable(self) -> None:
        tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
        request = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        result = tool.run(request)
        assert result.product.coordinates[0].numerator.coefficients == (0, 1)
        assert result.product.coordinates[1].numerator.coefficients == (1,)

    def test_catalog_discovery(self) -> None:
        ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in ids
        assert "number_theory.function_field.multiply.compute" not in ids
