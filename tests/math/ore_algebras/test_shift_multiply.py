"""Tests for exact shift Ore-operator multiplication."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    ShiftOperatorMultiplyRequest,
    ShiftOreOperator,
)
from jacobian.math.ore_algebras.operations import shift_operator_multiply
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_canonical_rational_function,
)


def _rf(
    numerator: tuple[tuple[int, int], ...],
    denominator: tuple[tuple[int, int], ...] = ((1, 0),),
) -> RationalFunction:
    def _part(terms: tuple[tuple[int, int], ...]) -> dict[str, object]:
        return {
            "terms": [
                {"coefficient": {"num": coefficient, "den": 1}, "exponents": [exponent]}
                for coefficient, exponent in terms
            ]
        }

    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["n"],
            "numerator": _part(numerator),
            "denominator": _part(denominator),
        }
    )


def _op(terms: tuple[tuple[int, RationalFunction], ...]) -> ShiftOreOperator:
    return ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": exponent, "coefficient": coefficient}
                for exponent, coefficient in terms
            ],
        }
    )


def _degree_rf(degree: int, constant: int = 1) -> RationalFunction:
    return _rf(((1, 0),), ((1, degree), (constant, 0)))


def _product_num(
    operator: ShiftOreOperator,
) -> tuple[tuple[int, tuple[tuple[int, Fraction], ...]], ...]:
    rows = []
    for term in operator.terms:
        num = tuple(
            (t.exponents[0], t.coefficient.as_fraction())
            for t in term.coefficient.numerator.terms
        )
        rows.append((term.exponent, num))
    return tuple(rows)


ONE = _rf(((1, 0),))
N = _rf(((1, 1),))
N_PLUS_1 = _rf(((1, 1), (1, 0)))

RfPair = tuple[dict[int, Fraction], dict[int, Fraction]]


def _shift() -> ShiftOreOperator:
    return _op(((1, ONE),))


class TestShiftMultiplyKnownAnswers:
    def test_shift_times_coordinate(self) -> None:
        result = shift_operator_multiply(_shift(), _op(((0, N),)))
        assert _product_num(result.product) == (
            (1, ((1, Fraction(1)), (0, Fraction(1)))),
        )
        assert len(result.ledger) == 1
        row = result.ledger[0]
        assert (row.left_exponent, row.right_exponent, row.result_exponent) == (1, 0, 1)

    def test_shift_squared_times_coordinate(self) -> None:
        square = shift_operator_multiply(_shift(), _shift()).product
        result = shift_operator_multiply(square, _op(((0, N),)))
        assert _product_num(result.product) == (
            (2, ((1, Fraction(1)), (0, Fraction(2)))),
        )

    def test_coordinate_times_shift(self) -> None:
        result = shift_operator_multiply(_op(((0, N),)), _shift())
        assert _product_num(result.product) == ((1, ((1, Fraction(1)),)),)

    def test_identity_and_zero_boundaries(self) -> None:
        identity = _op(((0, ONE),))
        zero = _op(())
        target = _op(((0, ONE), (2, N)))
        assert shift_operator_multiply(identity, target).product == target
        assert shift_operator_multiply(target, identity).product == target
        assert shift_operator_multiply(zero, target).product == zero
        assert shift_operator_multiply(target, zero).product == zero
        assert shift_operator_multiply(zero, zero).product == zero

    def test_scalar_left_multiple(self) -> None:
        two = _rf(((2, 0),))
        result = shift_operator_multiply(_op(((0, two),)), _shift())
        assert _product_num(result.product) == ((1, ((0, Fraction(2)),)),)


class TestShiftMultiplyInvariants:
    def _fixture_operators(
        self,
    ) -> tuple[ShiftOreOperator, ShiftOreOperator, ShiftOreOperator]:
        n_minus_1 = _rf(((1, 1), (-1, 0)))
        left = _op(((0, N), (1, ONE)))
        middle = _op(((0, n_minus_1), (2, ONE)))
        right = _op(((1, N),))
        return left, middle, right

    def test_associativity(self) -> None:
        left, middle, right = self._fixture_operators()
        first = shift_operator_multiply(
            shift_operator_multiply(left, middle).product, right
        ).product
        second = shift_operator_multiply(
            left, shift_operator_multiply(middle, right).product
        ).product
        assert _product_num(first) == _product_num(second)

    def test_distributivity_over_addition(self) -> None:
        from jacobian.math.ore_algebras.operations import (
            _decode_rf,
            _encode_rf,
            _rf_add,
        )

        left, middle, _ = self._fixture_operators()
        summed_terms: dict[int, RfPair] = {}
        for term in (*left.terms, *middle.terms):
            decoded = _decode_rf(term.coefficient)
            if term.exponent in summed_terms:
                summed_terms[term.exponent] = _rf_add(
                    summed_terms[term.exponent], decoded
                )
            else:
                summed_terms[term.exponent] = decoded
        total = _op(
            tuple(
                (exponent, _encode_rf(value))
                for exponent, value in sorted(summed_terms.items())
            )
        )
        right = _op(((0, N),))
        direct = _product_num(shift_operator_multiply(total, right).product)
        first = shift_operator_multiply(left, right).product
        second = shift_operator_multiply(middle, right).product
        replay: dict[int, RfPair] = {}
        for term in (*first.terms, *second.terms):
            decoded = _decode_rf(term.coefficient)
            if term.exponent in replay:
                replay[term.exponent] = _rf_add(replay[term.exponent], decoded)
            else:
                replay[term.exponent] = decoded
        combined = _op(
            tuple(
                (exponent, _encode_rf(value))
                for exponent, value in sorted(replay.items())
            )
        )
        assert direct == _product_num(combined)

    def test_ledger_replays_product(self) -> None:
        from jacobian.math.ore_algebras.operations import _decode_rf, _rf_add

        left, middle, _ = self._fixture_operators()
        result = shift_operator_multiply(left, middle)
        replay: dict[int, RfPair] = {}
        for row in result.ledger:
            decoded = _decode_rf(row.contribution)
            assert row.result_exponent == row.left_exponent + row.right_exponent
            if row.result_exponent in replay:
                replay[row.result_exponent] = _rf_add(
                    replay[row.result_exponent], decoded
                )
            else:
                replay[row.result_exponent] = decoded
        for term in result.product.terms:
            assert _decode_rf(term.coefficient) == replay[term.exponent]


class TestShiftMultiplyAdmission:
    @pytest.mark.parametrize(("right_degree", "expected_degree"), ((1, 65), (64, 128)))
    def test_degree_64_inputs_produce_wider_canonical_values_and_consumers(
        self, right_degree: int, expected_degree: int
    ) -> None:
        left = _op(((0, _degree_rf(64, 1)),))
        right = _op(((0, _degree_rf(right_degree, 2)),))

        result = shift_operator_multiply(left, right)
        coefficient = result.product.terms[0].coefficient
        assert coefficient.denominator.terms[0].exponents == (expected_degree,)

        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored == result
        consumed = RationalFunctionMap(
            source_variables=("n",),
            target_coordinates=("y",),
            components=(coefficient,),
        )
        assert (
            RationalFunctionMap.model_validate_json(consumed.model_dump_json())
            == consumed
        )

    def test_carrier_widening_does_not_widen_ore_input_admission(self) -> None:
        wide = _rf(((1, 128),))
        assert RationalFunction.model_validate_json(wide.model_dump_json()) == wide
        with pytest.raises(ValueError, match="64-degree operation budget"):
            require_canonical_rational_function(wide)
        with pytest.raises(OperationDomainValidationError):
            shift_operator_multiply(_op(((0, wide),)), _op(((0, ONE),)))

    def test_accumulated_degree_over_128_is_rejected_before_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jacobian.math.ore_algebras import operations

        left_coefficient = _degree_rf(64, 1)
        right_coefficient = _degree_rf(64, 2)
        left = _op(((0, left_coefficient), (1, left_coefficient)))
        right = _op(((0, right_coefficient), (1, right_coefficient)))

        def fail_ledger_construction(*args: object, **kwargs: object) -> object:
            raise AssertionError("ledger construction must follow degree admission")

        monkeypatch.setattr(
            operations.ShiftMultiplyLedgerRow,
            "model_construct",
            fail_ledger_construction,
        )
        with pytest.raises(OperationResourceAdmissionError, match="degree"):
            shift_operator_multiply(left, right)

    def test_shifted_coefficient_height_is_rejected_before_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jacobian.math.ore_algebras import operations

        high_digit_coefficient = _rf(((10**63, 64), (1, 0)))
        left = _op(((16, high_digit_coefficient),))
        right = _op(((0, high_digit_coefficient),))

        def fail_ledger_construction(*args: object, **kwargs: object) -> object:
            raise AssertionError("ledger construction must follow height admission")

        monkeypatch.setattr(
            operations.ShiftMultiplyLedgerRow,
            "model_construct",
            fail_ledger_construction,
        )
        with pytest.raises(OperationResourceAdmissionError, match="coefficient-digit"):
            shift_operator_multiply(left, right)

    def test_left_coefficient_height_is_rejected_before_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jacobian.math.ore_algebras import operations

        left = _op(((20, _rf(((10**63, 0),))),))
        right = _op(((0, _rf(((10**63, 15),))),))

        def fail_ledger_construction(*args: object, **kwargs: object) -> object:
            raise AssertionError("ledger construction must follow height admission")

        monkeypatch.setattr(
            operations.ShiftMultiplyLedgerRow,
            "model_construct",
            fail_ledger_construction,
        )
        with pytest.raises(OperationResourceAdmissionError, match="coefficient-digit"):
            shift_operator_multiply(left, right)

    def test_exact_littlewood_self_product_at_128_digits_is_admitted(self) -> None:
        signs = "--++--+-+-++-++--+--+-++++++-+-+++-----++---+-+-++---------+-+--"
        assert len(signs) == 64
        coefficient = _rf(
            tuple(
                ((1 if sign == "+" else -1) * 10**63, exponent)
                for exponent, sign in zip(range(63, -1, -1), signs, strict=True)
            )
        )
        operator = _op(((0, coefficient),))

        result = shift_operator_multiply(operator, operator)

        heights = [
            max(
                len(str(abs(term.coefficient.num))),
                len(str(term.coefficient.den)),
            )
            for term in result.product.terms[0].coefficient.numerator.terms
        ]
        assert max(heights) == 128

    def test_negative_exponent_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShiftOreOperator.model_validate(
                {"variable": "n", "terms": [{"exponent": -1, "coefficient": ONE}]}
            )

    def test_unsorted_terms_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _op(((1, ONE), (0, ONE)))

    def test_zero_coefficient_term_rejected(self) -> None:
        zero_rf = RationalFunction.model_validate(
            {
                "domain": "QQ",
                "variables": ["n"],
                "numerator": {"terms": []},
                "denominator": {
                    "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
                },
            }
        )
        with pytest.raises(ValidationError):
            _op(((0, zero_rf),))

    def test_noncanonical_coefficient_rejected(self) -> None:
        bad = _rf(((1, 1),), ((1, 1),))
        with pytest.raises(OperationDomainValidationError):
            shift_operator_multiply(_op(((0, bad),)), _op(((0, ONE),)))

    def test_order_envelope_refused_as_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jacobian.math.ore_algebras import operations

        left = _op(((16, ONE),))
        right = _op(((16, ONE),))
        monkeypatch.setattr(operations, "MAX_SHIFT_RESULT_ORDER", 1)
        with pytest.raises(OperationResourceAdmissionError):
            shift_operator_multiply(left, right)

    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.ore_algebras._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "ore.shift.operator.multiply.compute"
        )
        request = ShiftOperatorMultiplyRequest(left=_shift(), right=_op(((0, N),)))
        assert tool.run(request) == shift_operator_multiply(request.left, request.right)

    def test_published_example_validates(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.ore_algebras._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "ore.shift.operator.multiply.compute"
        )
        example = tool.examples[0]
        request = tool.request_type.model_validate_json(
            encode_strict_json(example.input), strict=True
        )
        assert _product_num(tool.run(request).product) == (
            (1, ((1, Fraction(1)), (0, Fraction(1)))),
        )
