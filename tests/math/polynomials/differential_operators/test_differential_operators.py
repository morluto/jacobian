"""Exact contract tests for constant-coefficient differential operators."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
from jacobian.math.polynomials.differential_operators._bounds import (
    MAX_APPLICATION_ITERATIONS,
    MAX_APPLICATION_RESULT_BYTES,
)
from jacobian.math.polynomials.differential_operators._models import (
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
)
from jacobian.math.polynomials.differential_operators._operations import (
    compute_differential_operator_application,
)
from jacobian.math.polynomials.differential_operators._tools import TOOLS
from jacobian.math.polynomials.differential_operators.operations import (
    apply_constant_coefficient_differential_operator,
)
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
    DifferentialOperatorTerm,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _operator(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> ConstantCoefficientDifferentialOperator:
    return ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=tuple(
            DifferentialOperatorTerm(
                coefficient=_rational(coefficient),
                orders=orders,
            )
            for orders, coefficient in sorted(terms.items(), reverse=True)
            if coefficient
        ),
    )


def _from_sympy(
    expression: sympy.Expr, symbols: tuple[sympy.Symbol, ...]
) -> RationalPolynomial:
    return rational_polynomial_from_sympy(
        sympy.Poly(expression, *symbols, domain=sympy.QQ),
        tuple(str(symbol) for symbol in symbols),
    )


def test_catalog_contains_the_single_atomic_application() -> None:
    assert [tool.operation_id for tool in TOOLS] == [
        "polynomial.differential_operator.apply.compute"
    ]


def test_known_second_iterate_and_exact_comparison() -> None:
    request = TOOLS[0].request_type.model_validate(TOOLS[0].examples[0].input)

    result = compute_differential_operator_application(request)

    assert result.output == request.expected
    assert result.is_zero is False
    assert result.matches_expected is True


def test_exact_rational_operator_coefficient() -> None:
    source = _polynomial(("x", "y"), {(3, 1): Fraction(1, 2)})
    operator = _operator(("x", "y"), {(2, 0): Fraction(2, 3)})

    assert apply_constant_coefficient_differential_operator(
        source, operator
    ) == _polynomial(("x", "y"), {(1, 1): 2})


def test_finite_iteration_equals_caller_composition() -> None:
    variables = ("x", "y")
    source = _polynomial(variables, {(4, 1): 1, (1, 3): -2, (0, 0): 5})
    operator = _operator(variables, {(1, 0): 1, (0, 1): 2})

    direct = apply_constant_coefficient_differential_operator(
        source, operator, iterations=3
    )
    composed = source
    for _ in range(3):
        composed = apply_constant_coefficient_differential_operator(composed, operator)

    assert direct == composed


def test_zero_and_identity_degeneracies_retain_the_axis() -> None:
    variables = ("x", "y")
    source = _polynomial(variables, {(2, 0): 1})
    zero_operator = _operator(variables, {})

    identity = apply_constant_coefficient_differential_operator(
        source, zero_operator, iterations=0
    )
    zero = apply_constant_coefficient_differential_operator(
        source, zero_operator, iterations=1
    )

    assert identity == source
    assert zero == _polynomial(variables, {})
    assert zero.variables == variables


def test_native_iteration_count_must_be_an_integer() -> None:
    source = _polynomial(("x",), {(1,): 1})
    operator = _operator(("x",), {(1,): 1})

    with pytest.raises(TypeError, match="must be an integer"):
        apply_constant_coefficient_differential_operator(
            source,
            operator,
            iterations=True,
        )


def test_positive_order_short_circuits_a_large_vanishing_iterate() -> None:
    variables = ("x", "y")
    source = _polynomial(variables, {(3, 2): 1})
    operator = _operator(variables, {(1, 0): 1, (0, 1): -1})

    output = apply_constant_coefficient_differential_operator(
        source,
        operator,
        iterations=MAX_APPLICATION_ITERATIONS,
    )

    assert output == _polynomial(variables, {})


def test_iteration_cap_admits_non_expanding_requests_beyond_the_limit() -> None:
    variables = ("x", "y")
    zero_operator = _operator(variables, {})
    tall_iterations = MAX_APPLICATION_ITERATIONS + 1

    request = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(variables, {(1, 1): 1}),
        operator=zero_operator,
        iterations=tall_iterations,
    )
    result = compute_differential_operator_application(request)

    assert result.is_zero is True
    assert result.output == _polynomial(variables, {})
    assert apply_constant_coefficient_differential_operator(
        _polynomial(variables, {}),
        _operator(variables, {(1, 0): 1}),
        iterations=tall_iterations,
    ) == _polynomial(variables, {})


def test_iteration_cap_gates_only_operator_power_expansion() -> None:
    variables = ("x",)
    scaling = _operator(variables, {(0,): 2})
    source = _polynomial(variables, {(3,): 1})
    tall_iterations = MAX_APPLICATION_ITERATIONS + 1

    scaled = DifferentialOperatorApplyRequest(
        polynomial=source,
        operator=scaling,
        iterations=tall_iterations,
    )
    assert compute_differential_operator_application(scaled).output == _polynomial(
        variables,
        {(3,): 2**tall_iterations},
    )

    expanding = _operator(variables, {(1,): 1, (0,): 1})
    with pytest.raises(ValidationError, match="operation limit"):
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=expanding,
            iterations=tall_iterations,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        apply_constant_coefficient_differential_operator(
            source,
            expanding,
            iterations=-1,
        )


def test_scalar_iterate_growth_is_bounded_by_the_coefficient_budget() -> None:
    variables = ("x",)
    scaling = _operator(variables, {(0,): 2})

    with pytest.raises(ValidationError, match="coefficient-digit budget"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(0,): 1}),
            operator=scaling,
            iterations=400_000,
        )


def test_identity_iterate_admits_sources_beyond_the_former_input_cap() -> None:
    variables = ("x",)
    wide_source = _polynomial(
        variables,
        dict.fromkeys(((index,) for index in range(600)), 1),
    )
    operator = _operator(variables, {(1,): 1})

    identity = apply_constant_coefficient_differential_operator(
        wide_source,
        operator,
        iterations=0,
    )
    assert identity == wide_source


def test_identity_result_retains_expected_and_replays_bound() -> None:
    variables = ("x",)
    source = _polynomial(variables, {(129,): 1})
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(1,): 1}),
            iterations=0,
            expected=source,
        )
    )

    assert result.output == source
    assert result.matches_expected is True
    assert result.is_zero is False
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )
    assert replayed == result


def test_identity_power_admits_sources_beyond_expansion_caps() -> None:
    variables = ("x",)
    tall_source = _polynomial(variables, {(129,): 1})
    unit = _operator(variables, {(0,): 1})

    identity = apply_constant_coefficient_differential_operator(
        tall_source,
        unit,
        iterations=1,
    )

    assert identity == tall_source


def test_identity_power_retains_expected_and_replays_bound() -> None:
    variables = ("x",)
    tall_source = _polynomial(variables, {(129,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=tall_source,
            operator=_operator(variables, {(0,): 1}),
            iterations=7,
            expected=tall_source,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == tall_source
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_nonidentity_scalar_operators_follow_scale_only_budgets() -> None:
    variables = ("x",)
    negation = _operator(variables, {(0,): -1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(129,): 1}),
            operator=negation,
            iterations=1,
            expected=_polynomial(variables, {(129,): -1}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(129,): -1})
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result

    wide_source = _polynomial(
        variables,
        dict.fromkeys(((index,) for index in range(520)), 1),
    )
    assert apply_constant_coefficient_differential_operator(
        wide_source,
        negation,
    ) == _polynomial(
        variables,
        dict.fromkeys(((index,) for index in range(520)), -1),
    )


def test_signed_unit_scalar_iterate_admits_the_coefficient_boundary_source() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=coefficient, exponents=(1,)),)
        ),
    )
    negated = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="-1" + "0" * 32_767, den="1"),
                    exponents=(1,),
                ),
            )
        ),
    )
    negation = _operator(variables, {(0,): -1})

    for iterations in (1, 2, 3):
        result = compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=negation,
                iterations=iterations,
                expected=source if iterations % 2 == 0 else negated,
            )
        )
        replayed = DifferentialOperatorApplyResult.model_validate(
            result.model_dump(mode="json")
        )

        assert result.output == (source if iterations % 2 == 0 else negated)
        assert result.matches_expected is True
        assert result.is_zero is False
        assert replayed == result


def test_signed_unit_scalar_iterate_keeps_multiterm_heights_unchanged() -> None:
    variables = ("x", "y")
    tall = 10**32_767
    source = _polynomial(
        variables,
        {(2, 5): -tall, (0, 0): 7 * 10 ** (32_766)},
    )
    negation = _operator(variables, {(0, 0): -1})

    output = apply_constant_coefficient_differential_operator(
        source,
        negation,
        iterations=5,
    )

    assert output == _polynomial(
        variables,
        {(2, 5): tall, (0, 0): -7 * 10 ** (32_766)},
    )


def test_nonunit_scalar_growth_still_gates_at_the_coefficient_budget() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=coefficient, exponents=(1,)),)
        ),
    )

    with pytest.raises(ValidationError, match="coefficient-digit budget"):
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(0,): 11}),
            iterations=1,
        )


def test_componentwise_annihilation_admits_off_axis_sources_beyond_caps() -> None:
    variables = ("x", "y")
    tall_source = _polynomial(variables, {(0, 129): 1})

    vanished = apply_constant_coefficient_differential_operator(
        tall_source,
        _operator(variables, {(1, 0): 1}),
    )
    crossed_axes = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(0, 200): 1}),
        _operator(variables, {(1, 1): 1}),
    )

    assert vanished == _polynomial(variables, {})
    assert crossed_axes == _polynomial(variables, {})


def test_per_axis_annihilation_requires_strict_exponent_excess() -> None:
    variables = ("x", "y")

    annihilated = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(0, 200): 1}),
        _operator(variables, {(1, 1): 1}),
    )
    boundary = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(1, 1): 1}),
        _operator(variables, {(1, 1): 1}),
    )
    mixed_operator = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(1, 1): 1}),
        _operator(variables, {(1, 0): 1, (0, 1): 1}),
    )

    assert annihilated == _polynomial(variables, {})
    assert boundary == _polynomial(variables, {(0, 0): 1})
    assert mixed_operator == _polynomial(variables, {(1, 0): 1, (0, 1): 1})
    assert apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(0, 129): 1}),
        _operator(variables, {(0, 1): 1}),
    ) == _polynomial(variables, {(0, 128): 129})


def test_guaranteed_zero_admits_sources_beyond_expansion_caps() -> None:
    variables = ("x", "y")
    wide_source = _polynomial(
        variables,
        dict.fromkeys(
            ((index // 25, index % 25) for index in range(600)),
            1,
        ),
    )
    steep_operator = _operator(variables, {(64, 0): 1})

    vanished = apply_constant_coefficient_differential_operator(
        wide_source,
        steep_operator,
        iterations=9,
    )
    tall_vanished = apply_constant_coefficient_differential_operator(
        _polynomial(("x",), {(200,): 3}),
        _operator(("x",), {(64,): 1}),
        iterations=4,
    )

    assert len(wide_source.polynomial.terms) == 600
    assert vanished == _polynomial(variables, {})
    assert vanished.variables == variables
    assert tall_vanished == _polynomial(("x",), {})


def test_input_digit_budget_only_gates_paths_that_expand() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num=str(10**300 - 1), den="1")
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=coefficient, exponents=(0,)),)
        ),
    )

    copied = apply_constant_coefficient_differential_operator(
        source,
        _operator(variables, {(1,): 1}),
        iterations=0,
    )
    vanished = apply_constant_coefficient_differential_operator(
        source,
        _operator(variables, {(1,): 1}),
        iterations=1,
    )

    assert copied == source
    assert vanished == _polynomial(variables, {})
    assert (
        apply_constant_coefficient_differential_operator(
            source,
            _operator(variables, {(0,): 1}),
            iterations=1,
        )
        == source
    )
    assert apply_constant_coefficient_differential_operator(
        source,
        _operator(variables, {(0,): 2}),
        iterations=1,
    ) == RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=str(2 * (10**300 - 1)), den="1"),
                    exponents=(0,),
                ),
            )
        ),
    )
    with pytest.raises(ValueError, match="256-digit bound"):
        apply_constant_coefficient_differential_operator(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=coefficient,
                            exponents=(5,),
                        ),
                    )
                ),
            ),
            _operator(variables, {(1,): 1}),
            iterations=1,
        )


def test_operator_digit_budget_only_gates_paths_that_power_the_operator() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num=str(10**300 - 1), den="1")
    operator = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=(DifferentialOperatorTerm(coefficient=coefficient, orders=(1,)),),
    )

    copied = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(5,): 1}),
        operator,
        iterations=0,
    )
    vanished = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(0,): 1}),
        operator,
    )

    assert copied == _polynomial(variables, {(5,): 1})
    assert vanished == _polynomial(variables, {})
    with pytest.raises(ValueError, match="256-digit bound"):
        apply_constant_coefficient_differential_operator(
            _polynomial(variables, {(5,): 1}),
            operator,
        )


def test_degenerate_shortcuts_still_honor_the_retained_byte_budget() -> None:
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")

    def oversized_source(term_count: int) -> RationalPolynomial:
        return RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=coefficient,
                        exponents=(exponent,),
                    )
                    for exponent in reversed(range(term_count))
                )
            ),
        )

    admitted = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=oversized_source(280),
            operator=_operator(("x",), {}),
            iterations=1,
        )
    )
    assert admitted.is_zero is True

    with pytest.raises(ValidationError, match="serialized-output budget"):
        DifferentialOperatorApplyRequest(
            polynomial=oversized_source(300),
            operator=_operator(("x",), {}),
            iterations=1,
        )
    per_term_bytes = 32_768 + 1 + 3 + 2 + 96
    assert 280 * per_term_bytes <= MAX_APPLICATION_RESULT_BYTES
    assert 300 * per_term_bytes > MAX_APPLICATION_RESULT_BYTES


def test_operator_and_polynomial_axes_must_match_exactly() -> None:
    with pytest.raises(ValidationError, match="same ordered variables"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            operator=_operator(("y", "x"), {(1, 0): 1}),
        )


def test_expected_polynomial_uses_the_same_axis() -> None:
    with pytest.raises(ValidationError, match="expected polynomial"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            operator=_operator(("x", "y"), {(1, 0): 1}),
            expected=_polynomial(("y", "x"), {(0, 0): 1}),
        )


def test_expected_comparison_admits_values_beyond_the_kernel_regime() -> None:
    variables = ("x",)
    unreachable = _polynomial(variables, {(129,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(1,): 1}),
            operator=_operator(variables, {(1,): 1}),
            iterations=1,
            expected=unreachable,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(0,): 1})
    assert result.matches_expected is False
    assert replayed == result


def test_expected_retention_still_honors_the_retained_byte_budget() -> None:
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")
    heavy_expected = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=(exponent,),
                )
                for exponent in reversed(range(300))
            )
        ),
    )

    with pytest.raises(ValidationError, match="serialized-output budget"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(1,): 1}),
            operator=_operator(("x",), {(1,): 1}),
            iterations=1,
            expected=heavy_expected,
        )


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        (
            (
                DifferentialOperatorTerm(coefficient=_rational(1), orders=(0, 1)),
                DifferentialOperatorTerm(coefficient=_rational(1), orders=(1, 0)),
            ),
            "descending lexicographic",
        ),
        (
            (
                DifferentialOperatorTerm(coefficient=_rational(1), orders=(1, 0)),
                DifferentialOperatorTerm(coefficient=_rational(2), orders=(1, 0)),
            ),
            "unique",
        ),
    ],
)
def test_operator_terms_must_be_canonical(
    terms: tuple[DifferentialOperatorTerm, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ConstantCoefficientDifferentialOperator(
            variables=("x", "y"),
            terms=terms,
        )


def test_operator_rejects_zero_coefficient_terms() -> None:
    with pytest.raises(ValidationError, match="zero differential-operator"):
        DifferentialOperatorTerm(coefficient=_rational(0), orders=(1, 0))


def test_sparse_high_degree_sources_are_admitted_by_derived_derivative_work() -> None:
    derivative = _operator(("x",), {(1,): 1})
    accepted = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(("x",), {(128,): 1}),
        operator=derivative,
    )
    assert compute_differential_operator_application(accepted).output == _polynomial(
        ("x",),
        {(127,): 128},
    )

    first_derivative = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(129,): 1}),
            operator=derivative,
        )
    )
    assert first_derivative.output == _polynomial(("x",), {(128,): 129})

    second_derivative = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(3_200,): 1}),
            operator=_operator(("x",), {(2,): 1}),
        )
    )
    assert second_derivative.output == _polynomial(
        ("x",),
        {(3_198,): 3_200 * 3_199},
    )

    dense = _polynomial(
        ("x",),
        dict.fromkeys(((index,) for index in range(513)), 1),
    )
    assert apply_constant_coefficient_differential_operator(dense, derivative) == (
        _polynomial(("x",), {(index - 1,): index for index in range(1, 513)})
    )


def test_tall_orders_cross_the_former_total_order_cap() -> None:
    factorial = math.factorial(65)
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(65,): 1}),
            operator=_operator(("x",), {(65,): 1}),
            iterations=1,
            expected=_polynomial(("x",), {(0,): factorial}),
        )
    )

    assert result.output == _polynomial(("x",), {(0,): factorial})
    assert result.matches_expected is True
    assert result.is_zero is False


def test_orders_stay_inside_the_interoperable_integer_range() -> None:
    boundary = DifferentialOperatorTerm(
        coefficient=_rational(1),
        orders=((1 << 53) - 1,),
    )
    assert boundary.orders == ((1 << 53) - 1,)
    with pytest.raises(ValidationError, match="less than or equal"):
        DifferentialOperatorTerm(
            coefficient=_rational(1),
            orders=((1 << 53),),
        )


def test_sixty_five_term_operators_follow_derived_support_budgets() -> None:
    variables = ("x",)
    operator = _operator(
        variables,
        dict.fromkeys(((order,) for order in range(65)), 1),
    )
    expected = _polynomial(
        variables,
        {
            (64 - order,): math.factorial(64) // math.factorial(64 - order)
            for order in range(65)
        },
    )
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(64,): 1}),
            operator=operator,
            iterations=1,
            expected=expected,
        )
    )

    assert result.output == expected
    assert result.matches_expected is True


def test_dense_source_boundary_follows_the_candidate_support_budget() -> None:
    variables = ("x",)
    derivative_plus_one = _operator(variables, {(1,): 1, (0,): 1})

    boundary = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(
            variables,
            dict.fromkeys(((index,) for index in range(2_048)), 1),
        ),
        operator=derivative_plus_one,
    )
    assert isinstance(
        compute_differential_operator_application(boundary),
        DifferentialOperatorApplyResult,
    )

    with pytest.raises(ValidationError, match="candidate-term budget"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(
                variables,
                dict.fromkeys(((index,) for index in range(2_049)), 1),
            ),
            operator=derivative_plus_one,
        )


def test_canonical_width_source_is_admitted_to_the_last_expanding_term() -> None:
    accepted = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(
            ("x",),
            dict.fromkeys(((index,) for index in range(MAX_POLYNOMIAL_TERMS)), 1),
        ),
        operator=_operator(("x",), {(1,): 1}),
    )
    result = compute_differential_operator_application(accepted)

    assert result.output == _polynomial(
        ("x",),
        {(index - 1,): index for index in range(1, MAX_POLYNOMIAL_TERMS)},
    )
    assert result.is_zero is False


def test_astronomical_orders_annihilate_without_expanding() -> None:
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(65,): 1}),
            operator=_operator(("x",), {(10**15,): 1}),
            iterations=1,
        )
    )

    assert result.is_zero is True
    assert result.output == _polynomial(("x",), {})


def test_mixed_astronomical_orders_keep_only_surviving_powered_terms() -> None:
    annihilator = _operator(("x",), {(10**15,): 1, (1,): -1})
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(65,): 1}),
            operator=annihilator,
            iterations=2,
            expected=_polynomial(("x",), {(63,): 65 * 64}),
        )
    )

    assert result.output == _polynomial(("x",), {(63,): 65 * 64})
    assert result.matches_expected is True


def test_sparse_power_work_boundary_is_admitted_then_rejected() -> None:
    source = _polynomial(("x",), {(128,): 1})
    operator = _operator(("x",), {(1,): 1, (0,): 1})

    accepted = DifferentialOperatorApplyRequest(
        polynomial=source,
        operator=operator,
        iterations=1_432,
    )
    assert isinstance(
        compute_differential_operator_application(accepted),
        DifferentialOperatorApplyResult,
    )

    with pytest.raises(ValidationError, match="deterministic work budget"):
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1_433,
        )


def test_candidate_output_term_boundary_is_admitted_then_rejected() -> None:
    exponents = sorted(
        ((left, right) for left in range(23) for right in range(23 - left)),
        reverse=True,
    )[:256]
    source = _polynomial(("x", "y"), dict.fromkeys(exponents, 1))
    operator = _operator(("x", "y"), {(1, 0): 1, (0, 0): 1})

    accepted = DifferentialOperatorApplyRequest(
        polynomial=source,
        operator=operator,
        iterations=15,
    )
    assert isinstance(
        compute_differential_operator_application(accepted),
        DifferentialOperatorApplyResult,
    )

    with pytest.raises(ValidationError, match="candidate-term budget"):
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=16,
        )


def test_coefficient_growth_boundary_is_admitted_then_rejected() -> None:
    source = _polynomial(("x",), {(0,): 1})
    coefficient = CanonicalRational(num=str(10**255), den="1")
    operator = ConstantCoefficientDifferentialOperator(
        variables=("x",),
        terms=(DifferentialOperatorTerm(coefficient=coefficient, orders=(0,)),),
    )

    accepted = DifferentialOperatorApplyRequest(
        polynomial=source,
        operator=operator,
        iterations=128,
    )
    result = compute_differential_operator_application(accepted)
    assert len(result.output.polynomial.terms[0].coefficient.num) == 32_641
    assert (
        len(encode_strict_json(result.model_dump(mode="json")))
        <= MAX_APPLICATION_RESULT_BYTES
    )

    with pytest.raises(ValidationError, match="coefficient-digit budget"):
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=129,
        )


def test_result_replay_rejects_source_operator_iteration_and_output_mutations() -> None:
    request = TOOLS[0].request_type.model_validate(TOOLS[0].examples[0].input)
    result = compute_differential_operator_application(request)

    for mutate in (
        lambda payload: payload.__setitem__("iterations", 1),
        lambda payload: payload["operator"]["terms"][0]["coefficient"].__setitem__(
            "num", "-1"
        ),
        lambda payload: payload["output"]["polynomial"]["terms"][0][
            "coefficient"
        ].__setitem__("num", "-5"),
    ):
        payload = result.model_dump(mode="json")
        mutate(payload)
        with pytest.raises(ValidationError, match="not bound"):
            DifferentialOperatorApplyResult.model_validate(payload)


def test_result_rejects_forged_zero_and_expected_decisions() -> None:
    request = TOOLS[0].request_type.model_validate(TOOLS[0].examples[0].input)
    result = compute_differential_operator_application(request)

    payload = result.model_dump(mode="json")
    payload["is_zero"] = True
    with pytest.raises(ValidationError, match="is_zero"):
        DifferentialOperatorApplyResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["matches_expected"] = False
    with pytest.raises(ValidationError, match="matches_expected"):
        DifferentialOperatorApplyResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["expected"]["polynomial"]["terms"][-1]["coefficient"]["num"] = "5"
    with pytest.raises(ValidationError, match="matches_expected"):
        DifferentialOperatorApplyResult.model_validate(payload)


def test_dvorsky_finite_identities_for_m_one_through_six() -> None:
    a, b, c, d, t = sympy.symbols("a b c d t")
    symbols = (a, b, c, d, t)
    variables = tuple(str(symbol) for symbol in symbols)
    p = (t + c) * (a * d + b * t)
    q = c
    lambda_operator = _operator(
        variables,
        {
            (1, 0, 0, 1, 1): 1,
            (0, 1, 1, 0, 1): -1,
        },
    )

    for m in range(1, 7):
        pure = compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_from_sympy(p**m, symbols),
                operator=lambda_operator,
                iterations=m,
                expected=_from_sympy(sympy.Integer(0), symbols),
            )
        )
        assert pure.is_zero is True
        assert pure.matches_expected is True

    for m in range(2, 7):
        expected = (-1) ** m * math.factorial(m) ** 2 * math.factorial(m + 1) * t
        with_q = compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_from_sympy(q * p**m, symbols),
                operator=lambda_operator,
                iterations=m,
                expected=_from_sympy(expected, symbols),
            )
        )
        assert with_q.matches_expected is True


def test_flint_kernel_matches_independent_sympy_replay() -> None:
    rng = random.Random(1177)
    x, y, z = sympy.symbols("x y z")
    symbols = (x, y, z)
    variables = ("x", "y", "z")

    for _ in range(20):
        source_terms: dict[tuple[int, ...], Fraction] = {}
        for _ in range(6):
            exponents = tuple(rng.randrange(5) for _ in variables)
            source_terms[exponents] = source_terms.get(
                exponents, Fraction()
            ) + Fraction(rng.randrange(-4, 5), rng.randrange(1, 5))
        operator_terms: dict[tuple[int, ...], Fraction] = {}
        while len(operator_terms) < 3:
            orders = tuple(rng.randrange(3) for _ in variables)
            if sum(orders) <= 3:
                operator_terms[orders] = Fraction(
                    rng.choice((-2, -1, 1, 2)), rng.randrange(1, 4)
                )
        source = _polynomial(variables, source_terms)
        operator = _operator(variables, operator_terms)
        iterations = rng.randrange(4)

        expected_expression = sympy.Poly(
            sum(
                sympy.Rational(*term.coefficient.as_integer_ratio())
                * sympy.prod(
                    symbol**exponent
                    for symbol, exponent in zip(symbols, term.exponents, strict=True)
                )
                for term in source.polynomial.terms
            ),
            *symbols,
            domain=sympy.QQ,
        ).as_expr()
        for _ in range(iterations):
            expected_expression = sympy.expand(
                sum(
                    sympy.Rational(*term.coefficient.as_integer_ratio())
                    * (
                        sympy.diff(
                            expected_expression,
                            *tuple(
                                (symbol, order)
                                for symbol, order in zip(
                                    symbols, term.orders, strict=True
                                )
                                if order
                            ),
                        )
                        if any(term.orders)
                        else expected_expression
                    )
                    for term in operator.terms
                )
            )
        expected = _from_sympy(expected_expression, symbols)

        assert (
            apply_constant_coefficient_differential_operator(
                source,
                operator,
                iterations=iterations,
            )
            == expected
        )
