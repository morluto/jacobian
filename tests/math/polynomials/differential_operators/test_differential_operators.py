"""Exact contract tests for constant-coefficient differential operators."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping
from fractions import Fraction

import pytest
import sympy
from tests.math.polynomials._support import polynomial_validation_error

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
from jacobian.math.polynomials.differential_operators._bounds import (
    MAX_APPLICATION_RESULT_BYTES,
    ApplicationEnvelope,
    _common_denominator_height,
    _decimal_digits_from_bits,
    _distinct_powered_orders,
    validate_application_envelope,
)
from jacobian.math.polynomials.differential_operators._models import (
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
)
from jacobian.math.polynomials.differential_operators._tools import (
    TOOLS,
    compute_differential_operator_application,
)
from jacobian.math.polynomials.differential_operators.operations import (
    apply_constant_coefficient_differential_operator,
)
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
    DifferentialOperatorTerm,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial[Order: tuple[int, ...]](
    variables: tuple[str, ...],
    terms: Mapping[Order, int | Fraction],
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


def _operator[Order: tuple[int, ...]](
    variables: tuple[str, ...],
    terms: Mapping[Order, int | Fraction],
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


def _coprime_tall_denominators(count: int, digits: int) -> list[int]:
    primes = list(sympy.primerange(2, sympy.prime(count + 1) + 1))
    return [prime ** math.ceil(digits / math.log10(prime)) for prime in primes[:count]]


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

    with pytest.raises(OperationDomainValidationError, match="must be an integer"):
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
        iterations=4_096,
    )

    assert output == _polynomial(variables, {})


def test_non_expanding_requests_are_admitted_at_any_iteration_count() -> None:
    variables = ("x", "y")
    zero_operator = _operator(variables, {})
    tall_iterations = 4_097

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


def test_expanding_iterates_gate_on_derived_support_and_growth() -> None:
    variables = ("x",)
    scaling = _operator(variables, {(0,): 2})
    source = _polynomial(variables, {(3,): 1})
    tall_iterations = 4_097

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
    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=expanding,
                iterations=tall_iterations,
            )
        )
    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=expanding,
                iterations=10**12,
            )
        )
    with pytest.raises(ValueError, match="nonnegative"):
        apply_constant_coefficient_differential_operator(
            source,
            expanding,
            iterations=-1,
        )


def test_tall_expanding_iterates_are_admitted_by_derived_budgets() -> None:
    derivative = _operator(("x",), {(1,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(5_000,): 1}),
            operator=derivative,
            iterations=4_097,
            expected=_polynomial(("x",), {(903,): math.perm(5_000, 4_097)}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(("x",), {(903,): math.perm(5_000, 4_097)})
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result

    annihilated = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(("x",), {(5_000,): 1}),
        operator=derivative,
        iterations=10**12,
    )
    assert annihilated.iterations == 10**12
    assert compute_differential_operator_application(annihilated).output == _polynomial(
        ("x",),
        {},
    )


def test_colliding_powered_orders_are_counted_distinctly() -> None:
    variables = ("x",)
    operator = _operator(
        variables,
        dict.fromkeys(((order,) for order in range(100)), 1),
    )
    source = _polynomial(variables, {(200,): 1})

    def paths(order: int) -> int:
        return min(order, 99) - max(0, order - 99) + 1

    expected = _polynomial(
        variables,
        {(200 - order,): paths(order) * math.perm(200, order) for order in range(199)},
    )
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=2,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_annihilating_powered_terms_are_excluded_from_the_candidate_cap() -> None:
    variables = ("x",)
    operator = _operator(
        variables,
        {(0,): 1, **dict.fromkeys(((order,) for order in range(2, 1_050)), 1)},
    )
    source = _polynomial(variables, {(0,): 1, (1,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1,
            expected=source,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == source
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_scalar_iterate_growth_is_bounded_by_the_coefficient_budget() -> None:
    variables = ("x",)
    scaling = _operator(variables, {(0,): 2})

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(0,): 1}),
                operator=scaling,
                iterations=400_000,
            )
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


def test_no_growth_derivative_regime_keeps_multivariate_heights() -> None:
    variables = ("x", "y")
    tall = 10**32_767
    source = _polynomial(variables, {(1, 1): tall})

    crossed = apply_constant_coefficient_differential_operator(
        source,
        _operator(variables, {(1, 1): 1}),
    )
    split = apply_constant_coefficient_differential_operator(
        source,
        _operator(variables, {(1, 0): 1, (0, 1): 1}),
    )

    assert crossed == _polynomial(variables, {(0, 0): tall})
    assert split == _polynomial(variables, {(1, 0): tall, (0, 1): tall})


def test_no_growth_derivative_mixed_survival_and_annihilation_keeps_heights() -> None:
    variables = ("x",)
    tall = 10**32_767

    output = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(1,): tall, (0,): tall}),
        _operator(variables, {(1,): 1}),
    )

    assert output == _polynomial(variables, {(0,): tall})


def test_merged_unit_paths_still_gate_at_the_coefficient_budget() -> None:
    variables = ("x", "y")
    tall = 6 * 10**32_767

    # Under 1 + ∂x the monomials x·y and y both contribute a unit-height
    # copy to the output monomial y, so its merged coefficient 12·10^32767
    # carries 32,769 digits even though each path alone preserves height.
    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(1, 1): tall, (0, 1): tall}),
                operator=_operator(variables, {(0, 0): 1, (1, 0): 1}),
                iterations=1,
            )
        )


@pytest.mark.scale
def test_growing_derivatives_still_gate_at_the_coefficient_budget() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")

    def boundary_source(exponent: int) -> RationalPolynomial:
        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=coefficient,
                        exponents=(exponent,),
                    ),
                )
            ),
        )

    # A small multiplier keeps the boundary height inside its digit count.
    doubled = apply_constant_coefficient_differential_operator(
        boundary_source(2),
        _operator(variables, {(1,): 1}),
    )
    assert doubled == _polynomial(variables, {(1,): 2 * 10**32_767})

    # Differentiating x^11 multiplies the boundary height by 11, producing a
    # 32,769-digit coefficient that exceeds the budget.
    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=boundary_source(11),
                operator=_operator(variables, {(1,): 1}),
                iterations=1,
            )
        )

    # A coefficient-10 derivative scales the surviving height to 10^32768,
    # whose 32,769 digits also exceed the budget.
    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=boundary_source(1),
                operator=_operator(variables, {(1,): 10}),
                iterations=1,
            )
        )


@pytest.mark.scale
def test_nonunit_scalar_growth_still_gates_at_the_coefficient_budget() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num="1" + "0" * 32_767, den="1")
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=coefficient, exponents=(1,)),)
        ),
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=_operator(variables, {(0,): 11}),
                iterations=1,
            )
        )


@pytest.mark.scale
def test_no_growth_derivatives_are_admitted_at_the_coefficient_boundary() -> None:
    variables = ("x",)
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1" + "0" * 32_767, den="1"),
                    exponents=(1,),
                ),
            )
        ),
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(1,): 1}),
            iterations=1,
            expected=_polynomial(variables, {(0,): 10**32_767}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(0,): 10**32_767})
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result
    assert len(result.output.polynomial.terms[0].coefficient.num) == 32_768


@pytest.mark.scale
def test_multinomial_path_multiplicity_gates_the_coefficient_bound() -> None:
    variables = ("x",)
    operator = _operator(variables, {(0,): 1, (1,): 1})
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1" + "0" * 32_701, den="1"),
                    exponents=(32,),
                ),
            )
        ),
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=140,
            )
        )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=20,
            expected=_polynomial(
                variables,
                {
                    (32 - order,): math.comb(20, order)
                    * math.perm(32, order)
                    * 10**32_701
                    for order in range(21)
                },
            ),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.matches_expected is True
    assert result.is_zero is False
    assert len(result.output.polynomial.terms[0].coefficient.num) > 32_700
    assert replayed == result


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


@pytest.mark.scale
def test_tall_source_coefficients_are_admitted_by_derived_growth() -> None:
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
    differentiated = apply_constant_coefficient_differential_operator(
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
    assert differentiated == _polynomial(
        variables,
        {(4,): 5 * (10**300 - 1)},
    )

    reviewer = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1" + "0" * 299, den="1"),
                            exponents=(1,),
                        ),
                    )
                ),
            ),
            operator=_operator(variables, {(1,): 1}),
            iterations=1,
            expected=_polynomial(variables, {(0,): 10**299}),
        )
    )
    assert reviewer.output == _polynomial(variables, {(0,): 10**299})
    assert reviewer.matches_expected is True

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(
                    variables,
                    {(32_768,): 10**32_000},
                ),
                operator=_operator(variables, {(1,): 1}),
                iterations=200,
            )
        )


def test_tall_operator_coefficients_are_admitted_by_derived_growth() -> None:
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
    applied = apply_constant_coefficient_differential_operator(
        _polynomial(variables, {(5,): 1}),
        operator,
    )

    assert copied == _polynomial(variables, {(5,): 1})
    assert vanished == _polynomial(variables, {})
    assert applied == _polynomial(
        variables,
        {(4,): 5 * (10**300 - 1)},
    )

    reviewer = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(1,): 1}),
            operator=ConstantCoefficientDifferentialOperator(
                variables=variables,
                terms=(
                    DifferentialOperatorTerm(
                        coefficient=CanonicalRational(num="1" + "0" * 299, den="1"),
                        orders=(1,),
                    ),
                ),
            ),
            iterations=1,
            expected=_polynomial(variables, {(0,): 10**299}),
        )
    )
    assert reviewer.output == _polynomial(variables, {(0,): 10**299})
    assert reviewer.matches_expected is True

    with pytest.raises(ValueError, match="coefficient-digit budget"):
        apply_constant_coefficient_differential_operator(
            _polynomial(variables, {(2,): 1}),
            ConstantCoefficientDifferentialOperator(
                variables=variables,
                terms=(
                    DifferentialOperatorTerm(
                        coefficient=CanonicalRational(num="1" + "0" * 32_700, den="1"),
                        orders=(1,),
                    ),
                ),
            ),
            iterations=2,
        )


@pytest.mark.scale
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

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=oversized_source(300),
                operator=_operator(("x",), {}),
                iterations=1,
            )
        )
    per_term_bytes = 32_768 + 1 + 3 + 2 + 96
    assert 280 * per_term_bytes <= MAX_APPLICATION_RESULT_BYTES
    assert 300 * per_term_bytes > MAX_APPLICATION_RESULT_BYTES


def test_operator_and_polynomial_axes_must_match_exactly() -> None:
    with polynomial_validation_error():
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            operator=_operator(("y", "x"), {(1, 0): 1}),
        )


def test_expected_polynomial_uses_the_same_axis() -> None:
    with polynomial_validation_error():
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


@pytest.mark.scale
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

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(("x",), {(1,): 1}),
                operator=_operator(("x",), {(1,): 1}),
                iterations=1,
                expected=heavy_expected,
            )
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
    with polynomial_validation_error():
        ConstantCoefficientDifferentialOperator(
            variables=("x", "y"),
            terms=terms,
        )


def test_operator_rejects_zero_coefficient_terms() -> None:
    with polynomial_validation_error():
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


def test_dense_expanding_source_is_admitted_by_derived_budgets() -> None:
    variables = ("x",)
    derivative = _operator(variables, {(1,): 1})
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
    with polynomial_validation_error():
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

    # Acting pairs overcount the result: an identity shift and a derivative
    # shift collide on one output exponent each, so the candidate budget
    # follows distinct target exponents instead of (term, shift) pairs.
    merged = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(
            variables,
            dict.fromkeys(((index,) for index in range(2_049)), 1),
        ),
        operator=derivative_plus_one,
    )
    assert isinstance(
        compute_differential_operator_application(merged),
        DifferentialOperatorApplyResult,
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(
                    variables,
                    dict.fromkeys(((4 * index,) for index in range(1_100)), 1),
                ),
                operator=_operator(variables, {(3,): 1, (2,): 1, (1,): 1, (0,): 1}),
                iterations=1,
            )
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


def test_full_width_operators_follow_the_shared_term_representation() -> None:
    variables = ("x",)
    full_width = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=tuple(
            DifferentialOperatorTerm(coefficient=_rational(1), orders=(order,))
            for order in reversed(range(MAX_POLYNOMIAL_TERMS))
        ),
    )

    assert apply_constant_coefficient_differential_operator(
        _polynomial(variables, {}),
        full_width,
    ) == _polynomial(variables, {})

    with polynomial_validation_error():
        ConstantCoefficientDifferentialOperator(
            variables=variables,
            terms=tuple(
                DifferentialOperatorTerm(coefficient=_rational(1), orders=(order,))
                for order in reversed(range(MAX_POLYNOMIAL_TERMS + 1))
            ),
        )


def test_tall_order_growth_gates_on_the_coefficient_budget() -> None:
    edge = MAX_POLYNOMIAL_EXPONENT

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(("x",), {(edge,): 1}),
                operator=_operator(("x",), {(edge,): 1}),
                iterations=1,
            )
        )


@pytest.mark.scale
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

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=1_433,
            )
        )


def test_candidate_output_term_boundary_is_admitted_then_rejected() -> None:
    variables = ("x",)
    operator = _operator(variables, {(1,): 1, (0,): 1})

    def source(term_count: int) -> RationalPolynomial:
        return _polynomial(
            variables,
            dict.fromkeys(((4 * index + 3,) for index in range(term_count)), 1),
        )

    accepted = DifferentialOperatorApplyRequest(
        polynomial=source(1_024),
        operator=operator,
        iterations=3,
    )
    assert isinstance(
        compute_differential_operator_application(accepted),
        DifferentialOperatorApplyResult,
    )
    expected_terms: dict[tuple[int, ...], int] = {}
    for index in range(1_024):
        exponent = 4 * index + 3
        for order in range(4):
            key = (exponent - order,)
            expected_terms[key] = expected_terms.get(key, 0) + math.comb(
                3, order
            ) * math.perm(exponent, order)
    result = compute_differential_operator_application(accepted)
    assert result.output == _polynomial(variables, expected_terms)

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source(1_025),
                operator=operator,
                iterations=3,
            )
        )


def test_correlated_powered_axes_are_counted_distinctly() -> None:
    variables = ("x", "y")
    operator = _operator(
        variables,
        dict.fromkeys(((order, order) for order in range(100)), 1),
    )
    source = _polynomial(variables, {(200, 200): 1})

    def paths(order: int) -> int:
        return min(order, 99) - max(0, order - 99) + 1

    expected = _polynomial(
        variables,
        {
            (200 - order, 200 - order): paths(order) * math.perm(200, order) ** 2
            for order in range(199)
        },
    )
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=2,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_correlated_support_survives_the_enumeration_cutoff() -> None:
    variables = ("x", "y")
    terms = {(0, 0): 1}
    terms.update({(order, order): 1 for order in range(600)})
    terms[(1, 0)] = 1
    # D^2(x+y) = x+y+2: only the identity and first-order diagonal shifts act,
    # so the exact 1,800-shift correlated power must stay admissible even
    # though its pair enumeration outworks the coarse fixed cutoff.
    expected = _polynomial(variables, {(1, 0): 1, (0, 1): 1, (0, 0): 2})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(1, 0): 1, (0, 1): 1}),
            operator=_operator(variables, terms),
            iterations=2,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_widened_correlated_power_still_follows_the_work_budget() -> None:
    variables = ("x", "y")
    terms = {(0, 0): 1}
    terms.update({(order, order): 1 for order in range(1_000)})
    terms[(1, 0)] = 1

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(1, 0): 1, (0, 1): 1}),
                operator=_operator(variables, terms),
                iterations=2,
            )
        )


def test_per_monomial_annihilation_is_counted_in_the_candidate_bound() -> None:
    variables = ("x", "y")
    operator = _operator(
        variables,
        {(i, j): 1 for i in range(32) for j in range(32)},
    )
    source = _polynomial(variables, {(32, 0): 1, (0, 32): 1})
    expected_terms: dict[tuple[int, int], int] = {}
    for exponent in range(32):
        coefficient = math.factorial(32) // math.factorial(32 - exponent)
        expected_terms[(32 - exponent, 0)] = (
            expected_terms.get((32 - exponent, 0), 0) + coefficient
        )
        expected_terms[(0, 32 - exponent)] = (
            expected_terms.get((0, 32 - exponent), 0) + coefficient
        )
    expected = _polynomial(variables, expected_terms)

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_scalar_on_source_regime_skips_unreachable_expansion() -> None:
    variables = ("x",)
    operator = _operator(variables, {(0,): 1, (1,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(0,): 1}),
            operator=operator,
            iterations=4_096,
            expected=_polynomial(variables, {(0,): 1}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(0,): 1})
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result

    growing = _operator(variables, {(0,): 2, (1,): 1})
    grown = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(0,): 1}),
            operator=growing,
            iterations=20,
            expected=_polynomial(variables, {(0,): 2**20}),
        )
    )
    assert grown.output == _polynomial(variables, {(0,): 2**20})
    assert grown.matches_expected is True

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(0,): 1}),
                operator=growing,
                iterations=200_000,
            )
        )


@pytest.mark.scale
def test_distinct_source_denominators_are_not_merged_without_collision() -> None:
    variables = ("x",)
    numerator_p = 10**20000 + 1
    numerator_q = 10**20000 + 3
    source = _polynomial(
        variables,
        {
            (2,): Fraction(1, numerator_p),
            (1,): Fraction(1, numerator_q),
        },
    )
    expected = _polynomial(
        variables,
        {
            (1,): Fraction(2, numerator_p),
            (0,): Fraction(1, numerator_q),
        },
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(1,): 1}),
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_signed_unit_scalars_short_circuit_by_exponent_parity() -> None:
    variables = ("x",)
    source = _polynomial(variables, {(5,): 3})
    negation = _operator(variables, {(0,): -1})
    even_iterations = 2**52
    odd_iterations = even_iterations + 1

    even = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=negation,
            iterations=even_iterations,
            expected=source,
        )
    )
    assert even.output == source
    assert even.matches_expected is True

    odd = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=negation,
            iterations=odd_iterations,
            expected=_polynomial(variables, {(5,): -3}),
        )
    )
    assert odd.output == _polynomial(variables, {(5,): -3})
    assert odd.matches_expected is True


@pytest.mark.scale
def test_cross_canceling_operator_weights_keep_true_heights() -> None:
    variables = ("x",)
    numerator_n = 10**20000 + 1
    source = _polynomial(variables, {(2,): Fraction(1, numerator_n)})
    operator = _operator(variables, {(1,): numerator_n})
    expected = _polynomial(variables, {(0,): 2 * numerator_n})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=2,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_annihilating_rescale_terms_are_excluded_from_growth() -> None:
    variables = ("x",)
    numerator_n = 10**20000 + 1
    source = _polynomial(variables, {(1,): 1})
    operator = _operator(variables, {(0,): 1, (2,): numerator_n})
    expected = _polynomial(variables, {(1,): 1})

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=2,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_weight_enumeration_aborts_before_huge_powers_materialize() -> None:
    variables = ("x",)
    numerator_n = CanonicalRational(num="1" + "0" * 32_767, den="1")
    source = _polynomial(variables, {(400,): 1})
    operator = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=(
            DifferentialOperatorTerm(coefficient=_rational(1), orders=(1,)),
            DifferentialOperatorTerm(coefficient=numerator_n, orders=(0,)),
        ),
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=400,
            )
        )


def test_annihilating_rescale_scan_follows_the_request_work_budget() -> None:
    variables = tuple("abcdefgh")
    terms = {(0,) * len(variables): 1}
    terms.update(
        {(order,) + (0,) * (len(variables) - 1): 1 for order in range(2, 1_002)}
    )
    operator = _operator(variables, terms)
    source = _polynomial(
        variables,
        {
            tuple((index >> shift) & 1 for shift in reversed(range(8))): 1
            for index in range(256)
        },
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1,
            expected=source,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == source
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_rescale_scan_overflow_falls_back_to_expansion_gates() -> None:
    variables = ("x",)
    source = _polynomial(variables, dict.fromkeys(((i,) for i in range(4_096)), 1))
    operator = _operator(
        variables,
        {(0,): 1, **dict.fromkeys(((j,) for j in range(2, 1_002)), 1)},
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=1,
            )
        )


@pytest.mark.scale
def test_rescale_scaling_reduces_against_source_denominators() -> None:
    variables = ("x",)
    numerator_n = "1" + "0" * 19999 + "1"
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den=numerator_n),
                    exponents=(1,),
                ),
            )
        ),
    )
    scaled_operator = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=(
            DifferentialOperatorTerm(
                coefficient=CanonicalRational(num=numerator_n, den="1"),
                orders=(0,),
            ),
        ),
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=scaled_operator,
            iterations=1,
            expected=_polynomial(variables, {(1,): 1}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(1,): 1})
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_boundary_scalar_coefficient_is_measured_exactly() -> None:
    variables = ("x",)
    coefficient = CanonicalRational(num="9" * 32_768, den="1")
    source = _polynomial(variables, {(0,): 1})
    operator = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=(DifferentialOperatorTerm(coefficient=coefficient, orders=(0,)),),
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1,
            expected=_polynomial(variables, {(0,): Fraction(10**32_768 - 1)}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(0,): 10**32_768 - 1})
    assert len(result.output.polynomial.terms[0].coefficient.num) == 32_768
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_wide_operator_weight_growth_gates_at_the_height_cap() -> None:
    variables = ("x",)
    n_coefficient = CanonicalRational(num="1" + "0" * 1_499, den="1")
    source = _polynomial(variables, {(400,): 1})
    operator = ConstantCoefficientDifferentialOperator(
        variables=variables,
        terms=(
            DifferentialOperatorTerm(coefficient=n_coefficient, orders=(2,)),
            DifferentialOperatorTerm(coefficient=_rational(1), orders=(1,)),
        ),
    )

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=400,
            )
        )


@pytest.mark.scale
def test_coprime_tall_operator_denominators_defer_the_shared_height() -> None:
    variables = ("x",)

    def coprime_operator(
        digit_count: int,
    ) -> tuple[ConstantCoefficientDifferentialOperator, list[int]]:
        denominators = _coprime_tall_denominators(1_024, digit_count)
        operator = _operator(
            variables,
            {(order,): Fraction(1, denominators[order]) for order in range(1_024)},
        )
        return operator, denominators

    # Only the order-0 and order-1 aggregates act on x, so admission answers
    # from the per-exponent accounting alone and never constructs a shared
    # height across the 1,024 pairwise-coprime tall denominators.
    tall_operator, _ = coprime_operator(512)
    envelope = validate_application_envelope(
        _polynomial(variables, {(1,): 1}),
        tall_operator,
        1,
        None,
    )
    assert envelope == ApplicationEnvelope(
        guaranteed_zero=False,
        expanded_operator_terms=1_024,
        candidate_output_terms=2,
        rescale_only=False,
    )

    # Inside the kernel's coefficient regime the same shape applies exactly:
    # D(x) = x/q_0 + 1/q_1 with every nonacting term discarded.
    operator, denominators = coprime_operator(24)
    expected = _polynomial(
        variables,
        {(1,): Fraction(1, denominators[0]), (0,): Fraction(1, denominators[1])},
    )
    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(1,): 1}),
            operator=operator,
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_falling_factorial_growth_is_measured_exactly() -> None:
    variables = ("x",)
    factorial = math.factorial(8_500)

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(8_500,): 1}),
            operator=_operator(variables, {(8_500,): 1}),
            iterations=1,
            expected=_polynomial(variables, {(0,): factorial}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(0,): factorial})
    assert len(result.output.polynomial.terms[0].coefficient.num) == 29_711
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(8_501,): 10**32_000}),
                operator=_operator(variables, {(8_501,): 1}),
                iterations=1,
            )
        )


@pytest.mark.scale
def test_falling_factorial_cancels_against_source_denominators() -> None:
    variables = ("x",)
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        Fraction(1, math.factorial(2_000))
                    ),
                    exponents=(10_000,),
                ),
            )
        ),
    )
    expected = _polynomial(
        variables,
        {
            (0,): Fraction(math.factorial(10_000), math.factorial(2_000)),
        },
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(10_000,): 1}),
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert len(result.output.polynomial.terms[0].coefficient.num) == 29_924
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_class_sums_are_reduced_before_height_measurement() -> None:
    variables = ("x",)
    big = 10**32_767
    shared_denominator = 6 * big + 1
    source = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        Fraction(3 * big, shared_denominator)
                    ),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        Fraction(9 * big + 2, shared_denominator)
                    ),
                    exponents=(0,),
                ),
            )
        ),
    )
    operator = _operator(variables, {(0,): 1, (1,): 1})
    expected = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        Fraction(3 * big, shared_denominator)
                    ),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(2)),
                    exponents=(0,),
                ),
            )
        ),
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=operator,
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert len(result.output.polynomial.terms[0].coefficient.num) == 32_768
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


def test_colliding_output_exponents_share_the_candidate_budget() -> None:
    variables = ("x",)
    source = _polynomial(variables, {(degree,): 1 for degree in range(2_049)})
    # The identity shift maps x^j to x^j while the derivative shift maps
    # x^(j+1) to x^j, so all 4,097 acting pairs land on 2,049 output terms.
    expected = _polynomial(
        variables,
        {(degree,): degree + 2 for degree in range(2_048)} | {(2_048,): 1},
    )

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=source,
            operator=_operator(variables, {(0,): 1, (1,): 1}),
            iterations=1,
            expected=expected,
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == expected
    assert len(result.output.polynomial.terms) == 2_049
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result


@pytest.mark.scale
def test_class_heights_keep_signed_cancellation() -> None:
    variables = ("x",)
    height = 6 * 10**32_767

    result = compute_differential_operator_application(
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(variables, {(1,): height, (0,): height}),
            operator=_operator(variables, {(0,): 1, (1,): -1}),
            iterations=1,
            expected=_polynomial(variables, {(1,): height}),
        )
    )
    replayed = DifferentialOperatorApplyResult.model_validate(
        result.model_dump(mode="json")
    )

    assert result.output == _polynomial(variables, {(1,): height})
    assert len(result.output.polynomial.terms[0].coefficient.num) == 32_768
    assert result.matches_expected is True
    assert result.is_zero is False
    assert replayed == result

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=_polynomial(variables, {(1,): height, (0,): height}),
                operator=_operator(variables, {(0,): 1, (1,): 1}),
                iterations=1,
            )
        )


def test_iterations_stay_inside_the_interoperable_integer_range() -> None:
    schema = DifferentialOperatorApplyRequest.model_json_schema()
    assert schema["properties"]["iterations"]["maximum"] == (1 << 53) - 1

    with polynomial_validation_error():
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(0,): 1}),
            operator=_operator(("x",), {(0,): 1}),
            iterations=1 << 53,
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

    with pytest.raises(OperationDomainValidationError):
        compute_differential_operator_application(
            DifferentialOperatorApplyRequest(
                polynomial=source,
                operator=operator,
                iterations=129,
            )
        )


def test_result_rejects_forged_zero_and_expected_decisions() -> None:
    request = TOOLS[0].request_type.model_validate(TOOLS[0].examples[0].input)
    result = compute_differential_operator_application(request)

    payload = result.model_dump(mode="json")
    payload["is_zero"] = True
    with polynomial_validation_error():
        DifferentialOperatorApplyResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["matches_expected"] = False
    with polynomial_validation_error():
        DifferentialOperatorApplyResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["expected"]["polynomial"]["terms"][-1]["coefficient"]["num"] = "5"
    with polynomial_validation_error():
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


def test_common_denominator_height_gate_is_exact_at_its_boundary() -> None:
    # 2**1000 has exactly 302 decimal digits and its bit-length estimate
    # agrees, so the gate admits at 302 and refuses one digit earlier.
    assert _common_denominator_height([Fraction(1, 2**1000)], maximum_digits=302) == (
        2**1000,
        1,
    )
    assert (
        _common_denominator_height([Fraction(1, 2**1000)], maximum_digits=301) is None
    )
    assert _common_denominator_height([Fraction(3, 2**1000)], maximum_digits=302) == (
        2**1000,
        3,
    )
    assert _common_denominator_height([], maximum_digits=1) == (1, 0)

    # Defining invariant: the gated height is either the exact ungated pair or
    # a refusal whose ungated decimal length provably exceeds the budget.
    rng = random.Random(1177)
    for _ in range(200):
        values = [
            Fraction(
                rng.randrange(1, 50),
                rng.choice((2, 3, 5, 7)) ** rng.randrange(0, 40),
            )
            for _ in range(rng.randrange(1, 12))
        ]
        ungated = _common_denominator_height(values)
        assert ungated is not None
        cap = rng.randrange(1, 400)
        gated = _common_denominator_height(values, maximum_digits=cap)
        if gated is None:
            estimated = _decimal_digits_from_bits(ungated[0].bit_length())
            assert estimated > cap
        else:
            assert gated == ungated


def test_distinct_powered_orders_match_the_composition_family() -> None:
    def brute_force(
        orders: list[tuple[int, ...]], iterations: int
    ) -> set[tuple[int, ...]]:
        return {
            tuple(map(sum, zip(*combo, strict=True)))
            for combo in itertools.combinations_with_replacement(orders, iterations)
        }

    rng = random.Random(2342)
    for _ in range(25):
        axis_count = rng.randrange(1, 4)
        variables = tuple("xyz"[:axis_count])
        term_count = rng.randrange(1, 6)
        iterations = rng.randrange(1, 5)
        coefficients: dict[tuple[int, ...], int] = {
            tuple(rng.randrange(3) for _ in range(axis_count)): rng.choice((1, -1))
            for _ in range(term_count)
        }
        operator = _operator(variables, coefficients)
        orders = [term.orders for term in operator.terms]

        enumerated = _distinct_powered_orders(operator, iterations, 500, 100_000)
        assert enumerated is not None
        assert enumerated == tuple(sorted(brute_force(orders, iterations)))

    wide = _operator(("x", "y"), {(j, j): 1 for j in range(6)})
    zero_iterations = _distinct_powered_orders(wide, 0, 500, 100_000)
    assert zero_iterations == ()
    assert _distinct_powered_orders(wide, 2, 3, 100_000) is None
    assert _distinct_powered_orders(wide, 2, 500, 0) is None
