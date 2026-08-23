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
    MAX_APPLICATION_INPUT_EXPONENT,
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
    MAX_DIFFERENTIAL_OPERATOR_TOTAL_ORDER,
    ConstantCoefficientDifferentialOperator,
    DifferentialOperatorTerm,
)
from jacobian.math.polynomials.values import (
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


def test_operator_rejects_zero_coefficients_and_excess_total_order() -> None:
    with pytest.raises(ValidationError, match="zero differential-operator"):
        DifferentialOperatorTerm(coefficient=_rational(0), orders=(1, 0))
    with pytest.raises(ValidationError, match="total-order"):
        DifferentialOperatorTerm(
            coefficient=_rational(1),
            orders=(MAX_DIFFERENTIAL_OPERATOR_TOTAL_ORDER, 1),
        )


def test_degree_boundary_is_admitted_then_rejected() -> None:
    operator = _operator(("x",), {(1,): 1})
    accepted = DifferentialOperatorApplyRequest(
        polynomial=_polynomial(("x",), {(MAX_APPLICATION_INPUT_EXPONENT,): 1}),
        operator=operator,
    )
    assert compute_differential_operator_application(accepted).output == _polynomial(
        ("x",),
        {(MAX_APPLICATION_INPUT_EXPONENT - 1,): MAX_APPLICATION_INPUT_EXPONENT},
    )

    with pytest.raises(ValidationError, match="degree operation budget"):
        DifferentialOperatorApplyRequest(
            polynomial=_polynomial(("x",), {(MAX_APPLICATION_INPUT_EXPONENT + 1,): 1}),
            operator=operator,
        )


def test_sparse_power_work_boundary_is_admitted_then_rejected() -> None:
    source = _polynomial(("x",), {(MAX_APPLICATION_INPUT_EXPONENT,): 1})
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
