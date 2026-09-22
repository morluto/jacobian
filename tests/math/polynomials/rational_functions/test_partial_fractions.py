"""Exact contract tests for rational partial-fraction decomposition."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from sympy import Poly, Rational, Symbol, cancel

from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions._models import (
    PartialFractionsRequest,
    PartialFractionsResult,
    PartialFractionTerm,
)
from jacobian.math.polynomials.rational_functions._tools import (
    TOOLS,
    compute_partial_fractions,
)
from jacobian.math.polynomials.rational_functions.operations import (
    partial_fractions,
    verify_partial_fractions,
)
from jacobian.math.polynomials.rational_functions.structured_models import (
    RationalPrimitiveResult,
)

x = Symbol("x")


def _decompose(expression: object) -> PartialFractionsResult:
    function = rational_function_from_sympy(expression, ("x",))
    return partial_fractions(function)


def _term_key(term: PartialFractionTerm) -> tuple[str, int]:
    return (str(rational_polynomial_to_sympy(term.factor).as_expr()), term.exponent)


def _terms_by_pole(result: PartialFractionsResult) -> dict[tuple[str, int], Poly]:
    return {
        _term_key(term): rational_polynomial_to_sympy(term.numerator)
        for term in result.terms
    }


def test_repeated_pole_known_answer() -> None:
    expression = 1 / ((x - 1) ** 2 * (x + 1))

    result = _decompose(expression)

    assert rational_polynomial_to_sympy(result.polynomial_part).as_expr() == 0
    assert {
        (str(rational_polynomial_to_sympy(factor.factor).as_expr()), factor.exponent)
        for factor in result.factors
    } == {("x - 1", 2), ("x + 1", 1)}
    assert _terms_by_pole(result) == {
        ("x - 1", 1): Poly(Rational(-1, 4), x, domain="QQ"),
        ("x - 1", 2): Poly(Rational(1, 2), x, domain="QQ"),
        ("x + 1", 1): Poly(Rational(1, 4), x, domain="QQ"),
    }
    assert result.hermite_agreement


def test_decomposition_reconstructs_source_by_common_denominator() -> None:
    expression = 1 / ((x - 1) ** 2 * (x + 1))

    result = _decompose(expression)

    assert result.reconstructed == result.function
    replayed = rational_polynomial_to_sympy(result.polynomial_part)
    for term in result.terms:
        replayed += rational_polynomial_to_sympy(term.numerator) / (
            rational_polynomial_to_sympy(term.factor) ** term.exponent
        )
    assert cancel(replayed - expression) == 0


def test_improper_input_splits_polynomial_part() -> None:
    result = _decompose(x**3 / (x - 1))

    assert (
        rational_polynomial_to_sympy(result.polynomial_part).as_expr() == x**2 + x + 1
    )
    assert _terms_by_pole(result) == {("x - 1", 1): Poly(1, x, domain="QQ")}


def test_irreducible_quadratic_is_not_split_into_roots() -> None:
    result = _decompose(1 / ((x**2 + 1) * (x + 1)))

    assert {
        (str(rational_polynomial_to_sympy(factor.factor).as_expr()), factor.exponent)
        for factor in result.factors
    } == {("x**2 + 1", 1), ("x + 1", 1)}
    assert result.hermite_agreement


def test_polynomial_input_is_pure_polynomial_part() -> None:
    result = _decompose(2)

    assert rational_polynomial_to_sympy(result.polynomial_part).as_expr() == 2
    assert result.terms == ()
    assert result.factors == ()
    assert result.reconstructed == result.function


def test_serialized_claim_is_source_bound_and_verifiable() -> None:
    result = _decompose(1 / ((x - 1) ** 2 * (x + 1)))
    decoded = PartialFractionsResult.model_validate_json(result.model_dump_json())

    assert verify_partial_fractions(decoded)

    forged_terms = tuple(
        term.model_copy(update={"numerator": term.numerator}) for term in decoded.terms
    )
    swapped = (forged_terms[1], forged_terms[0], *forged_terms[2:])
    forged = decoded.model_copy(update={"terms": swapped})
    assert not verify_partial_fractions(forged)


def test_native_and_catalog_paths_agree() -> None:
    expression = 1 / ((x - 1) ** 2 * (x + 1))
    function = rational_function_from_sympy(expression, ("x",))
    request = PartialFractionsRequest(function=function)

    assert compute_partial_fractions(request) == partial_fractions(function)


def test_contradictory_rational_primitive_state_is_rejected() -> None:
    function = rational_function_from_sympy(1 / (x - 1), ("x",))
    with pytest.raises(ValidationError):
        RationalPrimitiveResult(
            source=function,
            status="RATIONAL_PRIMITIVE",
            rational_part=function,
            remainder=function,
        )
    payload = {
        "source": function.model_dump(mode="json"),
        "status": "RATIONAL_PRIMITIVE",
        "rational_part": function.model_dump(mode="json"),
        "remainder": function.model_dump(mode="json"),
    }
    with pytest.raises(ValidationError):
        RationalPrimitiveResult.model_validate_json(json.dumps(payload))


def test_declaration_is_published_with_one_executable_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "rational_function.partial_fractions.compute"
    )

    assert len(tool.examples) >= 1
    example_request = PartialFractionsRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(example_request)
    assert isinstance(result, PartialFractionsResult)
    assert result.reconstructed == result.function
