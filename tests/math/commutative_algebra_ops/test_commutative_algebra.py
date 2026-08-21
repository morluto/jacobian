"""Contract and mathematical tests for commutative algebra operations."""

from __future__ import annotations

import shutil
from fractions import Fraction
from itertools import combinations

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.math.commutative_algebra_ops import _singular
from jacobian.math.commutative_algebra_ops._models import (
    IdealComputationBudget,
    IdealQuotientRequest,
    IdealRadicalMembershipRequest,
    IdealRadicalRequest,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_ideal_quotient,
    compute_ideal_radical,
    compute_ideal_radical_membership,
)
from jacobian.math.commutative_algebra_ops._tools import TOOLS
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _ideal(
    variables: tuple[str, ...],
    *generators: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(_polynomial(variables, generator) for generator in generators),
    )


def _contains(ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial) -> bool:
    symbols = sympy.symbols(ideal.variables)
    basis = sympy.groebner(
        [rational_polynomial_to_sympy(item).as_expr() for item in ideal.generators],
        *symbols,
        domain=sympy.QQ,
    )
    return basis.reduce(rational_polynomial_to_sympy(polynomial).as_expr())[1] == 0


def _equal(left: RationalPolynomialIdeal, right: RationalPolynomialIdeal) -> bool:
    return all(_contains(left, generator) for generator in right.generators) and all(
        _contains(right, generator) for generator in left.generators
    )


def _contains_product(
    ideal: RationalPolynomialIdeal,
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> bool:
    symbols = sympy.symbols(ideal.variables)
    basis = sympy.groebner(
        [rational_polynomial_to_sympy(item).as_expr() for item in ideal.generators],
        *symbols,
        domain=sympy.QQ,
    )
    product = (
        rational_polynomial_to_sympy(left).as_expr()
        * rational_polynomial_to_sympy(right).as_expr()
    )
    return basis.reduce(product)[1] == 0


def _monomial_radical_oracle(
    variables: tuple[str, ...], generators: tuple[tuple[int, ...], ...]
) -> RationalPolynomialIdeal:
    """Compute monomial radicals combinatorially, independently of a CAS."""

    square_free = {
        tuple(1 if exponent else 0 for exponent in generator)
        for generator in generators
    }
    minimal = tuple(
        generator
        for generator in sorted(square_free)
        if not any(
            candidate != generator
            and all(
                left <= right for left, right in zip(candidate, generator, strict=True)
            )
            for candidate in square_free
        )
    )
    return _ideal(variables, *({generator: 1} for generator in minimal))


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.ideal.radical.compute",
        "polynomial.ideal.radical_membership.decide",
        "polynomial.ideal.quotient.compute",
    }


def test_ideal_contract_rejects_mixed_polynomial_rings() -> None:
    with pytest.raises(ValueError, match="declared ordered ring"):
        RationalPolynomialIdeal(
            variables=("x",),
            generators=(_polynomial(("y",), {(1,): 1}),),
        )


def test_backend_unavailability_is_a_typed_execution_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_singular.shutil, "which", lambda _name: None)
    result = compute_ideal_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(2,): 1})))
    assert result.outcome == "UNAVAILABLE"
    assert result.radical is None
    assert result.detail == "The supported Singular 4.4 backend is not installed."


def test_singular_protocol_parser_rejects_trailing_output() -> None:
    output = b"\n".join(
        [
            b"JACOBIAN_SINGULAR_IDEAL_V1",
            b"44100",
            b"1",
            b"GENERATOR",
            b"1|1",
            b"END_GENERATOR",
            b"END",
            b"unexpected",
        ]
    )
    with pytest.raises(ValueError, match="trailing data"):
        _singular._parse_output(
            output,
            variables=("x",),
            budget=IdealComputationBudget(),
        )


def test_singular_protocol_accepts_coefficients_beyond_input_budget() -> None:
    coefficient = "1" + "0" * 256

    assert _singular._parse_coefficient(coefficient).num == coefficient


def test_singular_protocol_classifies_unrepresentable_coefficients_as_a_limit() -> None:
    coefficient = "1" + "0" * 32_768

    with pytest.raises(ValueError, match="canonical exact-result digit limit"):
        _singular._parse_coefficient(coefficient)


def test_radical_membership_schema_publishes_operation_bounds() -> None:
    properties = IdealRadicalMembershipRequest.model_json_schema()["properties"]

    assert "at most 6 variables" in properties["ideal"]["description"]
    assert "total degree at most 12" in properties["polynomial"]["description"]


def test_singular_script_uses_internal_identifiers_not_caller_names() -> None:
    source = _singular._script(
        "radical",
        _ideal(("callerVariable",), {(2,): 1}),
        None,
    ).decode("ascii")
    assert "callerVariable" not in source
    assert "jv1" in source


def test_radical_membership_uses_canonical_polynomials() -> None:
    source = _ideal(("x", "y"), {(2, 0): 1}, {(1, 1): 1})
    assert compute_ideal_radical_membership(
        IdealRadicalMembershipRequest(
            ideal=source,
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
        )
    ).in_radical
    assert not compute_ideal_radical_membership(
        IdealRadicalMembershipRequest(
            ideal=source,
            polynomial=_polynomial(("x", "y"), {(0, 1): 1}),
        )
    ).in_radical


requires_singular = pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_radical_counterexample_is_exact() -> None:
    source = _ideal(
        ("x", "y"),
        {(2, 0): 1, (0, 1): -1},
        {(0, 2): 1},
    )
    result = compute_ideal_radical(IdealRadicalRequest(ideal=source))
    assert result.outcome == "COMPUTED"
    assert result.radical is not None
    assert _equal(
        result.radical,
        _ideal(("x", "y"), {(1, 0): 1}, {(0, 1): 1}),
    )
    assert all(_contains(result.radical, generator) for generator in source.generators)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_radical_is_idempotent() -> None:
    first = compute_ideal_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(4,): 1})))
    assert first.outcome == "COMPUTED"
    assert first.radical is not None
    second = compute_ideal_radical(IdealRadicalRequest(ideal=first.radical))
    assert second.outcome == "COMPUTED"
    assert second.radical is not None
    assert _equal(first.radical, second.radical)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_radical_is_invariant_under_equivalent_generators() -> None:
    first = compute_ideal_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(2,): 1})))
    second = compute_ideal_radical(
        IdealRadicalRequest(ideal=_ideal(("x",), {(2,): 1}, {(2,): 2}))
    )

    assert first.radical is not None
    assert second.radical is not None
    assert _equal(first.radical, second.radical)


@requires_singular
@pytest.mark.requires_backend("singular")
@pytest.mark.parametrize("generator", [{}, {(0,): 1}])
def test_ideal_radical_handles_zero_and_unit_ideals(
    generator: dict[tuple[int, ...], int],
) -> None:
    source = _ideal(("x",), generator)
    result = compute_ideal_radical(IdealRadicalRequest(ideal=source))

    assert result.radical is not None
    assert _equal(result.radical, source)


@requires_singular
@pytest.mark.requires_backend("singular")
@pytest.mark.parametrize(
    "generators",
    tuple(
        combination
        for size in range(1, 4)
        for combination in combinations(((2, 0), (0, 3), (2, 2), (3, 1), (1, 3)), size)
    ),
)
def test_ideal_radical_agrees_with_independent_monomial_oracle(
    generators: tuple[tuple[int, int], ...],
) -> None:
    source = _ideal(("x", "y"), *({generator: 1} for generator in generators))
    result = compute_ideal_radical(IdealRadicalRequest(ideal=source))

    assert result.outcome == "COMPUTED"
    assert result.radical is not None
    assert _equal(
        result.radical,
        _monomial_radical_oracle(("x", "y"), generators),
    )


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_counterexample_is_exact() -> None:
    result = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=_ideal(("x", "y"), {(1, 0): 1}),
            divisor=_ideal(("x", "y"), {(2, 0): 1, (0, 1): 1}),
        )
    )
    assert result.outcome == "COMPUTED"
    assert result.quotient is not None
    assert _equal(result.quotient, _ideal(("x", "y"), {(1, 0): 1}))
    dividend = _ideal(("x", "y"), {(1, 0): 1})
    divisor = _ideal(("x", "y"), {(2, 0): 1, (0, 1): 1})
    assert all(
        _contains_product(dividend, quotient_generator, divisor_generator)
        for quotient_generator in result.quotient.generators
        for divisor_generator in divisor.generators
    )


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_is_invariant_under_equivalent_divisor_generators() -> None:
    dividend = _ideal(("x",), {(2,): 1})
    first = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(("x",), {(1,): 1}),
        )
    )
    second = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(("x",), {(1,): 1}, {(1,): 2}),
        )
    )

    assert first.quotient is not None
    assert second.quotient is not None
    assert _equal(first.quotient, second.quotient)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_by_zero_is_the_unit_ideal() -> None:
    result = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=_ideal(("x",), {(1,): 1}),
            divisor=_ideal(("x",), {}),
        )
    )
    assert result.outcome == "COMPUTED"
    assert result.quotient is not None
    assert _equal(result.quotient, _ideal(("x",), {(0,): 1}))


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_by_unit_is_the_dividend() -> None:
    dividend = _ideal(("x", "y"), {(2, 0): 1}, {(1, 1): 1})
    result = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(("x", "y"), {(0, 0): 1}),
        )
    )

    assert result.outcome == "COMPUTED"
    assert result.quotient is not None
    assert _equal(result.quotient, dividend)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_singular_codec_round_trips_a_fractional_multivariate_ideal() -> None:
    source = _ideal(
        ("x", "y"),
        {(2, 0): 1, (0, 1): Fraction(1, 2)},
        {(0, 2): 1, (0, 0): Fraction(-2, 3)},
    )

    result = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=source,
            divisor=_ideal(("x", "y"), {(0, 0): 1}),
        )
    )

    assert result.outcome == "COMPUTED"
    assert result.quotient == _ideal(
        ("x", "y"),
        {(0, 2): 3, (0, 0): -2},
        {(2, 0): 2, (0, 1): 1},
    )
    assert _equal(result.quotient, source)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_by_generator_sum_is_intersection_of_colons() -> None:
    # (I : <x,y>) = (I : <x>) intersect (I : <y>) = <x>
    # for I = <x^2, xy>.
    variables = ("x", "y")
    dividend = _ideal(variables, {(2, 0): 1}, {(1, 1): 1})
    result = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}, {(0, 1): 1}),
        )
    )
    by_x = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}),
        )
    )
    by_y = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(0, 1): 1}),
        )
    )

    assert result.outcome == "COMPUTED"
    assert result.quotient is not None
    assert by_x.quotient is not None
    assert by_y.quotient is not None
    assert _equal(by_x.quotient, _ideal(variables, {(1, 0): 1}, {(0, 1): 1}))
    assert _equal(by_y.quotient, _ideal(variables, {(1, 0): 1}))
    assert _equal(result.quotient, _ideal(variables, {(1, 0): 1}))


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_by_product_equals_iterated_quotient() -> None:
    # (I : JK) = ((I : J) : K).
    variables = ("x", "y")
    dividend = _ideal(variables, {(3, 2): 1})
    by_product = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 1): 1}),
        )
    )
    first = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}),
        )
    )
    assert first.quotient is not None
    iterated = compute_ideal_quotient(
        IdealQuotientRequest(
            dividend=first.quotient,
            divisor=_ideal(variables, {(0, 1): 1}),
        )
    )

    assert by_product.quotient is not None
    assert iterated.quotient is not None
    assert _equal(by_product.quotient, iterated.quotient)
    assert _equal(by_product.quotient, _ideal(variables, {(2, 1): 1}))
