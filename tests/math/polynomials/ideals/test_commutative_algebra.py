"""Contract and mathematical tests for commutative algebra operations."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.ideals import _singular
from jacobian.math.polynomials.ideals._models import (
    EliminationIdealResult,
    IdealComputationBudget,
    IdealQuotientRequest,
    IdealRadicalMembershipRequest,
    IdealRadicalRequest,
)
from jacobian.math.polynomials.ideals._tools import TOOLS
from jacobian.math.polynomials.ideals.operations import (
    ideal_quotient,
    ideal_radical,
    ideal_radical_membership,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _run_radical(request: IdealRadicalRequest):
    return ideal_radical(request.ideal, resource_budget=request.resource_budget)


def _run_radical_membership(request: IdealRadicalMembershipRequest):
    return ideal_radical_membership(request.ideal, request.polynomial)


def _run_quotient(request: IdealQuotientRequest):
    return ideal_quotient(
        request.dividend, request.divisor, resource_budget=request.resource_budget
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: Mapping[tuple[int, ...], int | Fraction],
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
    *generators: Mapping[tuple[int, ...], int | Fraction],
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
    return bool(
        basis.reduce(rational_polynomial_to_sympy(polynomial).as_expr())[1] == 0
    )


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
    return bool(basis.reduce(product)[1] == 0)


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
        "polynomial.ideal.saturation.compute",
        "polynomial.ideal.groebner_basis.compute",
        "polynomial.ideal.minimal_primes.compute",
        "polynomial.ideal.normal_form.compute",
        "polynomial.ideal.elimination.compute",
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
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = _run_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(2,): 1})))
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


def test_singular_protocol_decodes_coefficients_beyond_the_python_digit_cap() -> None:
    """Chunked canonical parsing survives CPython's 4300-digit int() limit."""

    coefficient = "1" + "0" * 5_000

    assert _singular._parse_coefficient(coefficient).num == coefficient
    assert _singular._parse_coefficient(f"-{coefficient}/3").as_fraction() == (
        Fraction(-(10**5000), 3)
    )


def test_singular_protocol_classifies_unrepresentable_coefficients_as_a_limit() -> None:
    coefficient = "1" + "0" * 32_768

    with pytest.raises(ValueError, match="canonical exact-result digit limit"):
        _singular._parse_coefficient(coefficient)


def test_singular_protocol_classifies_oversized_exponents_as_a_limit() -> None:
    output = b"\n".join(
        [
            b"JACOBIAN_SINGULAR_IDEAL_V1",
            b"44100",
            b"1",
            b"GENERATOR",
            b"1|32769",
            b"END_GENERATOR",
            b"END",
        ]
    )

    with pytest.raises(
        _singular._ResultLimitExceededError, match="representation limit"
    ):
        _singular._parse_output(
            output,
            variables=("x",),
            budget=IdealComputationBudget(),
        )
    component_output = b"\n".join(
        [
            b"JACOBIAN_SINGULAR_IDEAL_V1",
            b"44100",
            b"1",
            b"COMPONENT",
            b"1",
            b"GENERATOR",
            b"1|32769",
            b"END_GENERATOR",
            b"END_COMPONENT",
            b"END",
        ]
    )

    with pytest.raises(
        _singular._ResultLimitExceededError, match="representation limit"
    ):
        _singular._parse_minimal_primes_output(
            component_output,
            variables=("x",),
            budget=IdealComputationBudget(),
        )


def test_singular_protocol_classifies_oversized_generators_as_a_limit() -> None:
    terms = [f"1|{exponent}".encode("ascii") for exponent in range(4_096, -1, -1)]
    output = b"\n".join(
        [
            b"JACOBIAN_SINGULAR_IDEAL_V1",
            b"44100",
            b"1",
            b"GENERATOR",
            *terms,
            b"END_GENERATOR",
            b"END",
        ]
    )

    with pytest.raises(
        _singular._ResultLimitExceededError, match="term representation"
    ):
        _singular._parse_output(
            output,
            variables=("x",),
            budget=IdealComputationBudget(),
        )
    component_output = b"\n".join(
        [
            b"JACOBIAN_SINGULAR_IDEAL_V1",
            b"44100",
            b"1",
            b"COMPONENT",
            b"1",
            b"GENERATOR",
            *terms,
            b"END_GENERATOR",
            b"END_COMPONENT",
            b"END",
        ]
    )

    with pytest.raises(
        _singular._ResultLimitExceededError, match="term representation"
    ):
        _singular._parse_minimal_primes_output(
            component_output,
            variables=("x",),
            budget=IdealComputationBudget(),
        )


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
    assert _run_radical_membership(
        IdealRadicalMembershipRequest(
            ideal=source,
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
        )
    ).in_radical
    assert not _run_radical_membership(
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
    result = _run_radical(IdealRadicalRequest(ideal=source))
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
    first = _run_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(4,): 1})))
    assert first.outcome == "COMPUTED"
    assert first.radical is not None
    second = _run_radical(IdealRadicalRequest(ideal=first.radical))
    assert second.outcome == "COMPUTED"
    assert second.radical is not None
    assert _equal(first.radical, second.radical)


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_radical_is_invariant_under_equivalent_generators() -> None:
    first = _run_radical(IdealRadicalRequest(ideal=_ideal(("x",), {(2,): 1})))
    second = _run_radical(
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
    result = _run_radical(IdealRadicalRequest(ideal=source))

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
    monomial_generators: list[dict[tuple[int, ...], int]] = [
        {generator: 1} for generator in generators
    ]
    source = _ideal(("x", "y"), *monomial_generators)
    result = _run_radical(IdealRadicalRequest(ideal=source))

    assert result.outcome == "COMPUTED"
    assert result.radical is not None
    assert _equal(
        result.radical,
        _monomial_radical_oracle(("x", "y"), generators),
    )


@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_counterexample_is_exact() -> None:
    result = _run_quotient(
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
    first = _run_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(("x",), {(1,): 1}),
        )
    )
    second = _run_quotient(
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
    result = _run_quotient(
        IdealQuotientRequest(
            dividend=_ideal(("x",), {(1,): 1}),
            divisor=_ideal(("x",), {}),
        )
    )
    assert result.outcome == "COMPUTED"
    assert result.quotient is not None
    assert _equal(result.quotient, _ideal(("x",), {(0,): 1}))


def test_zero_basis_must_use_the_source_ideal_ring() -> None:
    from jacobian.math.polynomials.ideals._models import (
        GroebnerBasisRequest,
        GroebnerBasisResult,
    )

    request = GroebnerBasisRequest(ideal=_ideal(("x",), {}))
    foreign_zero_basis = _ideal(("y",), {})
    with pytest.raises(ValueError, match="source ideal's ordered ring"):
        GroebnerBasisResult(
            ideal=request.ideal,
            basis=foreign_zero_basis,
            generator_count=1,
            monomial_order="grevlex",
        )
    same_ring_zero_basis = _ideal(("x",), {})
    result = GroebnerBasisResult(
        ideal=request.ideal,
        basis=same_ring_zero_basis,
        generator_count=1,
        monomial_order="grevlex",
    )
    assert result.basis is not None


@requires_singular
@pytest.mark.requires_backend("singular")
@requires_singular
@pytest.mark.requires_backend("singular")
@requires_singular
@pytest.mark.requires_backend("singular")
@requires_singular
@pytest.mark.requires_backend("singular")
def test_ideal_quotient_by_unit_is_the_dividend() -> None:
    dividend = _ideal(("x", "y"), {(2, 0): 1}, {(1, 1): 1})
    result = _run_quotient(
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

    result = _run_quotient(
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
    result = _run_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}, {(0, 1): 1}),
        )
    )
    by_x = _run_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}),
        )
    )
    by_y = _run_quotient(
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
    by_product = _run_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 1): 1}),
        )
    )
    first = _run_quotient(
        IdealQuotientRequest(
            dividend=dividend,
            divisor=_ideal(variables, {(1, 0): 1}),
        )
    )
    assert first.quotient is not None
    iterated = _run_quotient(
        IdealQuotientRequest(
            dividend=first.quotient,
            divisor=_ideal(variables, {(0, 1): 1}),
        )
    )

    assert by_product.quotient is not None
    assert iterated.quotient is not None
    assert _equal(by_product.quotient, iterated.quotient)
    assert _equal(by_product.quotient, _ideal(variables, {(2, 1): 1}))


def _poly(
    variables: tuple[str, ...],
    terms: Sequence[tuple[tuple[int, ...], int | Fraction]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(c)),
                    exponents=e,
                )
                for e, c in terms
            )
        ),
    )


class TestIdealMembershipViaGroebnerBasis:
    def test_membership_needs_groebner_reduction(self) -> None:
        """x-y lies in <xy-1, y^2-1> though neither generator divides it."""
        from jacobian.math.polynomials.ideals.operations import ideal_normal_form

        ring = ("x", "y")
        ideal = RationalPolynomialIdeal(
            variables=ring,
            generators=(
                _poly(ring, [((1, 1), 1), ((0, 0), -1)]),
                _poly(ring, [((0, 2), 1), ((0, 0), -1)]),
            ),
        )
        result = ideal_normal_form(
            ideal,
            _poly(ring, [((1, 0), 1), ((0, 1), -1)]),
            "grevlex",
        )
        assert result.in_ideal is True
        assert result.remainder is not None
        assert len(result.remainder.polynomial.terms) == 0


class TestEliminationIdealSemantics:
    def _eliminate(
        self,
        generators: Sequence[Sequence[tuple[tuple[int, ...], int | Fraction]]],
        eliminated: tuple[str, ...],
    ) -> EliminationIdealResult:
        from jacobian.math.polynomials.ideals._models import IdealComputationBudget
        from jacobian.math.polynomials.ideals.operations import elimination_ideal

        ring = ("x", "y")
        ideal = RationalPolynomialIdeal(
            variables=ring,
            generators=tuple(_poly(ring, g) for g in generators),
        )
        return elimination_ideal(
            ideal, eliminated, resource_budget=IdealComputationBudget()
        )

    @staticmethod
    def _terms(
        result: EliminationIdealResult,
    ) -> list[tuple[str, str, tuple[int, ...]]]:
        assert result.elimination_ideal is not None
        return [
            (str(t.coefficient.num), str(t.coefficient.den), t.exponents)
            for g in result.elimination_ideal.generators
            for t in g.polynomial.terms
        ]

    def test_eliminated_variables_lead_lex_order(self) -> None:
        """Eliminating y from <x-y, y^2-1> yields <x^2-1>."""
        result = self._eliminate(
            [(((1, 0), 1), ((0, 1), -1)), (((0, 2), 1), ((0, 0), -1))],
            ("y",),
        )
        terms = self._terms(result)
        assert (("1", "1", (2,))) in [tuple(t) for t in terms]
        assert tuple((n, d, e) for n, d, e in terms)[:2] == (
            ("1", "1", (2,)),
            ("-1", "1", (0,)),
        )

    def test_zero_elimination_ideal_preserved(self) -> None:
        """<x> ∩ QQ[y] = (0): the zero ideal must not become the whole ring."""
        result = self._eliminate([(((1, 0), 1),)], ("x",))
        assert result.elimination_ideal is not None
        assert len(result.elimination_ideal.generators) == 1
        assert len(result.elimination_ideal.generators[0].polynomial.terms) == 0

    def test_unit_elimination_ideal(self) -> None:
        """An ideal containing 1 eliminates to the whole ring."""
        result = self._eliminate([(((0, 0), 1),)], ("y",))
        terms = self._terms(result)
        assert terms == [("1", "1", (0,))]
