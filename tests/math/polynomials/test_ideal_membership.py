"""Contract and mathematical tests for ideal membership and normal form."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._models import (
    IdealMembershipRequest,
    IdealMembershipResult,
    IdealNormalFormResult,
)
from jacobian.math.polynomials._operations import (
    polynomial_ideal_membership,
    polynomial_ideal_normal_form,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
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


_XY_VARS = ("x", "y")


def _ideal1() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=("x",),
        generators=(_poly(("x",), {(2,): 1}),),
    )


def _ideal_xy() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(_poly(_XY_VARS, {(1, 0): 1, (0, 1): 1}),),
    )


def _ideal_line_through_origin() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(
            _poly(_XY_VARS, {(1, 0): 1}),
            _poly(_XY_VARS, {(0, 1): 1, (1, 0): -1}),
        ),
    )


def _ideal_coordinate_axes() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(
            _poly(_XY_VARS, {(1, 0): 1}),
            _poly(_XY_VARS, {(0, 1): 1}),
        ),
    )


def test_operations_in_catalog() -> None:
    ids = {tool.operation_id for tool in POLYNOMIAL_INVARIANT_OPERATIONS}
    assert "polynomial.ideal.membership.decide" in ids
    assert "polynomial.ideal.normal_form.compute" in ids


def test_member_is_in_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(2,): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.status == "IN_IDEAL"
    assert result.normal_form is not None
    assert len(result.normal_form.polynomial.terms) == 0


def test_non_member_is_not_in_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(1,): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.status == "NOT_IN_IDEAL"
    assert result.normal_form is not None
    assert len(result.normal_form.polynomial.terms) == 1


def test_zero_is_in_any_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal_xy(),
        polynomial=_poly(("x", "y"), {}),
    )
    result = polynomial_ideal_membership(req)
    assert result.status == "IN_IDEAL"


def test_normal_form_returns_canonical_remainder() -> None:
    # x + 1 mod <x^2> = x + 1
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(1,): 1, (0,): 1}),
    )
    result = polynomial_ideal_normal_form(req)
    assert result.monomial_order == "grevlex"
    assert len(result.remainder.polynomial.terms) == 2
    assert result.remainder.polynomial.terms[0].exponents == (1,)


def test_membership_rejects_mismatched_rings() -> None:
    with pytest.raises(ValueError, match="same ordered ring"):
        IdealMembershipRequest(
            ideal=_ideal1(),
            polynomial=_poly(("y",), {(1,): 1}),
        )


def test_multivariate_membership() -> None:
    # Is y in <x, y-x> = <x, y>?
    req = IdealMembershipRequest(
        ideal=_ideal_line_through_origin(),
        polynomial=_poly(("x", "y"), {(0, 1): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.status == "IN_IDEAL"


def test_multivariate_non_membership() -> None:
    # Is 1 in <x, y>? No.
    req = IdealMembershipRequest(
        ideal=_ideal_coordinate_axes(),
        polynomial=_poly(("x", "y"), {(0, 0): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.status == "NOT_IN_IDEAL"


def _expanded_ideal(order: str = "lex") -> IdealMembershipRequest:
    """Ideal <x - (1+y+z)^4> with polynomial x^12 under lex order.

    The normal form is (1+y+z)^48, whose 1,226 terms exceed the 1,024-term
    exact-result boundary even though every admission budget is respected.
    """

    import sympy

    y_sym, z_sym = sympy.symbols("y z")
    expansion = sympy.Poly((1 + y_sym + z_sym) ** 4, y_sym, z_sym, domain=sympy.QQ)
    generator_terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(int(c))),
            exponents=(0, ey, ez),
        )
        for (ey, ez), c in expansion.terms()
    )
    ideal = RationalPolynomialIdeal(
        variables=("x", "y", "z"),
        generators=(
            RationalPolynomial(
                variables=("x", "y", "z"),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational.from_fraction(Fraction(1)),
                            exponents=(1, 0, 0),
                        ),
                        *generator_terms,
                    )
                ),
            ),
        ),
    )
    return IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(("x", "y", "z"), {(12, 0, 0): 1}),
        monomial_order=order,
    )


def test_expansion_beyond_result_boundary_reports_typed_budget_outcome() -> None:
    request = _expanded_ideal()
    result = polynomial_ideal_normal_form(request)
    assert result.status == "BUDGET_EXCEEDED"
    assert result.remainder is None
    membership = polynomial_ideal_membership(request)
    assert membership.status == "BUDGET_EXCEEDED"
    assert membership.normal_form is None


def test_budget_exceeded_with_retained_basis_validates() -> None:
    result = polynomial_ideal_normal_form(_expanded_ideal())
    assert result.groebner_basis is not None
    IdealNormalFormResult.model_validate(
        {
            "ideal": result.ideal,
            "polynomial": result.polynomial,
            "monomial_order": result.monomial_order,
            "status": "BUDGET_EXCEEDED",
            "groebner_basis": result.groebner_basis,
            "remainder": None,
        }
    )


def test_coefficient_growth_beyond_canonical_rational_reports_typed_budget() -> None:
    # Reducing x1^12 modulo <x1-C*x2^12, x2-C*x3^12, x3-C*x4^12> with
    # C=10**127 yields the single term C^1884*x4^20736: its term count and
    # exponent respect the output budgets while its 239,269-digit coefficient
    # leaves the canonical rational domain.  Both operations must report the
    # typed budget outcome instead of failing result validation.
    variables = ("x1", "x2", "x3", "x4")
    coefficient = 10**127

    def chain_generator(shift: int) -> RationalPolynomial:
        return _poly(
            variables,
            {
                tuple(12 if index == shift else 0 for index in range(4)): coefficient,
                tuple(1 if index == shift - 1 else 0 for index in range(4)): -1,
            },
        )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(chain_generator(shift) for shift in (1, 2, 3)),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(12, 0, 0, 0): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "BUDGET_EXCEEDED"
    assert result.remainder is None
    membership = polynomial_ideal_membership(request)
    assert membership.status == "BUDGET_EXCEEDED"
    assert membership.normal_form is None


def test_generator_count_and_total_degree_budgets_enforced() -> None:
    generators = tuple(_poly(("x", "y"), {(1, 0): 1}) for _ in range(17))
    with pytest.raises(ValueError, match="16-generator"):
        IdealMembershipRequest(
            ideal=RationalPolynomialIdeal(variables=_XY_VARS, generators=generators),
            polynomial=_poly(_XY_VARS, {(1, 0): 1}),
        )
    with pytest.raises(ValueError, match="total-degree"):
        IdealMembershipRequest(
            ideal=RationalPolynomialIdeal(
                variables=_XY_VARS,
                generators=(_poly(_XY_VARS, {(12, 12): 1}),),
            ),
            polynomial=_poly(_XY_VARS, {(1, 0): 1}),
        )


def test_detached_contradictory_membership_rejected() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(2,): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.normal_form is not None and result.groebner_basis is not None
    with pytest.raises(ValidationError):
        IdealMembershipResult(
            ideal=result.ideal,
            polynomial=result.polynomial,
            monomial_order=result.monomial_order,
            status="NOT_IN_IDEAL",
            groebner_basis=result.groebner_basis,
            normal_form=result.normal_form,
        )


def test_tampered_normal_form_does_not_validate() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(3,): 1}),
    )
    result = polynomial_ideal_normal_form(req)
    assert result.remainder is not None and result.groebner_basis is not None
    tampered = _poly(("x",), {(5,): 1})
    with pytest.raises(ValidationError):
        IdealNormalFormResult(
            ideal=result.ideal,
            polynomial=result.polynomial,
            monomial_order=result.monomial_order,
            status="COMPUTED",
            groebner_basis=result.groebner_basis,
            remainder=tampered,
        )


def test_computed_result_without_source_context_rejected() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal_xy(),
        polynomial=_poly(_XY_VARS, {(2, 0): 1, (0, 2): 1}),
    )
    result = polynomial_ideal_normal_form(req)
    assert result.remainder is not None
    with pytest.raises(ValidationError):
        IdealNormalFormResult(
            ideal=RationalPolynomialIdeal(
                variables=_XY_VARS,
                generators=(_poly(_XY_VARS, {(1, 0): 1, (0, 1): 2}),),
            ),
            polynomial=result.polynomial,
            monomial_order=result.monomial_order,
            status="COMPUTED",
            groebner_basis=result.groebner_basis,
            remainder=result.remainder,
        )


def test_expansion_beyond_intermediate_budget_reports_typed_outcome_quickly() -> None:
    # <x - (1+y1+y2+y3+y4+y5)^5> with polynomial x^12: the exact remainder
    # would expand to millions of monomials.  The bounded reduction must
    # report the typed budget outcome instead of materializing it.
    variables = ("x", "y1", "y2", "y3", "y4", "y5")

    def term(exponents: tuple[int, ...]) -> RationalPolynomialTerm:
        return RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            exponents=exponents,
        )

    # Every monomial of degree at most 5 in y1..y5: exactly
    # (1 + y1 + y2 + y3 + y4 + y5)^5, whose 252 terms plus the leading
    # x term give the reviewed 253-term generator.
    expansion_combos = [
        combo
        for combo in __import__("itertools").product(range(6), repeat=5)
        if sum(combo) <= 5
    ]
    assert len(expansion_combos) == 252
    generator_terms = [term((1, 0, 0, 0, 0, 0))] + [
        term((0, *combo)) for combo in expansion_combos
    ]
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=tuple(
                        sorted(generator_terms, key=lambda t: t.exponents, reverse=True)
                    )
                ),
            ),
        ),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(12, 0, 0, 0, 0, 0): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "BUDGET_EXCEEDED"
    membership = polynomial_ideal_membership(request)
    assert membership.status == "BUDGET_EXCEEDED"


def test_unsubstantiated_budget_outcome_without_basis_rejected() -> None:
    # <x^2>, polynomial x: the basis and remainder are in budget, so an
    # authored BUDGET_EXCEEDED with both outputs stripped asserts an
    # arbitrary execution outcome without evidence and must be rejected.
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(1,): 1}),
    )
    with pytest.raises(ValidationError, match="source basis to exceed"):
        IdealMembershipResult(
            ideal=req.ideal,
            polynomial=req.polynomial,
            monomial_order="grevlex",
            status="BUDGET_EXCEEDED",
            groebner_basis=None,
            normal_form=None,
        )


def test_basis_growth_beyond_representability_reports_non_conclusion() -> None:
    # The eight-variable lex chain <x1-C*x2^12, ..., x7-C*x8^12> with
    # C = 10**127 has a reduced basis element whose coefficient would carry
    # hundreds of millions of digits.  No prefix may decide that outcome
    # (later generators can shrink an ideal), and every complete strategy
    # exceeds its work bound here, so the operations report the typed
    # non-conclusion outcome instead of asserting an overflow they cannot
    # establish.
    variables = tuple(f"x{i}" for i in range(1, 9))
    coefficient = 10**127

    def chain_generator(shift: int) -> RationalPolynomial:
        return _poly(
            variables,
            {
                tuple(
                    12 if index == shift + 1 else 0 for index in range(8)
                ): coefficient,
                tuple(1 if index == shift else 0 for index in range(8)): -1,
            },
        )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(chain_generator(shift) for shift in range(7)),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(1,) + (0,) * 7: 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "UNKNOWN"
    assert result.groebner_basis is None and result.remainder is None
    membership = polynomial_ideal_membership(request)
    assert membership.status == "UNKNOWN"
    IdealNormalFormResult.model_validate(
        {
            "ideal": result.ideal,
            "polynomial": result.polynomial,
            "monomial_order": result.monomial_order,
            "status": "UNKNOWN",
            "groebner_basis": None,
            "remainder": None,
        }
    )


def test_unknown_result_must_not_carry_partial_artifacts() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(3,): 1}),
    )
    computed = polynomial_ideal_normal_form(req)
    assert computed.groebner_basis is not None and computed.remainder is not None
    with pytest.raises(ValidationError, match="UNKNOWN must not carry"):
        IdealNormalFormResult(
            ideal=req.ideal,
            polynomial=req.polynomial,
            monomial_order="grevlex",
            status="UNKNOWN",
            groebner_basis=computed.groebner_basis,
            remainder=None,
        )
    with pytest.raises(ValidationError, match="UNKNOWN must not carry"):
        IdealMembershipResult(
            ideal=req.ideal,
            polynomial=req.polynomial,
            monomial_order="grevlex",
            status="UNKNOWN",
            groebner_basis=None,
            normal_form=_poly(("x",), {}),
        )


def test_request_schema_publishes_polynomial_admission_limits() -> None:
    schema = IdealMembershipRequest.model_json_schema()
    description = schema["properties"]["polynomial"]["description"]
    assert "degree at most 12" in description
    assert "128" in description
    assert "1,024" in description


def test_queried_polynomial_total_degree_budget_enforced() -> None:
    # x^12*y^12 keeps every exponent at 12 while its total degree is 24;
    # the advertised work domain bounds total degree, so it is rejected.
    with pytest.raises(ValidationError, match="total-degree"):
        IdealMembershipRequest(
            ideal=_ideal_xy(),
            polynomial=_poly(("x", "y"), {(12, 12): 1}),
        )


def test_later_generator_collapse_not_reported_as_budget_overflow() -> None:
    # <x-(1+y+z)^4, x^12, 1>: an oversized prefix basis must not decide the
    # outcome; the trailing nonzero constant generator collapses the ideal
    # to the unit ideal regardless of presentation order, so x^12 is in the
    # ideal and no budget overflow is reported.
    from itertools import product

    variables = ("x", "y", "z")

    def term(exponents: tuple[int, ...]) -> RationalPolynomialTerm:
        return RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            exponents=exponents,
        )

    def poly(terms: tuple[tuple[int, ...], ...]):
        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    sorted(
                        (term(e) for e in terms),
                        key=lambda t: t.exponents,
                        reverse=True,
                    )
                )
            ),
        )

    combos = [c for c in product(range(5), repeat=2) if sum(c) <= 4]
    g1_terms: tuple[tuple[int, ...], ...] = (
        (1, 0, 0),
        *((0, c[0], c[1]) for c in combos),
    )
    assert len(g1_terms) == 16
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            poly(g1_terms),
            poly(((12, 0, 0),)),
            poly(((0, 0, 0),)),
        ),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=poly(((12, 0, 0),)),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "COMPUTED"
    assert result.groebner_basis is not None
    basis_terms = result.groebner_basis[0].polynomial.terms
    assert len(basis_terms) == 1
    assert basis_terms[0].coefficient.as_fraction() == 1


def _collapse_ideal_generators() -> tuple[RationalPolynomial, ...]:
    # Generators <x+f, x^12, x> where f is the sum of all monomials of
    # degree at most 4 in y,z.  The two-generator prefix <x+f, x^12> has
    # an oversized reduced basis containing f^12 (1,225 terms), while the
    # complete ideal is <x, f> with a 16-term reduced basis.
    from itertools import product

    variables = ("x", "y", "z")

    def term(exponents: tuple[int, ...]) -> RationalPolynomialTerm:
        return RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            exponents=exponents,
        )

    def poly(terms: tuple[tuple[int, ...], ...]):
        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    sorted(
                        (term(e) for e in terms),
                        key=lambda t: t.exponents,
                        reverse=True,
                    )
                )
            ),
        )

    combos = [c for c in product(range(5), repeat=2) if sum(c) <= 4]
    assert len(combos) == 15
    return (
        poly(((1, 0, 0), *((0, cy, cz) for cy, cz in combos))),
        poly(((12, 0, 0),)),
        poly(((1, 0, 0),)),
    )


def test_nonconstant_late_generator_collapse_not_reported_as_overflow() -> None:
    # An oversized prefix basis must never decide the outcome: processing
    # the trailing generator x collapses <x+f, x^12> to the small ideal
    # <x, f>, so x^12 has normal form zero and no budget overflow exists.
    request = IdealMembershipRequest(
        ideal=RationalPolynomialIdeal(
            variables=("x", "y", "z"),
            generators=_collapse_ideal_generators(),
        ),
        polynomial=_poly(("x", "y", "z"), {(12, 0, 0): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "COMPUTED"
    assert result.remainder is not None and len(result.remainder.polynomial.terms) == 0
    membership = polynomial_ideal_membership(request)
    assert membership.status == "IN_IDEAL"


def test_prefix_exponent_overflow_not_reported_for_late_unit_collapse() -> None:
    # Reviewer counterexample: <x1^11, x1-x2^11, ..., x4-x5^11, 1+x1^12>
    # under lex.  The five degree-11 generators come first in the canonical
    # order and their prefix basis contains x5^161051, whose exponent
    # leaves the shared representation limit; the trailing generator is
    # coprime to x1^11, so the complete ideal is the unit ideal with basis
    # (1) and x5 has normal form zero.  No prefix property may decide a
    # budget outcome that later generators invalidate.
    variables = tuple(f"x{i}" for i in range(1, 6))

    def chain_generator(shift: int) -> RationalPolynomial:
        return _poly(
            variables,
            {
                tuple(1 if index == shift else 0 for index in range(5)): 1,
                tuple(11 if index == shift + 1 else 0 for index in range(5)): -1,
            },
        )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            _poly(variables, {(11, 0, 0, 0, 0): 1}),
            *(chain_generator(shift) for shift in range(4)),
            _poly(variables, {(12, 0, 0, 0, 0): 1, (0, 0, 0, 0, 0): 1}),
        ),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(0, 0, 0, 0, 1): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "COMPUTED"
    assert result.groebner_basis is not None
    assert len(result.groebner_basis) == 1
    assert result.groebner_basis[0].polynomial.terms[0].exponents == (0,) * 5
    assert result.remainder is not None
    assert len(result.remainder.polynomial.terms) == 0
    membership = polynomial_ideal_membership(request)
    assert membership.status == "IN_IDEAL"


def test_complete_basis_exponent_overflow_reports_typed_outcome() -> None:
    # The same chain without the trailing coprime generator: no later
    # generator shrinks this ideal, and its complete reduced basis contains
    # x5^161051, whose exponent genuinely leaves the representability limit.
    variables = tuple(f"x{i}" for i in range(1, 6))

    def chain_generator(shift: int) -> RationalPolynomial:
        return _poly(
            variables,
            {
                tuple(1 if index == shift else 0 for index in range(5)): 1,
                tuple(11 if index == shift + 1 else 0 for index in range(5)): -1,
            },
        )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            _poly(variables, {(11, 0, 0, 0, 0): 1}),
            *(chain_generator(shift) for shift in range(4)),
        ),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(0, 0, 0, 0, 1): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "BUDGET_EXCEEDED"
    assert result.groebner_basis is None and result.remainder is None
    membership = polynomial_ideal_membership(request)
    assert membership.status == "BUDGET_EXCEEDED"
    IdealNormalFormResult.model_validate(
        {
            "ideal": result.ideal,
            "polynomial": result.polynomial,
            "monomial_order": result.monomial_order,
            "status": "BUDGET_EXCEEDED",
            "groebner_basis": None,
            "remainder": None,
        }
    )


def test_budget_outcome_is_independent_of_generator_presentation_order() -> None:
    # The bounded kernel must be a function of the ideal value, not of the
    # presentation order of an equivalent generating set.
    from itertools import permutations

    generators = _collapse_ideal_generators()
    polynomial = _poly(("x", "y", "z"), {(12, 0, 0): 1})
    statuses = set()
    for permutation in permutations(generators):
        membership = polynomial_ideal_membership(
            IdealMembershipRequest(
                ideal=RationalPolynomialIdeal(
                    variables=("x", "y", "z"), generators=permutation
                ),
                polynomial=polynomial,
                monomial_order="lex",
            )
        )
        statuses.add(membership.status)
    assert statuses == {"IN_IDEAL"}


def test_complete_basis_output_overflow_reports_typed_outcome() -> None:
    # <x-(1+y+z)^4, x^12>: no later generator shrinks this ideal, and its
    # complete reduced basis contains (1+y+z)^48 (1,225 aggregate terms),
    # genuinely beyond the 1,024-term output boundary.
    expanded = _expanded_ideal()
    padded = RationalPolynomialIdeal(
        variables=expanded.ideal.variables,
        generators=(
            *expanded.ideal.generators,
            _poly(expanded.ideal.variables, {(12, 0, 0): 1}),
        ),
    )
    request = IdealMembershipRequest(
        ideal=padded,
        polynomial=expanded.polynomial,
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "BUDGET_EXCEEDED"
    assert result.groebner_basis is None and result.remainder is None
    membership = polynomial_ideal_membership(request)
    assert membership.status == "BUDGET_EXCEEDED"


def test_prefix_cap_does_not_decide_outcome_for_late_collapse() -> None:
    # Reviewer counterexample (bounded instance): f is the sum of all
    # monomials of degree at most 8 in y,z (45 terms), so the canonical
    # prefix <x+f, x^12> has a reduced basis containing f^12 with 4,753
    # aggregate terms — beyond the retired 4,096-term intermediate work
    # envelope.  The complete ideal is instead <x, f> with a small basis,
    # and x has normal form zero, so the aggregate term count of a prefix
    # must never decide the outcome; only representability limits apply
    # before the complete source basis exists.
    from itertools import product

    variables = ("x", "y", "z")
    f_terms = {(0, a, b): 1 for a, b in product(range(9), repeat=2) if a + b <= 8}
    assert len(f_terms) == 45
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            _poly(variables, {(1, 0, 0): 1, **f_terms}),
            _poly(variables, {(12, 0, 0): 1}),
            _poly(variables, {(1, 0, 0): 1, (12, 0, 0): 1}),
        ),
    )
    request = IdealMembershipRequest(
        ideal=ideal,
        polynomial=_poly(variables, {(12, 0, 0): 1}),
        monomial_order="lex",
    )
    result = polynomial_ideal_normal_form(request)
    assert result.status == "COMPUTED"
    assert result.remainder is not None
    assert len(result.remainder.polynomial.terms) == 0
