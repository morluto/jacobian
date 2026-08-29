"""Commutative algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.ideals._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    GroebnerBasisResult,
    IdealMinimalPrimesRequest,
    IdealMinimalPrimesResult,
    IdealNormalFormRequest,
    IdealNormalFormResult,
    IdealQuotientRequest,
    IdealQuotientResult,
    IdealRadicalMembershipRequest,
    IdealRadicalMembershipResult,
    IdealRadicalRequest,
    IdealRadicalResult,
    IdealSaturationRequest,
    IdealSaturationResult,
)
from jacobian.math.polynomials.ideals.operations import (
    elimination_ideal,
    groebner_basis,
    ideal_minimal_primes,
    ideal_normal_form,
    ideal_quotient,
    ideal_radical,
    ideal_radical_membership,
    ideal_saturation,
)


def _run_minimal_primes(
    request: IdealMinimalPrimesRequest,
) -> IdealMinimalPrimesResult:
    return ideal_minimal_primes(request.ideal, resource_budget=request.resource_budget)


def _run_radical(request: IdealRadicalRequest) -> IdealRadicalResult:
    return ideal_radical(request.ideal, resource_budget=request.resource_budget)


def _run_radical_membership(
    request: IdealRadicalMembershipRequest,
) -> IdealRadicalMembershipResult:
    return ideal_radical_membership(request.ideal, request.polynomial)


def _run_quotient(request: IdealQuotientRequest) -> IdealQuotientResult:
    return ideal_quotient(
        request.dividend, request.divisor, resource_budget=request.resource_budget
    )


def _run_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    return ideal_saturation(
        request.ideal, request.denominator, resource_budget=request.resource_budget
    )


def _run_groebner(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    return groebner_basis(
        request.ideal,
        request.monomial_order,
        resource_budget=request.resource_budget,
    )


def _run_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    return ideal_normal_form(request.ideal, request.polynomial, request.monomial_order)


def _run_elimination(request: EliminationIdealRequest) -> EliminationIdealResult:
    return elimination_ideal(
        request.ideal,
        request.eliminated_variables,
        resource_budget=request.resource_budget,
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": list(exponents),
                }
                for numerator, denominator, exponents in sorted(
                    terms, key=lambda term: term[2], reverse=True
                )
            ]
        },
    }


def _ideal(
    variables: tuple[str, ...],
    *generators: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> dict[str, Any]:
    return {
        "variables": list(variables),
        "generators": [_polynomial(variables, generator) for generator in generators],
    }


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polynomial.ideal.minimal_primes.compute",
        "Compute minimal primes of a rational polynomial ideal",
        "Compute the complete minimal-prime family of a bounded ideal in "
        "QQ[x_1, ..., x_n] using Singular's minAssGTZ kernel. Components "
        "are prime ideals over QQ, not after extension to an algebraic closure; "
        "the result is canonically ordered and verified against the retained "
        "source by independent radical-intersection, minimality, and "
        "characteristic-set checks.",
        IdealMinimalPrimesRequest,
        IdealMinimalPrimesResult,
        _run_minimal_primes,
        "commutative-algebra",
        "minimal-primes",
        "irreducible-components",
        "exact",
        examples=(
            example(
                "coordinate_axes",
                "Compute the two QQ-minimal primes of <x*y> in Q[x,y].",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 1)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.radical.compute",
        "Compute the radical of an ideal",
        "Compute the exact radical sqrt(I) of a bounded polynomial ideal over "
        "QQ using the private Singular backend.",
        IdealRadicalRequest,
        IdealRadicalResult,
        _run_radical,
        "commutative-algebra",
        "radical",
        "exact",
        examples=(
            example(
                "ideal_xy",
                "Compute the radical of <x^2, xy> in Q[x,y]; every generator "
                "must use the same canonical ordered QQ polynomial ring.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.radical_membership.decide",
        "Check membership in the radical of an ideal",
        "Check whether a polynomial f lies in the radical sqrt(I) of the "
        "ideal I = <generators> in Q[variables], using the Rabinowitsch "
        "trick.",
        IdealRadicalMembershipRequest,
        IdealRadicalMembershipResult,
        _run_radical_membership,
        "commutative-algebra",
        "radical-membership",
        "exact",
        examples=(
            example(
                "membership_xy",
                "Check if x is in sqrt(<x^2>) in Q[x]; the ideal and "
                "polynomial must use the same canonical ordered QQ ring.",
                {
                    "ideal": _ideal(("x",), ((1, 1, (2,)),)),
                    "polynomial": _polynomial(("x",), ((1, 1, (1,)),)),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.quotient.compute",
        "Compute the ideal quotient (I : J)",
        "Compute the exact colon ideal (I : J) = {f : f*J subseteq I} over QQ "
        "using the private Singular backend.",
        IdealQuotientRequest,
        IdealQuotientResult,
        _run_quotient,
        "commutative-algebra",
        "ideal-quotient",
        "exact",
        examples=(
            example(
                "quotient_xy",
                "Compute (<x^2, xy> : <x>) in Q[x,y]; both ideals must use "
                "the same canonical ordered QQ polynomial ring.",
                {
                    "dividend": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
                    "divisor": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 0)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.saturation.compute",
        "Compute ideal saturation I : <d>^infinity",
        "Compute the exact saturation I : <d>^infinity of a bounded "
        "polynomial ideal I by a single nonzero polynomial d over QQ using "
        "the private Singular backend. The result is the saturated ideal "
        "with all components supported on the zero locus of d removed.",
        IdealSaturationRequest,
        IdealSaturationResult,
        _run_saturation,
        "commutative-algebra",
        "saturation",
        "exact",
        examples=(
            example(
                "saturation_xy",
                "Compute <xy> : <x>^infinity in Q[x,y]; this equals <y>. The "
                "denominator is one nonzero polynomial in the same canonical "
                "ordered QQ polynomial ring as the ideal.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 1)),),
                    ),
                    "denominator": _polynomial(
                        ("x", "y"),
                        ((1, 1, (1, 0)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.groebner_basis.compute",
        "Compute a reduced Groebner basis over QQ",
        "Compute a reduced Groewski basis for a bounded ideal in QQ[x_1, "
        "x_2, ..., x_n] using SymPy's exact groebner function. Returns the "
        "basis as a RationalPolynomialIdeal with the declared monomial order.",
        GroebnerBasisRequest,
        GroebnerBasisResult,
        _run_groebner,
        "commutative-algebra",
        "groebner-basis",
        "exact",
        examples=(
            example(
                "groebner_basis_xy",
                "Compute the Groebner basis of <x^2 - y, x*y - 1> in Q[x,y].",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)), (-1, 1, (0, 1))),
                        ((1, 1, (1, 1)), (-1, 1, (0, 0))),
                    ),
                    "monomial_order": "grevlex",
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.normal_form.compute",
        "Reduce a polynomial modulo an ideal",
        "Reduce one bounded polynomial modulo a bounded ideal in QQ[x_1, "
        "x_2, ..., x_n] using a Groebner basis remainder. Returns the exact "
        "remainder and whether the polynomial is in the ideal; a computation "
        "that exceeds the enforced wall-time bound returns a typed TIMEOUT "
        "outcome instead of a remainder.",
        IdealNormalFormRequest,
        IdealNormalFormResult,
        _run_normal_form,
        "commutative-algebra",
        "normal-form",
        "ideal-membership",
        "exact",
        examples=(
            example(
                "normal_form_xy",
                "Reduce x^2 modulo <x^2 - y^2> in Q[x,y]; the remainder is y^2, so x^2 is not in the ideal.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)), (-1, 1, (0, 2))),
                    ),
                    "polynomial": _polynomial(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.elimination.compute",
        "Compute an elimination ideal",
        "Compute the elimination ideal I ∩ QQ[remaining variables] by "
        "computing a lex Groebner basis and extracting the generators that "
        "involve only the remaining variables. A computation that exceeds the "
        "enforced wall-time budget returns a typed TIMEOUT outcome instead of "
        "an ideal.",
        EliminationIdealRequest,
        EliminationIdealResult,
        _run_elimination,
        "commutative-algebra",
        "elimination-ideal",
        "exact",
        examples=(
            example(
                "elimination_xy",
                "Compute <x^2 - y^2, x + y> ∩ Q[y] in Q[x,y]; eliminates x.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)), (-1, 1, (0, 2))),
                        ((1, 1, (1, 0)), (1, 1, (0, 1))),
                    ),
                    "eliminated_variables": ["x"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
