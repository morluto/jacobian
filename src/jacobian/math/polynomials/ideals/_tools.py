"""Commutative algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.ideals._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    GroebnerBasisResult,
    IdealContainmentRequest,
    IdealContainmentResult,
    IdealEqualityRequest,
    IdealEqualityResult,
    IdealMembershipCertificateRequest,
    IdealMembershipCertificateResult,
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
    MonomialIdealBettiRequest,
    MonomialIdealBettiResult,
)
from jacobian.math.polynomials.ideals.operations import (
    elimination_ideal,
    groebner_basis,
    ideal_containment,
    ideal_equality,
    ideal_membership_certificate,
    ideal_minimal_primes,
    ideal_normal_form,
    ideal_quotient,
    ideal_radical,
    ideal_radical_membership,
    ideal_saturation,
    monomial_ideal_graded_betti_table,
)


def _run_monomial_betti(
    request: MonomialIdealBettiRequest,
) -> MonomialIdealBettiResult:
    return monomial_ideal_graded_betti_table(request.ideal)


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


def _run_containment(request: IdealContainmentRequest) -> IdealContainmentResult:
    return ideal_containment(
        request.source,
        request.target,
        request.monomial_order,
        resource_budget=request.resource_budget,
    )


def _run_equality(request: IdealEqualityRequest) -> IdealEqualityResult:
    return ideal_equality(
        request.left,
        request.right,
        request.monomial_order,
        resource_budget=request.resource_budget,
    )


def _run_groebner(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    return groebner_basis(
        request.ideal,
        request.monomial_order,
        resource_budget=request.resource_budget,
    )


def _run_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    return ideal_normal_form(request.ideal, request.polynomial, request.monomial_order)


def _run_membership_certificate(
    request: IdealMembershipCertificateRequest,
) -> IdealMembershipCertificateResult:
    return ideal_membership_certificate(
        request.ideal, request.polynomial, request.cofactor_degree_bound
    )


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.monomial_ideal.graded_betti_table.compute",
        title="Compute a monomial ideal's graded Betti table",
        description="Compute every nonzero multigraded and standard-graded Betti number "
        "of a bounded minimally generated monomial ideal over QQ. The exact "
        "result includes the complete lcm-lattice crosscut homology profile, "
        "Castelnuovo--Mumford regularity, and whether the ideal has a linear "
        "resolution.",
        request_type=MonomialIdealBettiRequest,
        result_type=MonomialIdealBettiResult,
        run=_run_monomial_betti,
        tags=(
            "commutative-algebra",
            "monomial-ideal",
            "graded-betti-numbers",
            "free-resolution",
            "exact",
        ),
        examples=(
            OperationExample(
                name="two_quadrics",
                description="Compute the Betti table of <x^2,y^2> in QQ[x,y].",
                input={
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (0, 2)),),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.containment.decide",
        title="Decide containment of rational polynomial ideals",
        description="Decide whether one bounded ideal is contained in another ideal in "
        "the same ordered QQ polynomial ring. A positive result retains the "
        "exact normal form of every source generator; a negative result ends "
        "with the first nonzero normal-form obstruction.",
        request_type=IdealContainmentRequest,
        result_type=IdealContainmentResult,
        run=_run_containment,
        tags=("commutative-algebra", "ideal-containment", "normal-form", "exact"),
        examples=(
            OperationExample(
                name="contained_redundant_generators",
                description="Decide <x^2,xy> subseteq <x> in Q[x,y].",
                input={
                    "source": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
                    "target": _ideal(("x", "y"), ((1, 1, (1, 0)),)),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.equality.decide",
        title="Decide equality of rational polynomial ideals",
        description="Decide equality of two bounded ideals in one ordered QQ polynomial "
        "ring by mutual containment. Both source-ordered normal-form ledgers "
        "are computed under one request deadline.",
        request_type=IdealEqualityRequest,
        result_type=IdealEqualityResult,
        run=_run_equality,
        tags=("commutative-algebra", "ideal-equality", "mutual-containment", "exact"),
        examples=(
            OperationExample(
                name="equal_presentations",
                description="Compare <x,y> with the reordered, rescaled presentation <2y,3x>.",
                input={
                    "left": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 0)),),
                        ((1, 1, (0, 1)),),
                    ),
                    "right": _ideal(
                        ("x", "y"),
                        ((2, 1, (0, 1)),),
                        ((3, 1, (1, 0)),),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.minimal_primes.compute",
        runtime_requirements=("singular",),
        title="Compute minimal primes of a rational polynomial ideal",
        description="Compute the complete minimal-prime family of a bounded ideal in "
        "QQ[x_1, ..., x_n] using Singular's minAssGTZ kernel. Components "
        "are prime ideals over QQ, not after extension to an algebraic closure; "
        "the result is canonically ordered and verified against the retained "
        "source by independent radical-intersection, minimality, and "
        "characteristic-set checks.",
        request_type=IdealMinimalPrimesRequest,
        result_type=IdealMinimalPrimesResult,
        run=_run_minimal_primes,
        tags=(
            "commutative-algebra",
            "minimal-primes",
            "irreducible-components",
            "exact",
        ),
        examples=(
            OperationExample(
                name="coordinate_axes",
                description="Compute the two QQ-minimal primes of <x*y> in Q[x,y].",
                input={
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 1)),),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.radical.compute",
        runtime_requirements=("singular",),
        title="Compute the radical of an ideal",
        description="Compute the exact radical sqrt(I) of a bounded polynomial ideal over "
        "QQ using the private Singular backend.",
        request_type=IdealRadicalRequest,
        result_type=IdealRadicalResult,
        run=_run_radical,
        tags=("commutative-algebra", "radical", "exact"),
        examples=(
            OperationExample(
                name="ideal_xy",
                description="Compute the radical of <x^2, xy> in Q[x,y]; every generator "
                "must use the same canonical ordered QQ polynomial ring.",
                input={
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.radical_membership.decide",
        title="Check membership in the radical of an ideal",
        description="Check whether a polynomial f lies in the radical sqrt(I) of the "
        "ideal I = <generators> in Q[variables], using the Rabinowitsch "
        "trick.",
        request_type=IdealRadicalMembershipRequest,
        result_type=IdealRadicalMembershipResult,
        run=_run_radical_membership,
        tags=("commutative-algebra", "radical-membership", "exact"),
        examples=(
            OperationExample(
                name="membership_xy",
                description="Check if x is in sqrt(<x^2>) in Q[x]; the ideal and "
                "polynomial must use the same canonical ordered QQ ring.",
                input={
                    "ideal": _ideal(("x",), ((1, 1, (2,)),)),
                    "polynomial": _polynomial(("x",), ((1, 1, (1,)),)),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.quotient.compute",
        runtime_requirements=("singular",),
        title="Compute the ideal quotient (I : J)",
        description="Compute the exact colon ideal (I : J) = {f : f*J subseteq I} over QQ "
        "using the private Singular backend.",
        request_type=IdealQuotientRequest,
        result_type=IdealQuotientResult,
        run=_run_quotient,
        tags=("commutative-algebra", "ideal-quotient", "exact"),
        examples=(
            OperationExample(
                name="quotient_xy",
                description="Compute (<x^2, xy> : <x>) in Q[x,y]; both ideals must use "
                "the same canonical ordered QQ polynomial ring.",
                input={
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
    MathTool(
        operation_id="polynomial.ideal.saturation.compute",
        runtime_requirements=("singular",),
        title="Compute ideal saturation I : <d>^infinity",
        description="Compute the exact saturation I : <d>^infinity of a bounded "
        "polynomial ideal I by a single nonzero polynomial d over QQ using "
        "the private Singular backend. The result is the saturated ideal "
        "with all components supported on the zero locus of d removed.",
        request_type=IdealSaturationRequest,
        result_type=IdealSaturationResult,
        run=_run_saturation,
        tags=("commutative-algebra", "saturation", "exact"),
        examples=(
            OperationExample(
                name="saturation_xy",
                description="Compute <xy> : <x>^infinity in Q[x,y]; this equals <y>. The "
                "denominator is one nonzero polynomial in the same canonical "
                "ordered QQ polynomial ring as the ideal.",
                input={
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
    MathTool(
        operation_id="polynomial.ideal.groebner_basis.compute",
        title="Compute a reduced Groebner basis over QQ",
        description="Compute a reduced Groewski basis for a bounded ideal in QQ[x_1, "
        "x_2, ..., x_n] using SymPy's exact groebner function. Returns the "
        "basis as a RationalPolynomialIdeal with the declared monomial order.",
        request_type=GroebnerBasisRequest,
        result_type=GroebnerBasisResult,
        run=_run_groebner,
        tags=("commutative-algebra", "groebner-basis", "exact"),
        examples=(
            OperationExample(
                name="groebner_basis_xy",
                description="Compute the Groebner basis of <x^2 - y, x*y - 1> in Q[x,y].",
                input={
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
    MathTool(
        operation_id="polynomial.ideal.normal_form.compute",
        title="Reduce a polynomial modulo an ideal",
        description="Reduce one bounded polynomial modulo a bounded ideal in QQ[x_1, "
        "x_2, ..., x_n] using a Groebner basis remainder. Returns the exact "
        "remainder and whether the polynomial is in the ideal. Timeout or "
        "backend failure establishes no normal-form result.",
        request_type=IdealNormalFormRequest,
        result_type=IdealNormalFormResult,
        run=_run_normal_form,
        tags=("commutative-algebra", "normal-form", "ideal-membership", "exact"),
        examples=(
            OperationExample(
                name="normal_form_xy",
                description="Reduce x^2 modulo <x^2 - y^2> in Q[x,y]; the remainder is y^2, so x^2 is not in the ideal.",
                input={
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
    MathTool(
        operation_id="polynomial.ideal.membership_certificate.compute",
        title="Compute a bounded ideal-membership certificate",
        description=(
            "Search all generator cofactors through a declared total-degree bound. "
            "Return a primitive integral identity m*P = sum(A_i*G_i), or report "
            "that no such representation exists within that bound."
        ),
        request_type=IdealMembershipCertificateRequest,
        result_type=IdealMembershipCertificateResult,
        run=_run_membership_certificate,
        tags=(
            "commutative-algebra",
            "ideal-membership",
            "certificate",
            "generator-coefficients",
            "exact",
        ),
        discovery_terms=(
            "ideal representation",
            "polynomial cofactors",
            "generator coefficients",
            "Macaulay matrix",
        ),
        examples=(
            OperationExample(
                name="bounded_xy_certificate",
                description=(
                    "Express x^2-y^2 using the ordered generators x-y and x+y; "
                    "cofactor total degree is bounded by one."
                ),
                input={
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 0)), (-1, 1, (0, 1))),
                        ((1, 1, (1, 0)), (1, 1, (0, 1))),
                    ),
                    "polynomial": _polynomial(
                        ("x", "y"),
                        ((1, 1, (2, 0)), (-1, 1, (0, 2))),
                    ),
                    "cofactor_degree_bound": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.elimination.compute",
        title="Compute an elimination ideal",
        description="Compute the elimination ideal I ∩ QQ[remaining variables] by "
        "computing a lex Groebner basis and extracting the generators that "
        "involve only the remaining variables. Timeout or backend failure "
        "establishes no elimination ideal.",
        request_type=EliminationIdealRequest,
        result_type=EliminationIdealResult,
        run=_run_elimination,
        tags=("commutative-algebra", "elimination-ideal", "exact"),
        examples=(
            OperationExample(
                name="elimination_xy",
                description="Compute <x^2 - y^2, x + y> ∩ Q[y] in Q[x,y]; eliminates x.",
                input={
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
