"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import sympy

from jacobian.math.commutative_algebra_ops._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    GroebnerBasisResult,
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
from jacobian.math.commutative_algebra_ops._singular import (
    run_singular_ideal_operation,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)


def compute_ideal_radical(request: IdealRadicalRequest) -> IdealRadicalResult:
    """Compute an exact ideal radical through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "radical",
        request.ideal,
        None,
        request.resource_budget,
    )
    return IdealRadicalResult(
        outcome=backend.outcome,
        radical=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def compute_ideal_radical_membership(
    request: IdealRadicalMembershipRequest,
) -> IdealRadicalMembershipResult:
    """Decide radical membership by the exact Rabinowitsch criterion."""

    variable_symbols = symbols_for_variables(request.ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    polynomial = rational_polynomial_to_sympy(request.polynomial).as_expr()
    auxiliary = sympy.Dummy("jacobian_rabinowitsch")
    basis = sympy.groebner(
        [*ideal_generators, 1 - auxiliary * polynomial],
        *variable_symbols,
        auxiliary,
        order="grevlex",
        domain=sympy.QQ,
    )
    return IdealRadicalMembershipResult(in_radical=len(basis) == 1 and basis[0] == 1)


def compute_ideal_quotient(request: IdealQuotientRequest) -> IdealQuotientResult:
    """Compute an exact ideal quotient through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "quotient",
        request.dividend,
        request.divisor,
        request.resource_budget,
    )
    return IdealQuotientResult(
        outcome=backend.outcome,
        quotient=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )




def compute_ideal_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    """Compute an exact ideal saturation I : <d>^infinity through the bounded Singular backend."""

    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
    )
    saturation_ideal = RationalPolynomialIdeal(
        variables=request.ideal.variables,
        generators=(request.saturation_polynomial,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        saturation_ideal,
        request.resource_budget,
    )
    return IdealSaturationResult(
        outcome=backend.outcome,
        saturation=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )

__all__ = [
    "compute_ideal_quotient",
    "compute_ideal_radical",
    "compute_ideal_radical_membership",
    "compute_ideal_saturation",
]


def compute_groebner_basis(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    from jacobian.math.commutative_algebra_ops._models import (
        MAX_OUTPUT_TERMS,
        GroebnerBasisResult,
    )
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
    )

    variables = request.ideal.variables
    variable_symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map.get(request.monomial_order, "grevlex")

    # Enforce the advertised wall-time budget around the backend call so an
    # admitted request cannot occupy the inline execution path indefinitely.
    import signal

    class _BudgetExceededError(TimeoutError):
        pass

    def _on_budget(signum, frame):
        raise _BudgetExceededError("groebner computation exceeded wall_seconds")

    previous_handler = signal.signal(signal.SIGALRM, _on_budget)
    signal.setitimer(signal.ITIMER_REAL, request.resource_budget.wall_seconds)
    try:
        basis = sympy.groebner(
            ideal_generators,
            *variable_symbols,
            order=order,
            domain=sympy.QQ,
        )
    except _BudgetExceededError:
        return GroebnerBasisResult(
            outcome="TIMEOUT",
            monomial_order=request.monomial_order,
            detail=(
                "groebner computation exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s budget"
            ),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    basis_generators = tuple(
        rational_polynomial_from_sympy(
            sympy.Poly(expr, *variable_symbols, domain=sympy.QQ),
            variables,
            maximum_terms=MAX_OUTPUT_TERMS,
        )
        for expr in basis
    )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=basis_generators,
    )

    return GroebnerBasisResult(
        basis=ideal,
        generator_count=len(basis_generators),
        monomial_order=request.monomial_order,
    )


def compute_ideal_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    from jacobian.math.commutative_algebra_ops._models import IdealNormalFormResult
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy

    variables = request.ideal.variables
    variable_symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    poly = rational_polynomial_to_sympy(request.polynomial).as_expr()

    # Membership is decided only against a Groebner basis; reducing against
    # the raw generators computes a division remainder that proves nothing.
    basis = sympy.groebner(
        ideal_generators,
        *variable_symbols,
        order="grevlex",
        domain=sympy.QQ,
    )
    _, remainder = sympy.reduced(
        poly,
        list(basis.exprs),
        *variable_symbols,
        order="grevlex",
        domain=sympy.QQ,
    )

    remainder_poly = rational_polynomial_from_sympy(
        sympy.Poly(remainder, *variable_symbols, domain=sympy.QQ),
        variables,
    )

    # The polynomial is in the ideal if and only if the remainder is zero
    in_ideal = len(remainder_poly.polynomial.terms) == 0

    return IdealNormalFormResult(
        remainder=remainder_poly,
        in_ideal=in_ideal,
    )


def compute_elimination_ideal(request: EliminationIdealRequest) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""
    from jacobian.math.commutative_algebra_ops._models import EliminationIdealResult
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
        SparseRationalPolynomial,
    )

    variables = list(request.ideal.variables)
    eliminated_set = set(request.eliminated_variables)
    remaining = [v for v in variables if v not in eliminated_set]

    # The elimination theorem requires the eliminated variables to precede
    # the retained variables in the lex monomial order.
    ordered_variables = tuple(v for v in variables if v in eliminated_set) + tuple(remaining)
    ordered_symbols = symbols_for_variables(ordered_variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    basis = sympy.groebner(
        ideal_generators,
        *ordered_symbols,
        order="lex",
        domain=sympy.QQ,
    )

    remaining_symbols = symbols_for_variables(tuple(remaining))
    elimination_generators = []
    unit_ideal = False
    for expr in basis:
        poly = sympy.Poly(expr, *ordered_symbols, domain=sympy.QQ)
        involved = {str(s) for s in poly.free_symbols}
        if not involved:
            # A nonzero constant basis element means the whole ring.
            unit_ideal = True
            break
        if involved.issubset(set(remaining)):
            elimination_generators.append(
                rational_polynomial_from_sympy(
                    sympy.Poly(expr, *remaining_symbols, domain=sympy.QQ),
                    tuple(remaining),
                )
            )

    if unit_ideal:
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomials.values import (
            RationalPolynomial,
            RationalPolynomialTerm,
        )
        one = RationalPolynomial(
            variables=tuple(remaining),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,) * len(remaining),
                    ),
                )
            ),
        )
        elimination_generators = [one]
    elif not elimination_generators:
        # No retained-only basis elements: the elimination ideal is the
        # zero ideal, represented by its canonical zero generator.
        from jacobian.math.polynomials.values import RationalPolynomial
        zero = RationalPolynomial(
            variables=tuple(remaining),
            polynomial=SparseRationalPolynomial(terms=()),
        )
        elimination_generators = [zero]

    ideal = RationalPolynomialIdeal(
        variables=tuple(remaining),
        generators=tuple(elimination_generators),
    )

    return EliminationIdealResult(
        elimination_ideal=ideal,
        eliminated_variables=tuple(request.eliminated_variables),
    )
