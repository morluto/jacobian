"""Domain functions for commutative algebra operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

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
from jacobian.math.polynomials.values import RationalPolynomialIdeal


class _GroebnerBudgetExceededError(TimeoutError):
    """Internal signal-handler escape for the enforced Gröbner budget."""


class _NormalFormTimeoutError(TimeoutError):
    """Internal signal-handler escape for the enforced normal-form budget."""


class _EliminationTimeoutError(TimeoutError):
    """Internal signal-handler escape for the enforced elimination budget."""


class _ResultLimitExceededError(ValueError):
    """The exact backend result exceeds the declared output limits."""


def _run_under_wall_budget[T](work: Callable[[], T], wall_seconds: float) -> T | None:
    """Run bounded exact work under the declared wall-time budget.

    The budget is enforced by a supervised worker thread, so it holds on any
    MCP worker thread rather than only on the interpreter main thread where
    signal-based timers are forbidden. Returns ``None`` when the budget
    expires; the expired computation keeps running detached and its result
    is discarded, while exceptions propagate to the caller.
    """

    import threading

    finished = threading.Event()
    box: dict[str, T | BaseException] = {}

    def _target() -> None:
        try:
            box["value"] = work()
        except BaseException as error:
            box["error"] = error
        finally:
            finished.set()

    threading.Thread(target=_target, daemon=True).start()
    if not finished.wait(timeout=wall_seconds):
        return None
    outcome = box["error"] if "error" in box else box["value"]
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


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
    """Compute I : <d>^infinity through the bounded Singular backend."""

    denominator = RationalPolynomialIdeal(
        variables=request.denominator.variables,
        generators=(request.denominator,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        denominator,
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
        RationalPolynomial,
        RationalPolynomialIdeal,
        SparseRationalPolynomial,
    )

    variables = request.ideal.variables
    variable_symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map.get(request.monomial_order, "grevlex")

    def _work() -> GroebnerBasisResult:
        basis = sympy.groebner(
            ideal_generators,
            *variable_symbols,
            order=order,
            domain=sympy.QQ,
        )

        basis_generators = []
        for expr in basis:
            try:
                converted = rational_polynomial_from_sympy(
                    sympy.Poly(expr, *variable_symbols, domain=sympy.QQ),
                    variables,
                    maximum_terms=MAX_OUTPUT_TERMS,
                )
            except ValueError as error:
                raise _ResultLimitExceededError(error) from None
            basis_generators.append(converted)

        if not basis_generators:
            # sympy.groebner yields an empty iterable for the zero ideal
            # represented by a single zero polynomial; canonicalize to the
            # domain's zero-polynomial generator.
            zero = RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=()),
            )
            basis_generators.append(zero)

        ideal = RationalPolynomialIdeal(
            variables=variables,
            generators=tuple(basis_generators),
        )

        return GroebnerBasisResult(
            request=request,
            basis=ideal,
            generator_count=len(basis_generators),
            monomial_order=request.monomial_order,
        )

    # The declared wall-time budget covers the backend call, result
    # conversion, and source-bound invariant verification.
    try:
        computed = _run_under_wall_budget(
            _work, request.resource_budget.wall_seconds
        )
    except (
        _GroebnerBudgetExceededError,
        _NormalFormTimeoutError,
        _EliminationTimeoutError,
    ):
        computed = None
    except _ResultLimitExceededError as error:
        return GroebnerBasisResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            monomial_order=request.monomial_order,
            detail=(
                "the exact reduced Gröbner basis exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    if computed is None:
        return GroebnerBasisResult(
            request=request,
            outcome="TIMEOUT",
            monomial_order=request.monomial_order,
            detail=(
                "groebner computation or source-bound verification exceeded "
                f"the enforced {request.resource_budget.wall_seconds}s budget"
            ),
        )
    return computed


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

    def _work() -> IdealNormalFormResult:
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

        try:
            remainder_poly = rational_polynomial_from_sympy(
                sympy.Poly(remainder, *variable_symbols, domain=sympy.QQ),
                variables,
            )
        except ValueError as error:
            raise _ResultLimitExceededError(error) from None

        # The polynomial is in the ideal if and only if the remainder is zero
        in_ideal = len(remainder_poly.polynomial.terms) == 0

        return IdealNormalFormResult(
            request=request,
            remainder=remainder_poly,
            in_ideal=in_ideal,
        )

    # A conservative 10-second budget covers the inline Groebner computation
    # and its source-bound verification; hard ideals would otherwise occupy
    # the execution path indefinitely.
    try:
        computed = _run_under_wall_budget(_work, 10)
    except (_NormalFormTimeoutError, _GroebnerBudgetExceededError):
        computed = None
    except _ResultLimitExceededError as error:
        return IdealNormalFormResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            detail=(
                "the exact normal form exceeds the declared exact-result "
                f"limit: {error}"
            ),
        )
    if computed is None:
        # An admitted request must observe a typed mathematical outcome rather
        # than a host exception, so budget expiry is part of the result contract.
        return IdealNormalFormResult(
            request=request,
            outcome="TIMEOUT",
            detail=(
                "the Gröbner reduction or its source-bound verification "
                "exceeded the enforced 10s wall-time bound"
            ),
        )
    return computed


def _elimination_generators_from_basis(
    basis: Iterable[Any],
    ordered_symbols: tuple[Any, ...],
    remaining: list[str],
) -> tuple[list[Any], bool]:
    """Extract retained-variable generators from a lex Groebner basis."""
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy

    remaining_symbols = symbols_for_variables(tuple(remaining))
    elimination_generators: list[Any] = []
    unit_ideal = False
    for expr in basis:
        poly = sympy.Poly(expr, *ordered_symbols, domain=sympy.QQ)
        involved = {str(s) for s in poly.free_symbols}
        if not involved:
            # A nonzero constant basis element means the whole ring.
            unit_ideal = True
            break
        if involved.issubset(set(remaining)):
            try:
                elimination_generators.append(
                    rational_polynomial_from_sympy(
                        sympy.Poly(expr, *remaining_symbols, domain=sympy.QQ),
                        tuple(remaining),
                    )
                )
            except ValueError as error:
                raise _ResultLimitExceededError(error) from None
    return elimination_generators, unit_ideal


def compute_elimination_ideal(
    request: EliminationIdealRequest,
) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""
    from jacobian.math.commutative_algebra_ops._models import EliminationIdealResult
    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
        SparseRationalPolynomial,
    )

    variables = list(request.ideal.variables)
    eliminated_set = set(request.eliminated_variables)
    remaining = [v for v in variables if v not in eliminated_set]

    # The elimination theorem requires the eliminated variables to precede
    # the retained variables in the lex monomial order.
    ordered_variables = tuple(v for v in variables if v in eliminated_set) + tuple(
        remaining
    )
    ordered_symbols = symbols_for_variables(ordered_variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    def _work() -> EliminationIdealResult:
        basis = sympy.groebner(
            ideal_generators,
            *ordered_symbols,
            order="lex",
            domain=sympy.QQ,
        )

        elimination_generators, unit_ideal = _elimination_generators_from_basis(
            basis, ordered_symbols, remaining
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
            request=request,
            elimination_ideal=ideal,
            eliminated_variables=tuple(request.eliminated_variables),
        )

    try:
        computed = _run_under_wall_budget(
            _work, request.resource_budget.wall_seconds
        )
    except _EliminationTimeoutError:
        computed = None
    except _ResultLimitExceededError as error:
        return EliminationIdealResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the exact elimination ideal exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    if computed is None:
        # An admitted request must observe a typed mathematical outcome rather
        # than a host exception, so budget expiry is part of the result contract.
        return EliminationIdealResult(
            request=request,
            outcome="TIMEOUT",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the lex Gröbner elimination or its source-bound verification "
                "exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s wall-time budget"
            ),
        )
    return computed
