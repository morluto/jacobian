"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import json
import time
from typing import Any

import sympy

from jacobian.canonical import CanonicalizationError, canonicalize_json
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
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
    computed_minimal_primes_result,
)
from jacobian.math.polynomials.ideals._singular import (
    run_bounded_stdin_python_kernel,
    run_singular_ideal_operation,
    run_singular_minimal_primes,
    run_singular_minimal_primes_verification,
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


class _SympyKernelTimeoutError(TimeoutError):
    """The bounded SymPy worker exceeded the declared wall-time budget."""


class _SympyKernelError(RuntimeError):
    """The bounded SymPy worker failed without producing an exact result."""


def _require_transportable_minimal_primes_result(
    result: IdealMinimalPrimesResult,
) -> None:
    """Reject a family whose complete public result exceeds JSON limits."""

    try:
        canonicalize_json(result.model_dump(mode="json"))
    except CanonicalizationError as error:
        raise _ResultLimitExceededError(
            "the exact minimal-prime result exceeds the canonical transport bound"
        ) from error


_SYMPY_WORKER_SCRIPT = r"""
import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read().decode("ascii"))
    try:
        import sympy

        from jacobian.math.polynomials._conversions import (
            rational_polynomial_from_sympy,
            rational_polynomial_to_sympy,
            symbols_for_variables,
        )
        from jacobian.math.polynomials.values import (
            MAX_POLYNOMIAL_EXPONENT,
            RationalPolynomial,
        )

        def require_admissible_exponents(poly: object) -> None:
            # Classify oversized output exponents as a result limit BEFORE
            # canonical conversion, so the parent reports LIMIT_EXCEEDED
            # instead of a post-hoc conversion failure.
            largest = max(
                (max(monom) if monom else 0 for monom in poly.monoms()),
                default=0,
            )
            if largest > MAX_POLYNOMIAL_EXPONENT:
                raise ValueError(
                    f"polynomial result exceeds the "
                    f"{MAX_POLYNOMIAL_EXPONENT}-exponent operation budget"
                )

        mode = payload["mode"]
        variables = tuple(payload["variables"])
        symbols = symbols_for_variables(variables)
        gens = [
            RationalPolynomial.model_validate(item)
            for item in payload["generators"]
        ]
        exprs = [rational_polynomial_to_sympy(g).as_expr() for g in gens]

        def dump(expr: object, terms_variable: tuple, maximum_terms: int) -> dict:
            poly = sympy.Poly(
                expr, *symbols_for_variables(terms_variable), domain=sympy.QQ
            )
            require_admissible_exponents(poly)
            converted = rational_polynomial_from_sympy(
                poly, terms_variable, maximum_terms=maximum_terms
            )
            return converted.model_dump(mode="json")

        if mode == "groebner":
            maximum_terms = payload["maximum_terms"]
            basis = sympy.groebner(
                exprs, *symbols, order=payload["order"], domain=sympy.QQ
            )
            generators = []
            aggregate_terms = 0
            for expr in basis:
                converted = dump(expr, variables, maximum_terms)
                aggregate_terms += len(converted["polynomial"]["terms"])
                # The exact-result budget bounds the whole basis, not each
                # polynomial separately, so classify a basis-wide crossing
                # as LIMIT_EXCEEDED before any oversized payload is emitted.
                if aggregate_terms > maximum_terms:
                    raise ValueError(
                        "the reduced Groebner basis exceeds the "
                        f"{maximum_terms}-term aggregate operation budget"
                    )
                generators.append(converted)
            _emit({"status": "ok", "generators": generators})
        elif mode == "normal_form":
            target = RationalPolynomial.model_validate(payload["polynomial"])
            poly_expr = rational_polynomial_to_sympy(target).as_expr()
            basis = sympy.groebner(
                exprs, *symbols, order=payload.get("order", "grevlex"), domain=sympy.QQ
            )
            _, remainder = sympy.reduced(
                poly_expr,
                list(basis.exprs),
                *symbols,
                order=payload.get("order", "grevlex"),
                domain=sympy.QQ,
            )
            remainder_poly = sympy.Poly(remainder, *symbols, domain=sympy.QQ)
            converted = rational_polynomial_from_sympy(
                remainder_poly, variables
            )
            _emit(
                {
                    "status": "ok",
                    "remainder": converted.model_dump(mode="json"),
                }
            )
        elif mode == "verify_ideal_equality":
            claimed = [
                RationalPolynomial.model_validate(item)
                for item in payload["basis"]
            ]
            claimed_exprs = [
                rational_polynomial_to_sympy(g).as_expr() for g in claimed
            ]
            source_basis = sympy.groebner(
                exprs, *symbols, order=payload.get("order", "grevlex"),
                domain=sympy.QQ,
            )
            for generator in claimed_exprs:
                _, rem = sympy.reduced(
                    generator,
                    list(source_basis.exprs),
                    *symbols,
                    order=payload.get("order", "grevlex"),
                    domain=sympy.QQ,
                )
                if rem != 0:
                    _emit({
                        "status": "ok",
                        "equal": False,
                        "detail": "a basis generator leaves a nonzero "
                                  "remainder modulo the source ideal",
                    })
                    return
            for expr in exprs:
                if expr.is_zero:
                    continue
                _, rem = sympy.reduced(
                    expr,
                    claimed_exprs,
                    *symbols,
                    order=payload.get("order", "grevlex"),
                    domain=sympy.QQ,
                )
                if rem != 0:
                    _emit({
                        "status": "ok",
                        "equal": False,
                        "detail": "a source generator is not contained in "
                                  "the claimed basis ideal",
                    })
                    return
            _emit({"status": "ok", "equal": True})
        elif mode == "verify_groebner_basis":
            # One bounded worker pass replays every defining invariant of a
            # claimed reduced Groebner basis: unit leading coefficients, no
            # generator divisible by another's leading monomial, Buchberger
            # S-polynomial reduction, and both ideal inclusions.
            order = payload.get("order", "grevlex")
            claimed = [
                RationalPolynomial.model_validate(item)
                for item in payload["basis"]
            ]
            claimed_exprs = [
                rational_polynomial_to_sympy(g).as_expr() for g in claimed
            ]
            nonzero = [expr for expr in claimed_exprs if not expr.is_zero]

            def fail(detail: str) -> None:
                _emit({"status": "ok", "equal": False, "detail": detail})

            leading_terms = [sympy.LT(e, *symbols, order=order) for e in nonzero]
            for index, expr in enumerate(nonzero):
                if sympy.LC(expr, *symbols, order=order) != 1:
                    fail("a reduced Groebner basis has unit leading coefficients")
                    return
                others = [lt for j, lt in enumerate(leading_terms) if j != index]
                if others:
                    _, remainder = sympy.reduced(
                        expr, others, *symbols, order=order, domain=sympy.QQ
                    )
                    if remainder != expr:
                        fail(
                            "reduced Groebner basis generators must contain no "
                            "other leading monomial"
                        )
                        return
            polys = [sympy.Poly(e, *symbols, domain=sympy.QQ) for e in nonzero]
            # Poly.monoms() ranks monomials lexicographically unless given
            # an explicit order object, so replay must resolve the declared
            # order; lex-default leading exponents would fabricate
            # S-polynomials that validate non-Groebner lists under
            # grlex/grevlex.
            monomial_order = getattr(sympy.polys.orderings, order)
            lead_exps = [poly.monoms(order=monomial_order)[0] for poly in polys]

            def monomial(exps: tuple) -> object:
                product = sympy.Integer(1)
                for sym, exponent in zip(symbols, exps):
                    if exponent:
                        product *= sym**exponent
                return product

            count = len(nonzero)
            for first in range(count):
                for second in range(first + 1, count):
                    lcm_exp = tuple(
                        max(x, y)
                        for x, y in zip(lead_exps[first], lead_exps[second])
                    )
                    s_poly = nonzero[first] * monomial(
                        tuple(x - y for x, y in zip(lcm_exp, lead_exps[first]))
                    ) - nonzero[second] * monomial(
                        tuple(x - y for x, y in zip(lcm_exp, lead_exps[second]))
                    )
                    _, remainder = sympy.reduced(
                        s_poly, nonzero, *symbols, order=order, domain=sympy.QQ
                    )
                    if remainder != 0:
                        fail(
                            "basis S-polynomials must reduce to zero; the list "
                            "is not a Groebner basis of the retained ideal"
                        )
                        return
            source_basis = sympy.groebner(exprs, *symbols, order=order, domain=sympy.QQ)
            for generator in nonzero:
                _, rem = sympy.reduced(
                    generator,
                    list(source_basis.exprs),
                    *symbols,
                    order=order,
                    domain=sympy.QQ,
                )
                if rem != 0:
                    fail(
                        "a basis generator leaves a nonzero remainder "
                        "modulo the source ideal"
                    )
                    return
            for expr in exprs:
                if expr.is_zero:
                    continue
                _, rem = sympy.reduced(
                    expr, claimed_exprs, *symbols, order=order, domain=sympy.QQ
                )
                if rem != 0:
                    fail(
                        "a source generator is not contained in "
                        "the claimed basis ideal"
                    )
                    return
            _emit({"status": "ok", "equal": True})
        elif mode == "elimination":
            eliminated = set(payload["eliminated"])
            remaining = [v for v in variables if v not in eliminated]
            ordered = tuple(v for v in variables if v in eliminated) + tuple(
                remaining
            )
            ordered_symbols = symbols_for_variables(ordered)
            basis = sympy.groebner(
                exprs, *ordered_symbols, order="lex", domain=sympy.QQ
            )
            remaining_symbols = symbols_for_variables(tuple(remaining))
            generators: list[dict] = []
            unit_ideal = False
            for expr in basis:
                poly = sympy.Poly(expr, *ordered_symbols, domain=sympy.QQ)
                involved = {str(s) for s in poly.free_symbols}
                if not involved:
                    unit_ideal = True
                    break
                if involved.issubset(set(remaining)):
                    converted_poly = sympy.Poly(
                        expr, *remaining_symbols, domain=sympy.QQ
                    )
                    require_admissible_exponents(converted_poly)
                    converted = rational_polynomial_from_sympy(
                        converted_poly,
                        tuple(remaining),
                    )
                    generators.append(converted.model_dump(mode="json"))
            _emit(
                {
                    "status": "ok",
                    "unit_ideal": unit_ideal,
                    "generators": generators,
                    "remaining": remaining,
                }
            )
        else:  # pragma: no cover - parent owns the mode vocabulary
            _emit({"status": "error", "detail": "unknown kernel mode"})
    except Exception as error:  # noqa: BLE001 - classified for the parent
        message = str(error)
        status = "limit" if "operation budget" in message else "error"
        _emit({"status": status, "detail": message})


def _emit(result: dict) -> None:
    json.dump(result, sys.stdout)
    sys.stdout.flush()


main()
"""

_STDOUT_LIMIT = 8 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024


def _run_sympy_kernel(payload: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    """Run one exact Groebner-kernel computation in a killable worker.

    The killable-process launch and executable discovery live in the
    domain's external-tool owner (``_singular``); this wrapper maps the
    bounded outcome onto the typed kernel exceptions.
    """
    try:
        timed_out, stdout, limit_exceeded = run_bounded_stdin_python_kernel(
            _SYMPY_WORKER_SCRIPT,
            json.dumps(payload),
            wall_seconds=wall_seconds,
            stdout_limit=_STDOUT_LIMIT,
            stderr_limit=_STDERR_LIMIT,
        )
    except OSError as error:
        raise _SympyKernelError(str(error)) from None
    if timed_out:
        raise _SympyKernelTimeoutError()
    if limit_exceeded:
        # A killed worker leaves truncated output: the exact result exceeded
        # the transport cap, which is the declared LIMIT_EXCEEDED outcome,
        # not an ordinary JSON failure to be misclassified as ERROR.
        raise _ResultLimitExceededError(
            "the exact kernel result exceeded the declared transport bound"
        )
    try:
        result = json.loads(stdout)
    except (UnicodeDecodeError, ValueError) as error:
        raise _SympyKernelError(str(error)) from None
    if result.get("status") == "limit":
        raise _ResultLimitExceededError(result.get("detail", ""))
    if result.get("status") != "ok":
        raise _SympyKernelError(str(result.get("detail", "kernel failed")))
    typed: dict[str, Any] = result
    return typed


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


def compute_ideal_minimal_primes(
    request: IdealMinimalPrimesRequest,
) -> IdealMinimalPrimesResult:
    """Compute the complete minimal-prime family over ``QQ``.

    The producing pass runs Singular's ``minAssGTZE`` kernel. A second
    bounded pass then verifies the family's defining invariants by
    independent evidence — radical-intersection equality, pairwise
    non-containment, and agreement with the characteristic-set
    decomposition — under ONE operation-level deadline: the verifier
    subprocess receives only the wall allowance remaining after the
    producing pass, so one request never exceeds its declared wall-time
    budget.
    """

    started = time.monotonic()
    backend = run_singular_minimal_primes(request.ideal, request.resource_budget)
    components = backend.components
    if backend.outcome != "COMPUTED" or components is None:
        return IdealMinimalPrimesResult(
            request=request,
            outcome=backend.outcome,
            components=None,
            backend_version=None,
            detail=backend.detail,
        )

    remaining = float(request.resource_budget.wall_seconds) - (
        time.monotonic() - started
    )
    if remaining <= 0:
        return IdealMinimalPrimesResult(
            request=request,
            outcome="TIMEOUT",
            components=None,
            backend_version=None,
            detail=(
                "The minimal-prime defining-invariant verification did not "
                "complete within the declared backend budget."
            ),
        )

    verdict = run_singular_minimal_primes_verification(
        request.ideal,
        components,
        request.resource_budget,
        wall_seconds=remaining,
    )
    if verdict == "VERIFIED":
        # The producing pass and the independent verification pass above
        # completed this request under one operation-level deadline. The
        # trusted factory skips only a repeated backend verification while
        # still enforcing shape, ring, exact-result envelopes, ordering, and
        # uniqueness; externally supplied JSON always runs the model
        # validator's own independent verification.
        try:
            result = computed_minimal_primes_result(
                request=request,
                components=components,
                backend_version=backend.backend_version,
            )
            _require_transportable_minimal_primes_result(result)
            return result
        except _ResultLimitExceededError as error:
            return IdealMinimalPrimesResult(
                request=request,
                outcome="LIMIT_EXCEEDED",
                components=None,
                backend_version=None,
                detail=str(error),
            )
        except ValueError:
            return IdealMinimalPrimesResult(
                request=request,
                outcome="ERROR",
                components=None,
                backend_version=None,
                detail=(
                    "The computed minimal-prime family violated its own shape, "
                    "ring, exact-result-envelope, ordering, or uniqueness "
                    "invariant."
                ),
            )
    if verdict == "REFUTED":
        return IdealMinimalPrimesResult(
            request=request,
            outcome="ERROR",
            components=None,
            backend_version=None,
            detail=(
                "The computed minimal-prime family failed its independent "
                "primality, minimality, or radical-intersection verification."
            ),
        )
    if verdict == "TIMEOUT":
        return IdealMinimalPrimesResult(
            request=request,
            outcome="TIMEOUT",
            components=None,
            backend_version=None,
            detail=(
                "The minimal-prime defining-invariant verification did not "
                "complete within the declared backend budget."
            ),
        )
    if verdict == "UNAVAILABLE":
        return IdealMinimalPrimesResult(
            request=request,
            outcome="UNAVAILABLE",
            components=None,
            backend_version=None,
            detail=(
                "The supported Singular backend became unavailable during "
                "independent verification."
            ),
        )
    if verdict == "CANCELLED":
        return IdealMinimalPrimesResult(
            request=request,
            outcome="CANCELLED",
            components=None,
            backend_version=None,
            detail=(
                "The minimal-prime defining-invariant verification was "
                "cancelled before producing a verdict."
            ),
        )
    return IdealMinimalPrimesResult(
        request=request,
        outcome="ERROR",
        components=None,
        backend_version=None,
        detail=(
            "The independent minimal-prime verification failed without "
            "producing a verdict."
        ),
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
    "compute_ideal_minimal_primes",
    "compute_ideal_quotient",
    "compute_ideal_radical",
    "compute_ideal_radical_membership",
    "compute_ideal_saturation",
]


def compute_groebner_basis(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    from jacobian.math.polynomials.ideals._models import GroebnerBasisResult
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialIdeal,
    )

    variables = request.ideal.variables
    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map.get(request.monomial_order, "grevlex")
    payload = {
        "mode": "groebner",
        "variables": list(variables),
        "order": order,
        "maximum_terms": request.resource_budget.maximum_output_terms,
        "generators": [
            generator.model_dump(mode="json") for generator in request.ideal.generators
        ],
    }

    # The unbounded search runs in a killable worker under the declared
    # wall-time budget; result assembly and source-bound verification then
    # operate only on the declared output limits.
    try:
        result_payload = _run_sympy_kernel(
            payload, request.resource_budget.wall_seconds
        )
    except _SympyKernelTimeoutError:
        return GroebnerBasisResult(
            request=request,
            outcome="TIMEOUT",
            monomial_order=request.monomial_order,
            detail=(
                "groebner computation exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s budget"
            ),
        )
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
    except _SympyKernelError as error:
        return GroebnerBasisResult(
            request=request,
            outcome="ERROR",
            monomial_order=request.monomial_order,
            detail=(
                "the bounded Groebner kernel failed without producing an "
                f"exact basis: {error}"
            ),
        )

    basis_generators = [
        RationalPolynomial.model_validate(item) for item in result_payload["generators"]
    ]
    if not basis_generators:
        from jacobian.math.polynomials.values import SparseRationalPolynomial

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


def compute_ideal_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    from jacobian.math.polynomials.ideals._models import IdealNormalFormResult
    from jacobian.math.polynomials.values import RationalPolynomial

    payload = {
        "mode": "normal_form",
        "variables": list(request.ideal.variables),
        "order": request.monomial_order,
        "generators": [
            generator.model_dump(mode="json") for generator in request.ideal.generators
        ],
        "polynomial": request.polynomial.model_dump(mode="json"),
    }

    # A conservative 10-second budget bounds the killable kernel that runs
    # the unbounded Gröbner search; the remainder conversion and its
    # source-bound replay then operate on declared output limits.
    try:
        result_payload = _run_sympy_kernel(payload, 10)
    except _SympyKernelTimeoutError:
        return IdealNormalFormResult(
            request=request,
            outcome="TIMEOUT",
            detail=("the Gröbner reduction exceeded the enforced 10s wall-time bound"),
        )
    except _ResultLimitExceededError as error:
        return IdealNormalFormResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            detail=(
                "the exact normal form exceeds the declared exact-result "
                f"limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return IdealNormalFormResult(
            request=request,
            outcome="ERROR",
            detail=(
                "the bounded reduction kernel failed without producing an "
                f"exact remainder: {error}"
            ),
        )

    remainder_poly = RationalPolynomial.model_validate(result_payload["remainder"])
    in_ideal = len(remainder_poly.polynomial.terms) == 0
    return IdealNormalFormResult(
        request=request,
        remainder=remainder_poly,
        in_ideal=in_ideal,
        monomial_order=request.monomial_order,
    )


def compute_elimination_ideal(
    request: EliminationIdealRequest,
) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.ideals._models import EliminationIdealResult
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialIdeal,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    variables = list(request.ideal.variables)
    payload = {
        "mode": "elimination",
        "variables": variables,
        "eliminated": list(request.eliminated_variables),
        "generators": [
            generator.model_dump(mode="json") for generator in request.ideal.generators
        ],
    }

    # The unbounded lex search runs in a killable worker under the declared
    # wall-time budget; canonicalization and source-bound verification then
    # operate only on the declared output limits.
    try:
        result_payload = _run_sympy_kernel(
            payload, request.resource_budget.wall_seconds
        )
    except _SympyKernelTimeoutError:
        return EliminationIdealResult(
            request=request,
            outcome="TIMEOUT",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the lex Gröbner elimination exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s wall-time budget"
            ),
        )
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
    except _SympyKernelError as error:
        return EliminationIdealResult(
            request=request,
            outcome="ERROR",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the bounded elimination kernel failed without producing "
                f"an exact ideal: {error}"
            ),
        )

    remaining_tuple = tuple(result_payload["remaining"])
    if result_payload["unit_ideal"]:
        one = RationalPolynomial(
            variables=remaining_tuple,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,) * len(remaining_tuple),
                    ),
                )
            ),
        )
        elimination_generators = [one]
    elif not result_payload["generators"]:
        zero = RationalPolynomial(
            variables=remaining_tuple,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        elimination_generators = [zero]
    else:
        elimination_generators = [
            RationalPolynomial.model_validate(item)
            for item in result_payload["generators"]
        ]

    ideal = RationalPolynomialIdeal(
        variables=remaining_tuple,
        generators=tuple(elimination_generators),
    )

    return EliminationIdealResult(
        request=request,
        elimination_ideal=ideal,
        eliminated_variables=tuple(request.eliminated_variables),
    )
