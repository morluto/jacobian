"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import json
from typing import Any, Literal

import sympy
from pydantic_core import PydanticCustomError

from jacobian.canonical import CanonicalizationError, canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.ideals._models import (
    MAX_COEFFICIENT_DIGITS,
    MAX_INPUT_EXPONENT,
    MAX_INPUT_TERMS,
    EliminationIdealResult,
    GroebnerBasisResult,
    IdealComputationBudget,
    IdealMinimalPrimesResult,
    IdealNormalFormResult,
    IdealQuotientResult,
    IdealRadicalMembershipResult,
    IdealRadicalResult,
    IdealSaturationResult,
    _require_computed_minimal_prime_family,
    _require_ideal_budget,
    _require_provable_family_fit,
    _validation_error,
)
from jacobian.math.polynomials.ideals._singular import (
    run_bounded_stdin_python_kernel,
    run_singular_ideal_operation,
    run_singular_minimal_primes,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    require_polynomial_budget,
)


def _run_admission(admission: Any) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.ideal_admission", message=str(exc)
        ) from exc


def _admit_source(ideal: RationalPolynomialIdeal, *, label: str) -> None:
    _require_ideal_budget(ideal, label=label)


def _admit_membership(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="membership polynomial",
    )


def _admit_saturation(
    ideal: RationalPolynomialIdeal, denominator: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    if not denominator.polynomial.terms:
        raise _validation_error("saturation denominator must be nonzero")
    require_polynomial_budget(
        denominator,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="saturation denominator",
    )


def _admit_quotient(
    dividend: RationalPolynomialIdeal, divisor: RationalPolynomialIdeal
) -> None:
    _admit_source(dividend, label="dividend ideal")
    _admit_source(divisor, label="divisor ideal")


def _admit_minimal_primes(ideal: RationalPolynomialIdeal) -> None:
    _admit_source(ideal, label="ideal")
    _require_provable_family_fit(ideal)


def _admit_groebner(ideal: RationalPolynomialIdeal) -> None:
    _admit_source(ideal, label="ideal")


def _admit_normal_form(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    if polynomial.variables != ideal.variables:
        raise _validation_error("polynomial must use the ideal's ordered ring")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="polynomial",
    )


def _admit_elimination(
    ideal: RationalPolynomialIdeal, eliminated_variables: tuple[str, ...]
) -> None:
    _admit_source(ideal, label="ideal")
    eliminated = set(eliminated_variables)
    if any(var not in ideal.variables for var in eliminated):
        raise _validation_error(
            "eliminated variables must be a subset of the ideal's variables"
        )
    if not tuple(v for v in ideal.variables if v not in eliminated):
        raise _validation_error(
            "elimination cannot remove every variable; at least one must remain"
        )


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


def ideal_radical(
    ideal: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealRadicalResult:
    """Compute an exact ideal radical through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_source(ideal, label="ideal"))

    backend = run_singular_ideal_operation(
        "radical",
        ideal,
        None,
        resource_budget,
    )
    return IdealRadicalResult(
        outcome=backend.outcome,
        radical=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def ideal_minimal_primes(
    ideal: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealMinimalPrimesResult:
    """Compute the complete minimal-prime family over ``QQ``."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_minimal_primes(ideal))

    backend = run_singular_minimal_primes(ideal, resource_budget)
    components = backend.components
    if backend.outcome != "COMPUTED" or components is None:
        return IdealMinimalPrimesResult(
            ideal=ideal,
            outcome=backend.outcome,
            components=None,
            backend_version=None,
            detail=backend.detail,
        )

    try:
        _require_computed_minimal_prime_family(ideal, components)
        result = IdealMinimalPrimesResult._from_kernel(
            ideal=ideal,
            components=components,
            backend_version=backend.backend_version,
        )
        _require_transportable_minimal_primes_result(result)
        return result
    except _ResultLimitExceededError as error:
        return IdealMinimalPrimesResult(
            ideal=ideal,
            outcome="LIMIT_EXCEEDED",
            components=None,
            backend_version=None,
            detail=str(error),
        )
    except ValueError:
        return IdealMinimalPrimesResult(
            ideal=ideal,
            outcome="ERROR",
            components=None,
            backend_version=None,
            detail=(
                "The computed minimal-prime family violated its shape, ring, "
                "exact-result envelope, ordering, or uniqueness invariant."
            ),
        )


def ideal_radical_membership(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> IdealRadicalMembershipResult:
    """Decide radical membership by the exact Rabinowitsch criterion."""

    _run_admission(lambda: _admit_membership(ideal, polynomial))

    variable_symbols = symbols_for_variables(ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in ideal.generators
    ]
    polynomial_expr = rational_polynomial_to_sympy(polynomial).as_expr()
    auxiliary = sympy.Dummy("jacobian_rabinowitsch")
    basis = sympy.groebner(
        [*ideal_generators, 1 - auxiliary * polynomial_expr],
        *variable_symbols,
        auxiliary,
        order="grevlex",
        domain=sympy.QQ,
    )
    return IdealRadicalMembershipResult(in_radical=len(basis) == 1 and basis[0] == 1)


def ideal_quotient(
    dividend: RationalPolynomialIdeal,
    divisor: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealQuotientResult:
    """Compute an exact ideal quotient through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_quotient(dividend, divisor))

    backend = run_singular_ideal_operation(
        "quotient",
        dividend,
        divisor,
        resource_budget,
    )
    return IdealQuotientResult(
        outcome=backend.outcome,
        quotient=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def ideal_saturation(
    ideal: RationalPolynomialIdeal,
    denominator: RationalPolynomial,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealSaturationResult:
    """Compute I : <d>^infinity through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_saturation(ideal, denominator))

    denominator_ideal = RationalPolynomialIdeal(
        variables=denominator.variables,
        generators=(denominator,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        ideal,
        denominator_ideal,
        resource_budget,
    )
    return IdealSaturationResult(
        outcome=backend.outcome,
        saturation=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


__all__ = [
    "elimination_ideal",
    "groebner_basis",
    "ideal_minimal_primes",
    "ideal_normal_form",
    "ideal_quotient",
    "ideal_radical",
    "ideal_radical_membership",
    "ideal_saturation",
]


def groebner_basis(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> GroebnerBasisResult:
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_groebner(ideal))
    source_ideal = ideal
    variables = source_ideal.variables
    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map[monomial_order]
    payload = {
        "mode": "groebner",
        "variables": list(variables),
        "order": order,
        "maximum_terms": resource_budget.maximum_output_terms,
        "generators": [
            generator.model_dump(mode="json") for generator in source_ideal.generators
        ],
    }

    # The unbounded search runs in a killable worker under the declared
    # wall-time budget; result assembly then operates only on the declared
    # output limits.
    try:
        result_payload = _run_sympy_kernel(payload, resource_budget.wall_seconds)
    except _SympyKernelTimeoutError:
        return GroebnerBasisResult(
            ideal=ideal,
            outcome="TIMEOUT",
            monomial_order=monomial_order,
            detail=(
                "groebner computation exceeded the enforced "
                f"{resource_budget.wall_seconds}s budget"
            ),
        )
    except _ResultLimitExceededError as error:
        return GroebnerBasisResult(
            ideal=ideal,
            outcome="LIMIT_EXCEEDED",
            monomial_order=monomial_order,
            detail=(
                "the exact reduced Gröbner basis exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return GroebnerBasisResult(
            ideal=ideal,
            outcome="ERROR",
            monomial_order=monomial_order,
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

    basis_ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(basis_generators),
    )

    return GroebnerBasisResult._from_kernel(source_ideal, basis_ideal, monomial_order)


def ideal_normal_form(
    ideal: RationalPolynomialIdeal,
    polynomial: RationalPolynomial,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
) -> IdealNormalFormResult:
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    _run_admission(lambda: _admit_normal_form(ideal, polynomial))
    from jacobian.math.polynomials.ideals._models import IdealNormalFormResult

    payload = {
        "mode": "normal_form",
        "variables": list(ideal.variables),
        "order": monomial_order,
        "generators": [
            generator.model_dump(mode="json") for generator in ideal.generators
        ],
        "polynomial": polynomial.model_dump(mode="json"),
    }

    # A conservative 10-second budget bounds the killable kernel that runs
    # the unbounded Gröbner search; remainder conversion then operates on
    # declared output limits.
    try:
        result_payload = _run_sympy_kernel(payload, 10)
    except _SympyKernelTimeoutError:
        return IdealNormalFormResult(
            ideal=ideal,
            polynomial=polynomial,
            monomial_order=monomial_order,
            outcome="TIMEOUT",
            detail=("the Gröbner reduction exceeded the enforced 10s wall-time bound"),
        )
    except _ResultLimitExceededError as error:
        return IdealNormalFormResult(
            ideal=ideal,
            polynomial=polynomial,
            monomial_order=monomial_order,
            outcome="LIMIT_EXCEEDED",
            detail=(
                "the exact normal form exceeds the declared exact-result "
                f"limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return IdealNormalFormResult(
            ideal=ideal,
            polynomial=polynomial,
            monomial_order=monomial_order,
            outcome="ERROR",
            detail=(
                "the bounded reduction kernel failed without producing an "
                f"exact remainder: {error}"
            ),
        )

    remainder_poly = RationalPolynomial.model_validate(result_payload["remainder"])
    return IdealNormalFormResult._from_kernel(
        ideal, polynomial, monomial_order, remainder_poly
    )


def _elimination_ideal_from_payload(
    result_payload: dict[str, Any],
) -> RationalPolynomialIdeal:
    """Convert one bounded elimination-kernel payload to its canonical ideal."""

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    remaining_tuple = tuple(result_payload["remaining"])
    if result_payload["unit_ideal"]:
        elimination_generators = [
            RationalPolynomial(
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
        ]
    elif not result_payload["generators"]:
        elimination_generators = [
            RationalPolynomial(
                variables=remaining_tuple,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        ]
    else:
        elimination_generators = [
            RationalPolynomial.model_validate(item)
            for item in result_payload["generators"]
        ]
    return RationalPolynomialIdeal(
        variables=remaining_tuple,
        generators=tuple(elimination_generators),
    )


def elimination_ideal(
    ideal: RationalPolynomialIdeal,
    eliminated_variables: tuple[str, ...],
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_elimination(ideal, eliminated_variables))

    variables = list(ideal.variables)
    payload = {
        "mode": "elimination",
        "variables": variables,
        "eliminated": list(eliminated_variables),
        "generators": [
            generator.model_dump(mode="json") for generator in ideal.generators
        ],
    }

    # The unbounded lex search runs in a killable worker under the declared
    # wall-time budget; canonicalization then operates only on the declared
    # output limits.
    try:
        result_payload = _run_sympy_kernel(payload, resource_budget.wall_seconds)
    except _SympyKernelTimeoutError:
        return EliminationIdealResult(
            ideal=ideal,
            outcome="TIMEOUT",
            eliminated_variables=tuple(eliminated_variables),
            detail=(
                "the lex Gröbner elimination exceeded the enforced "
                f"{resource_budget.wall_seconds}s wall-time budget"
            ),
        )
    except _ResultLimitExceededError as error:
        return EliminationIdealResult(
            ideal=ideal,
            outcome="LIMIT_EXCEEDED",
            eliminated_variables=tuple(eliminated_variables),
            detail=(
                "the exact elimination ideal exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return EliminationIdealResult(
            ideal=ideal,
            outcome="ERROR",
            eliminated_variables=tuple(eliminated_variables),
            detail=(
                "the bounded elimination kernel failed without producing "
                f"an exact ideal: {error}"
            ),
        )

    return EliminationIdealResult._from_kernel(
        ideal,
        eliminated_variables,
        _elimination_ideal_from_payload(result_payload),
    )
