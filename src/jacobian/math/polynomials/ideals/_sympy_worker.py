"""Isolated SymPy worker for exact polynomial-ideal operations."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
)

_PROTOCOL_VERSION = 1
JsonObject = dict[str, Any]


class _WorkerLimitError(ValueError):
    """The exact result exceeds an admitted operation limit."""


@dataclass(frozen=True, slots=True)
class _Context:
    payload: JsonObject
    variables: tuple[str, ...]
    symbols: tuple[Any, ...]
    expressions: tuple[Any, ...]


def _decimal_digit_count(value: int) -> int:
    value = abs(value)
    if value == 0:
        return 1
    estimate = (value.bit_length() * 30_103) // 100_000 + 1
    if value < 10 ** (estimate - 1):
        estimate -= 1
    elif value >= 10**estimate:
        estimate += 1
    return estimate


def _require_admissible_polynomial(
    polynomial: Any,
    *,
    maximum_terms: int,
) -> None:
    if len(polynomial.terms()) > maximum_terms:
        raise _WorkerLimitError(
            f"polynomial result exceeds the {maximum_terms}-term operation budget"
        )
    largest = max(
        (max(monomial) if monomial else 0 for monomial in polynomial.monoms()),
        default=0,
    )
    if largest > MAX_POLYNOMIAL_EXPONENT:
        raise _WorkerLimitError(
            "polynomial result exceeds the "
            f"{MAX_POLYNOMIAL_EXPONENT}-exponent operation budget"
        )
    for _, coefficient in polynomial.terms():
        digits = max(
            _decimal_digit_count(int(coefficient.p)),
            _decimal_digit_count(int(coefficient.q)),
        )
        if digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _WorkerLimitError(
                "polynomial result exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-coefficient-digit operation budget"
            )


def _decode_polynomials(items: Any) -> tuple[RationalPolynomial, ...]:
    if not isinstance(items, list):
        raise ValueError("polynomial family must be a list")
    return tuple(
        RationalPolynomial.model_validate_json(json.dumps(item)) for item in items
    )


def _context(payload: JsonObject) -> _Context:
    raw_variables = payload.get("variables")
    if not isinstance(raw_variables, list) or not all(
        isinstance(variable, str) for variable in raw_variables
    ):
        raise ValueError("variables must be a string list")
    variables = tuple(raw_variables)
    symbols = tuple(symbols_for_variables(variables))
    generators = _decode_polynomials(payload.get("generators"))
    expressions = tuple(
        rational_polynomial_to_sympy(generator).as_expr() for generator in generators
    )
    return _Context(payload, variables, symbols, expressions)


def _dump_polynomial(
    expression: Any,
    variables: tuple[str, ...],
    maximum_terms: int = MAX_POLYNOMIAL_TERMS,
) -> JsonObject:
    polynomial = sympy.Poly(
        expression,
        *symbols_for_variables(variables),
        domain=sympy.QQ,
    )
    _require_admissible_polynomial(polynomial, maximum_terms=maximum_terms)
    converted = rational_polynomial_from_sympy(
        polynomial,
        variables,
        maximum_terms=maximum_terms,
    )
    return converted.model_dump(mode="json")


def _compute_groebner(context: _Context) -> JsonObject:
    maximum_terms = int(context.payload["maximum_terms"])
    basis = sympy.groebner(
        context.expressions,
        *context.symbols,
        order=context.payload["order"],
        domain=sympy.QQ,
    )
    generators: list[JsonObject] = []
    aggregate_terms = 0
    for expression in basis:
        converted = _dump_polynomial(
            expression,
            context.variables,
            maximum_terms,
        )
        aggregate_terms += len(converted["polynomial"]["terms"])
        if aggregate_terms > maximum_terms:
            raise _WorkerLimitError(
                "the reduced Groebner basis exceeds the "
                f"{maximum_terms}-term aggregate operation budget"
            )
        generators.append(converted)
    return {"generators": generators}


def _compute_normal_form(context: _Context) -> JsonObject:
    target = RationalPolynomial.model_validate_json(
        json.dumps(context.payload["polynomial"])
    )
    polynomial_expression = rational_polynomial_to_sympy(target).as_expr()
    order = context.payload.get("order", "grevlex")
    basis = sympy.groebner(
        context.expressions,
        *context.symbols,
        order=order,
        domain=sympy.QQ,
    )
    _, remainder = sympy.reduced(
        polynomial_expression,
        list(basis.exprs),
        *context.symbols,
        order=order,
        domain=sympy.QQ,
    )
    return {"remainder": _dump_polynomial(remainder, context.variables)}


def _containment_ledger(
    context: _Context,
    source_expressions: tuple[Any, ...],
    target_expressions: tuple[Any, ...],
    *,
    maximum_terms: int,
    aggregate_terms: list[int],
) -> JsonObject:
    order = context.payload.get("order", "grevlex")
    basis = sympy.groebner(
        target_expressions,
        *context.symbols,
        order=order,
        domain=sympy.QQ,
    )
    normal_forms: list[JsonObject] = []
    obstruction: int | None = None
    for index, source_expression in enumerate(source_expressions):
        _, remainder = basis.reduce(source_expression)
        converted = _dump_polynomial(
            remainder,
            context.variables,
            maximum_terms,
        )
        aggregate_terms[0] += len(converted["polynomial"]["terms"])
        if aggregate_terms[0] > maximum_terms:
            raise _WorkerLimitError(
                "the containment ledger exceeds the "
                f"{maximum_terms}-term aggregate operation budget"
            )
        normal_forms.append(converted)
        if remainder != 0:
            obstruction = index
            break
    return {
        "contained": obstruction is None,
        "normal_forms": normal_forms,
        "first_obstruction_index": obstruction,
    }


def _compute_ideal_relation(context: _Context) -> JsonObject:
    right_generators = _decode_polynomials(context.payload["right_generators"])
    right_expressions = tuple(
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in right_generators
    )
    maximum_terms = int(context.payload["maximum_terms"])
    aggregate_terms = [0]
    response = {
        "left_in_right": _containment_ledger(
            context,
            context.expressions,
            right_expressions,
            maximum_terms=maximum_terms,
            aggregate_terms=aggregate_terms,
        )
    }
    if context.payload["mutual"]:
        response["right_in_left"] = _containment_ledger(
            context,
            right_expressions,
            context.expressions,
            maximum_terms=maximum_terms,
            aggregate_terms=aggregate_terms,
        )
    return response


def _verification_failure(detail: str) -> JsonObject:
    return {"equal": False, "detail": detail}


def _monomial(symbols: tuple[Any, ...], exponents: tuple[int, ...]) -> Any:
    product = sympy.Integer(1)
    for symbol, exponent in zip(symbols, exponents, strict=True):
        if exponent:
            product *= symbol**exponent
    return product


def _check_reduced_basis(
    expressions: tuple[Any, ...],
    symbols: tuple[Any, ...],
    order: str,
) -> JsonObject | None:
    leading_terms = [
        sympy.LT(expression, *symbols, order=order) for expression in expressions
    ]
    for index, expression in enumerate(expressions):
        if sympy.LC(expression, *symbols, order=order) != 1:
            return _verification_failure(
                "a reduced Groebner basis has unit leading coefficients"
            )
        others = [
            leading_term
            for other_index, leading_term in enumerate(leading_terms)
            if other_index != index
        ]
        if others:
            _, remainder = sympy.reduced(
                expression,
                others,
                *symbols,
                order=order,
                domain=sympy.QQ,
            )
            if remainder != expression:
                return _verification_failure(
                    "reduced Groebner basis generators must contain no other "
                    "leading monomial"
                )
    return None


def _check_s_polynomials(
    expressions: tuple[Any, ...],
    symbols: tuple[Any, ...],
    order: str,
) -> JsonObject | None:
    polynomials = [
        sympy.Poly(expression, *symbols, domain=sympy.QQ) for expression in expressions
    ]
    monomial_order = getattr(sympy.polys.orderings, order)
    leading_exponents = [
        polynomial.monoms(order=monomial_order)[0] for polynomial in polynomials
    ]
    for first in range(len(expressions)):
        for second in range(first + 1, len(expressions)):
            common = tuple(
                max(left, right)
                for left, right in zip(
                    leading_exponents[first],
                    leading_exponents[second],
                    strict=True,
                )
            )
            first_multiplier = tuple(
                value - source
                for value, source in zip(
                    common,
                    leading_exponents[first],
                    strict=True,
                )
            )
            second_multiplier = tuple(
                value - source
                for value, source in zip(
                    common,
                    leading_exponents[second],
                    strict=True,
                )
            )
            s_polynomial = expressions[first] * _monomial(
                symbols, first_multiplier
            ) - expressions[second] * _monomial(symbols, second_multiplier)
            _, remainder = sympy.reduced(
                s_polynomial,
                expressions,
                *symbols,
                order=order,
                domain=sympy.QQ,
            )
            if remainder != 0:
                return _verification_failure(
                    "basis S-polynomials must reduce to zero; the list is not a "
                    "Groebner basis of the retained ideal"
                )
    return None


def _check_same_ideal(
    context: _Context,
    claimed_expressions: tuple[Any, ...],
    order: str,
) -> JsonObject | None:
    source_basis = sympy.groebner(
        context.expressions,
        *context.symbols,
        order=order,
        domain=sympy.QQ,
    )
    for generator in claimed_expressions:
        _, remainder = sympy.reduced(
            generator,
            list(source_basis.exprs),
            *context.symbols,
            order=order,
            domain=sympy.QQ,
        )
        if remainder != 0:
            return _verification_failure(
                "a basis generator leaves a nonzero remainder modulo the source ideal"
            )
    for expression in context.expressions:
        if expression.is_zero:
            continue
        _, remainder = sympy.reduced(
            expression,
            claimed_expressions,
            *context.symbols,
            order=order,
            domain=sympy.QQ,
        )
        if remainder != 0:
            return _verification_failure(
                "a source generator is not contained in the claimed basis ideal"
            )
    return None


def _compute_verify_groebner_basis(context: _Context) -> JsonObject:
    order = str(context.payload.get("order", "grevlex"))
    claimed = _decode_polynomials(context.payload["basis"])
    claimed_expressions = tuple(
        rational_polynomial_to_sympy(generator).as_expr() for generator in claimed
    )
    nonzero = tuple(
        expression for expression in claimed_expressions if not expression.is_zero
    )
    return (
        _check_reduced_basis(nonzero, context.symbols, order)
        or _check_s_polynomials(nonzero, context.symbols, order)
        or _check_same_ideal(context, claimed_expressions, order)
        or {"equal": True}
    )


def _compute_elimination(context: _Context) -> JsonObject:
    eliminated = set(context.payload["eliminated"])
    remaining = [
        variable for variable in context.variables if variable not in eliminated
    ]
    ordered = tuple(
        variable for variable in context.variables if variable in eliminated
    ) + tuple(remaining)
    ordered_symbols = tuple(symbols_for_variables(ordered))
    basis = sympy.groebner(
        context.expressions,
        *ordered_symbols,
        order="lex",
        domain=sympy.QQ,
    )
    generators: list[JsonObject] = []
    unit_ideal = False
    for expression in basis:
        polynomial = sympy.Poly(expression, *ordered_symbols, domain=sympy.QQ)
        involved = {str(symbol) for symbol in polynomial.free_symbols}
        if not involved:
            unit_ideal = True
            break
        if involved.issubset(remaining):
            generators.append(_dump_polynomial(expression, tuple(remaining)))
    return {
        "unit_ideal": unit_ideal,
        "generators": generators,
        "remaining": remaining,
    }


_HANDLERS: dict[str, Callable[[_Context], JsonObject]] = {
    "groebner": _compute_groebner,
    "normal_form": _compute_normal_form,
    "ideal_relation": _compute_ideal_relation,
    "verify_groebner_basis": _compute_verify_groebner_basis,
    "elimination": _compute_elimination,
}


def _canonical_request_bytes(payload: JsonObject) -> bytes:
    return json.dumps(
        {"protocol_version": _PROTOCOL_VERSION, "payload": payload},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _decode_request() -> tuple[str, JsonObject]:
    request = json.loads(sys.stdin.buffer.read().decode("ascii"))
    if not isinstance(request, dict):
        raise ValueError("worker request must be an object")
    if request.get("protocol_version") != _PROTOCOL_VERSION:
        raise ValueError("unsupported worker protocol version")
    digest = request.get("request_digest")
    payload = request.get("payload")
    if not isinstance(digest, str) or not isinstance(payload, dict):
        raise ValueError("malformed worker request")
    expected = hashlib.sha256(_canonical_request_bytes(payload)).hexdigest()
    if digest != expected:
        raise ValueError("worker request digest mismatch")
    return digest, payload


def _emit(digest: str, status: str, result: JsonObject) -> None:
    response = {
        "protocol_version": _PROTOCOL_VERSION,
        "request_digest": digest,
        "status": status,
        **result,
    }
    json.dump(response, sys.stdout, ensure_ascii=True, separators=(",", ":"))
    sys.stdout.flush()


def main() -> None:
    digest = ""
    try:
        digest, payload = _decode_request()
        mode = payload.get("mode")
        if not isinstance(mode, str) or mode not in _HANDLERS:
            raise ValueError("unknown kernel mode")
        _emit(digest, "ok", _HANDLERS[mode](_context(payload)))
    except _WorkerLimitError as error:
        _emit(digest, "limit", {"detail": str(error)})
    except Exception as error:
        _emit(digest, "error", {"detail": str(error)})


if __name__ == "__main__":
    main()
