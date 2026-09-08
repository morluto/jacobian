"""Standalone SymPy worker for admitted covariant-derivative DAG arithmetic."""

from __future__ import annotations

import json
import sys
from typing import Any


def _polynomial(records: list[Any], symbols: tuple[Any, ...]) -> Any:
    from sympy import QQ, Poly, Rational

    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed polynomial record")
        exponents = tuple(record[:variable_count])
        numerator = record[-2]
        denominator = record[-1]
        if (
            any(type(exponent) is not int or exponent < 0 for exponent in exponents)
            or not isinstance(numerator, str)
            or not isinstance(denominator, str)
        ):
            raise ValueError("malformed polynomial record")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def _dump(polynomial: Any) -> list[list[Any]]:
    return [
        [*exponents, str(coefficient.p), str(coefficient.q)]
        for exponents, coefficient in polynomial.terms()
    ]


def _cancel(numerator: Any, denominator: Any) -> tuple[Any, Any]:
    from sympy import Poly

    if not numerator.is_zero:
        numerator_terms, denominator_terms = numerator.terms(), denominator.terms()
        common = tuple(
            min(
                min(exponents[axis] for exponents, _ in numerator_terms),
                min(exponents[axis] for exponents, _ in denominator_terms),
            )
            for axis in range(len(numerator.gens))
        )
        if any(common):

            def divide(value: Any) -> Any:
                return Poly.from_dict(
                    {
                        tuple(
                            exponent - shift
                            for exponent, shift in zip(exponents, common, strict=True)
                        ): coefficient
                        for exponents, coefficient in value.terms()
                    },
                    value.gens,
                    domain=value.domain,
                )

            numerator, denominator = divide(numerator), divide(denominator)
    numerator, denominator = numerator.cancel(denominator, include=True)
    leading = denominator.LC()
    return numerator.mul_ground(1 / leading), denominator.mul_ground(1 / leading)


def _apply_operation(
    operation: object,
    arguments: list[int],
    node: dict[str, Any],
    cache: list[Any],
    generators: tuple[Any, ...],
    variable_count: int,
) -> Any:
    from sympy import QQ

    if operation == "ZERO":
        return cache[0]
    if operation == "ONE":
        return cache[1]
    if operation == "SOURCE":
        source = node.get("source")
        if not isinstance(source, list):
            raise ValueError("malformed SOURCE node")
        return _polynomial(source, generators)
    if operation == "SCALE":
        scalar = node.get("scalar")
        if (
            not isinstance(scalar, list)
            or len(scalar) != 2
            or not isinstance(scalar[0], str)
            or not isinstance(scalar[1], str)
            or len(arguments) != 1
        ):
            raise ValueError("malformed SCALE node")
        return cache[arguments[0]].mul_ground(QQ(int(scalar[0]), int(scalar[1])))
    if operation == "MULTIPLY":
        if len(arguments) != 2:
            raise ValueError("malformed MULTIPLY node")
        return cache[arguments[0]] * cache[arguments[1]]
    if operation == "ADD":
        return sum((cache[argument] for argument in arguments), cache[0])
    if operation == "DERIVATIVE":
        if len(arguments) != 2:
            raise ValueError("malformed DERIVATIVE node")
        axis = node.get("axis")
        if type(axis) is not int or not 0 <= axis < variable_count:
            raise ValueError("malformed DERIVATIVE node")
        numerator = cache[arguments[0]]
        denominator = cache[arguments[1]]
        return numerator.diff(axis) * denominator - numerator * denominator.diff(axis)
    raise ValueError("unknown DAG operation")


def _expand_nodes(payload: dict[str, Any]) -> tuple[list[Any], int]:
    from sympy import QQ, Poly, Symbol

    variables = payload["variables"]
    nodes = payload["nodes"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(name, str) or not name for name in variables)
        or not isinstance(nodes, list)
        or len(nodes) < 2
    ):
        raise ValueError("malformed DAG request")
    variable_count = len(variables)
    generators = tuple(Symbol(name) for name in variables)
    cache: list[Any] = [Poly(0, *generators, domain=QQ), Poly(1, *generators, domain=QQ)]
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError("malformed DAG node")
        arguments = node.get("arguments")
        if not isinstance(arguments, list) or any(
            type(argument) is not int or argument < 0 or argument >= index
            for argument in arguments
        ):
            raise ValueError("malformed DAG node")
        result = _apply_operation(
            node.get("operation"),
            arguments,
            node,
            cache,
            generators,
            variable_count,
        )
        if index < 2:
            continue
        cache.append(result)
    return cache, variable_count


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"variables", "nodes", "fractions", "determinants"}:
        raise ValueError("malformed covariant-derivative request")
    fractions = payload["fractions"]
    determinants = payload["determinants"]
    cache, _variable_count = _expand_nodes(payload)
    if not isinstance(determinants, list) or any(
        type(index) is not int or index < 0 or index >= len(cache)
        for index in determinants
    ):
        raise ValueError("malformed determinant request")
    for index in determinants:
        if cache[index].is_zero:
            return {"status": "singular"}
    if not isinstance(fractions, list):
        raise ValueError("malformed fraction request")
    cancelled: list[dict[str, Any]] = []
    for pair in fractions:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                type(index) is not int or index < 0 or index >= len(cache)
                for index in pair
            )
        ):
            raise ValueError("malformed fraction request")
        numerator, denominator = _cancel(cache[pair[0]], cache[pair[1]])
        cancelled.append(
            {"numerator": _dump(numerator), "denominator": _dump(denominator)}
        )
    return {
        "status": "ok",
        "fractions": cancelled,
        "determinants": [_dump(cache[index]) for index in determinants],
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed covariant-derivative request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
