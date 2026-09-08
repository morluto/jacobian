"""Standalone SymPy worker for admitted metric-curvature DAG expansion."""

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


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import QQ, Poly, Symbol

    if set(payload) != {"variables", "nodes"}:
        raise ValueError("malformed DAG request")
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
        operation = node.get("operation")
        arguments = node.get("arguments")
        if not isinstance(arguments, list) or any(
            type(argument) is not int or argument < 0 or argument >= index
            for argument in arguments
        ):
            raise ValueError("malformed DAG node")
        if operation == "ZERO":
            result = cache[0]
        elif operation == "ONE":
            result = cache[1]
        elif operation == "SOURCE":
            source = node.get("source")
            if not isinstance(source, list):
                raise ValueError("malformed SOURCE node")
            result = _polynomial(source, generators)
        elif operation == "SCALE":
            scalar = node.get("scalar")
            if (
                not isinstance(scalar, list)
                or len(scalar) != 2
                or not isinstance(scalar[0], str)
                or not isinstance(scalar[1], str)
                or len(arguments) != 1
            ):
                raise ValueError("malformed SCALE node")
            result = cache[arguments[0]].mul_ground(QQ(int(scalar[0]), int(scalar[1])))
        elif operation == "MULTIPLY":
            if len(arguments) != 2:
                raise ValueError("malformed MULTIPLY node")
            result = cache[arguments[0]] * cache[arguments[1]]
        elif operation == "ADD":
            result = sum((cache[argument] for argument in arguments), cache[0])
        elif operation == "DERIVATIVE":
            if len(arguments) != 2:
                raise ValueError("malformed DERIVATIVE node")
            axis = node.get("axis")
            if type(axis) is not int or not 0 <= axis < variable_count:
                raise ValueError("malformed DERIVATIVE node")
            numerator = cache[arguments[0]]
            denominator = cache[arguments[1]]
            result = numerator.diff(axis) * denominator - numerator * denominator.diff(
                axis
            )
        else:
            raise ValueError("unknown DAG operation")
        if index < 2:
            continue
        cache.append(result)
    return {"status": "ok", "values": [_dump(value) for value in cache]}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed DAG request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
