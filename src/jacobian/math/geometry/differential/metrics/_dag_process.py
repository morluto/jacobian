"""Killable SymPy expansion for an admitted metric-curvature polynomial DAG."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.math.geometry.differential.metrics._dag import Node
from jacobian.math.polynomials._conversions import symbols_for_variables
from jacobian.math.polynomials.values import SparseRationalPolynomial
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_dag_worker.py")
_STDOUT_BYTES = 64 * 1024 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 0.05


def _source_payload(polynomial: SparseRationalPolynomial) -> list[list[object]]:
    return [
        [*term.exponents, str(term.coefficient.num), str(term.coefficient.den)]
        for term in polynomial.terms
    ]


def _node_payload(node: Node) -> dict[str, object]:
    payload: dict[str, object] = {
        "operation": node.operation,
        "arguments": list(node.arguments),
    }
    if node.source is not None:
        payload["source"] = _source_payload(node.source)
    if node.operation == "SCALE":
        payload["scalar"] = [
            str(node.scalar.numerator),
            str(node.scalar.denominator),
        ]
    if node.operation == "DERIVATIVE":
        payload["axis"] = node.axis
    return payload


def _poly_from_payload(records: object, symbols: tuple[Any, ...]) -> Any:
    from sympy import QQ, Poly, Rational

    if not isinstance(records, list):
        raise ValueError("malformed expanded polynomial")
    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed expanded polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError("malformed expanded polynomial")
        numerator, denominator = record[-2], record[-1]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("malformed expanded polynomial")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def evaluate_polynomial_dag(
    nodes: list[Node], axis: tuple[str, ...], *, deadline: float
) -> tuple[list[Any], tuple[Any, ...]]:
    """Expand one admitted DAG in a killable worker under the shared deadline.

    Expanded polynomials remain as worker term dumps. The parent materializes
    only the nodes later consumed, under the remaining request deadline.
    """

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "metric curvature deadline expired before DAG expansion"
        )
    payload = encode_strict_json(
        {
            "variables": list(axis),
            "nodes": [_node_payload(node) for node in nodes],
        }
    )
    try:
        with TemporaryDirectory(prefix="jacobian-metric-dag-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_BYTES,
                stderr_limit=_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_STDOUT_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded metric-curvature DAG worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "metric curvature cancelled during polynomial DAG expansion"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "metric curvature deadline expired during polynomial DAG expansion"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded metric-curvature DAG worker did not return expanded polynomials"
        )
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_STDOUT_BYTES,
                max_output_bytes=_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(
            "bounded metric-curvature DAG worker returned malformed output"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or not isinstance(response.get("values"), list)
        or len(response["values"]) != len(nodes)
    ):
        raise RuntimeError(
            "bounded metric-curvature DAG worker returned malformed output"
        )
    generators = symbols_for_variables(axis)
    return response["values"], generators


def materialize_expanded_polynomial(
    records: object, symbols: tuple[Any, ...], *, deadline: float
) -> Any:
    """Decode one worker polynomial under the remaining request deadline."""

    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "metric curvature deadline expired during polynomial DAG decoding"
        )
    return _poly_from_payload(records, symbols)
