"""Killable SymPy expansion, recognition, and cancellation for metric DAGs."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from sympy import QQ, Poly, Rational

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag import Node
from jacobian.math.geometry.differential.metrics._plan import singular
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial
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


@dataclass(frozen=True, slots=True)
class RationalDagWorkerMessages:
    timeout_before: str
    timeout_after: str
    timeout_during: str
    cancelled_during: str
    start_failure: str
    malformed: str
    directory_prefix: str
    checkpoint_prefix: str
    noncanonical_location: tuple[str, ...]
    noncanonical_code: str
    noncanonical_message: str


def _source_payload(polynomial: SparseRationalPolynomial) -> list[list[object]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
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


def _poly_from_payload(records: object, symbols: tuple[Any, ...], *, kind: str) -> Any:
    if not isinstance(records, list):
        raise ValueError(f"malformed {kind} polynomial")
    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError(f"malformed {kind} polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError(f"malformed {kind} polynomial")
        numerator, denominator = record[-2], record[-1]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError(f"malformed {kind} polynomial")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def _run_worker(
    payload: bytes,
    *,
    deadline: float,
    directory_prefix: str,
    timeout_during: str,
    cancelled_during: str,
    start_failure: str,
    malformed: str,
    missing_output: str,
) -> object:
    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(timeout_during)
    try:
        with TemporaryDirectory(prefix=directory_prefix) as worker_directory:
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
        raise RuntimeError(start_failure) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(cancelled_during)
    if completed.timed_out:
        raise OperationExecutionTimeoutError(timeout_during)
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(missing_output)
    try:
        return loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_STDOUT_BYTES,
                max_output_bytes=_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(malformed) from exc


def evaluate_admitted_rational_dag(
    nodes: list[Node],
    fraction_pairs: list[list[int]],
    determinant_indices: list[int],
    axis: tuple[str, ...],
    sources: tuple[RationalFunction, ...],
    *,
    deadline: float,
    messages: RationalDagWorkerMessages,
) -> tuple[tuple[RationalFunction, ...], tuple[Any, ...]]:
    """Expand, recognize, and cancel an admitted DAG in one killable worker."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(messages.timeout_before)
    request_checkpoint(f"before {messages.checkpoint_prefix} payload encoding")
    payload = encode_strict_json(
        {
            "variables": list(axis),
            "nodes": [_node_payload(node) for node in nodes],
            "fractions": fraction_pairs,
            "determinants": determinant_indices,
            "sources": [
                {
                    "numerator": _source_payload(component.numerator),
                    "denominator": _source_payload(component.denominator),
                }
                for component in sources
            ],
        }
    )
    request_checkpoint(f"after {messages.checkpoint_prefix} payload encoding")
    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(messages.timeout_after)
    response = _run_worker(
        payload,
        deadline=deadline,
        directory_prefix=messages.directory_prefix,
        timeout_during=messages.timeout_during,
        cancelled_during=messages.cancelled_during,
        start_failure=messages.start_failure,
        malformed=messages.malformed,
        missing_output=messages.malformed,
    )
    if not isinstance(response, dict):
        raise RuntimeError(messages.malformed)
    if response.get("status") == "singular":
        raise singular()
    if response.get("status") == "noncanonical":
        raise OperationDomainValidationError(
            location=messages.noncanonical_location,
            code=messages.noncanonical_code,
            message=messages.noncanonical_message,
        )
    if (
        response.get("status") != "ok"
        or not isinstance(response.get("fractions"), list)
        or len(response["fractions"]) != len(fraction_pairs)
        or not isinstance(response.get("determinants"), list)
        or len(response["determinants"]) != len(determinant_indices)
    ):
        raise RuntimeError(messages.malformed)
    symbols = symbols_for_variables(axis)
    components = []
    for record in response["fractions"]:
        request_checkpoint(f"before {messages.checkpoint_prefix} result decoding")
        if (
            not isinstance(record, dict)
            or "numerator" not in record
            or "denominator" not in record
        ):
            raise RuntimeError(messages.malformed)
        numerator = _poly_from_payload(record["numerator"], symbols, kind="cancelled")
        denominator = _poly_from_payload(
            record["denominator"], symbols, kind="cancelled"
        )
        components.append(
            RationalFunction(
                variables=axis,
                numerator=sparse_rational_polynomial_from_sympy(
                    numerator, axis, maximum_terms=256
                ),
                denominator=sparse_rational_polynomial_from_sympy(
                    denominator, axis, maximum_terms=256
                ),
            )
        )
    guards = tuple(
        sparse_rational_polynomial_from_sympy(
            _poly_from_payload(record, symbols, kind="cancelled").monic(),
            axis,
            maximum_terms=256,
        )
        for record in response["determinants"]
    )
    return tuple(components), guards


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
    response = _run_worker(
        payload,
        deadline=deadline,
        directory_prefix="jacobian-metric-dag-",
        timeout_during=(
            "metric curvature deadline expired during polynomial DAG expansion"
        ),
        cancelled_during="metric curvature cancelled during polynomial DAG expansion",
        start_failure="bounded metric-curvature DAG worker could not be started",
        malformed="bounded metric-curvature DAG worker returned malformed output",
        missing_output=(
            "bounded metric-curvature DAG worker did not return expanded polynomials"
        ),
    )
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
    return _poly_from_payload(records, symbols, kind="expanded")


__all__ = [
    "RationalDagWorkerMessages",
    "evaluate_admitted_rational_dag",
    "evaluate_polynomial_dag",
    "materialize_expanded_polynomial",
]
