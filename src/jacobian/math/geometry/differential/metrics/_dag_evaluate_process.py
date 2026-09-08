"""Killable SymPy expansion and cancellation for admitted metric DAGs."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
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
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_dag_evaluate_worker.py")
_STDOUT_BYTES = 64 * 1024 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0


def _source_payload(polynomial: Any) -> list[list[object]]:
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


def _poly_from_payload(records: object, symbols: tuple[Any, ...]) -> Any:
    if not isinstance(records, list):
        raise ValueError("malformed cancelled polynomial")
    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed cancelled polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError("malformed cancelled polynomial")
        numerator, denominator = record[-2], record[-1]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("malformed cancelled polynomial")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def _load_worker_response(
    stdout: bytes, *, owner: str, deadline: float
) -> object:
    request_checkpoint(f"before {owner} worker output decode")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired during worker output decode"
        )
    try:
        response = loads_strict_json(
            stdout,
            limits=CanonicalLimits(
                max_input_bytes=_STDOUT_BYTES,
                max_output_bytes=_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(f"bounded {owner} worker returned malformed output") from exc
    request_checkpoint(f"after {owner} worker output decode")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired during worker output decode"
        )
    return response


def evaluate_admitted_dag(
    nodes: Sequence[Node],
    axis: tuple[str, ...],
    *,
    fractions: Sequence[tuple[int, int]],
    determinants: Sequence[int],
    sources: Sequence[RationalFunction],
    deadline: float,
    owner: str,
    singular_metric: Callable[[], OperationDomainValidationError],
    noncanonical_location: tuple[str, ...] = (),
    noncanonical_code: str = "",
    noncanonical_message: str = "",
) -> tuple[tuple[RationalFunction, ...], tuple[Any, ...]]:
    """Expand and cancel an admitted DAG in one killable worker."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired before DAG expansion"
        )
    fraction_pairs = [list(pair) for pair in fractions]
    determinant_indices = list(determinants)
    request_checkpoint(f"before {owner} payload encoding")
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
    request_checkpoint(f"after {owner} payload encoding")
    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired after payload encoding"
        )
    try:
        with TemporaryDirectory(
            prefix=f"jacobian-{owner.replace(' ', '-')}-"
        ) as worker_directory:
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
        raise RuntimeError(f"bounded {owner} worker could not be started") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            f"{owner} cancelled during polynomial DAG expansion"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            f"{owner} deadline expired during polynomial DAG expansion"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(f"bounded {owner} worker did not return cancelled fractions")
    response = _load_worker_response(completed.stdout, owner=owner, deadline=deadline)
    if not isinstance(response, dict):
        raise RuntimeError(f"bounded {owner} worker returned malformed output")
    if response.get("status") == "singular":
        raise singular_metric()
    if response.get("status") == "noncanonical":
        raise OperationDomainValidationError(
            location=noncanonical_location,
            code=noncanonical_code,
            message=noncanonical_message,
        )
    if (
        response.get("status") != "ok"
        or not isinstance(response.get("fractions"), list)
        or len(response["fractions"]) != len(fraction_pairs)
        or not isinstance(response.get("determinants"), list)
        or len(response["determinants"]) != len(determinant_indices)
    ):
        raise RuntimeError(f"bounded {owner} worker returned malformed output")
    symbols = symbols_for_variables(axis)
    components = []
    for record in response["fractions"]:
        request_checkpoint(f"before {owner} result decoding")
        if (
            not isinstance(record, dict)
            or "numerator" not in record
            or "denominator" not in record
        ):
            raise RuntimeError(f"bounded {owner} worker returned malformed output")
        numerator = _poly_from_payload(record["numerator"], symbols)
        denominator = _poly_from_payload(record["denominator"], symbols)
        components.append(
            RationalFunction._from_kernel(
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
            _poly_from_payload(record, symbols).monic(),
            axis,
            maximum_terms=256,
        )
        for record in response["determinants"]
    )
    return tuple(components), guards
