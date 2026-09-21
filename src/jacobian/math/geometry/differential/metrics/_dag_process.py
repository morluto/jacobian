"""Bounded parent adapter for admitted metric-expression DAGs."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, Literal

from sympy import QQ, Poly, Rational

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.geometry.differential.metrics._dag import Node
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_PROTOCOL_VERSION = 1
_WORKER_PATH = Path(__file__).resolve().with_name("_dag_worker.py")
_STDOUT_BYTES = 64 * 1024 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DagEvaluation:
    fractions: tuple[RationalFunction, ...]
    determinants: tuple[SparseRationalPolynomial, ...]


@dataclass(frozen=True, slots=True)
class DagDegeneracy:
    kind: Literal["singular", "undefined"]


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
        payload["scalar"] = [str(node.scalar.numerator), str(node.scalar.denominator)]
    if node.operation == "DERIVATIVE":
        payload["axis"] = node.axis
    return payload


def _canonical_request_bytes(request: dict[str, object]) -> bytes:
    return encode_strict_json(
        {"protocol_version": _PROTOCOL_VERSION, "request": request}
    )


def _encode_request(request: dict[str, object]) -> tuple[str, bytes]:
    digest = hashlib.sha256(_canonical_request_bytes(request)).hexdigest()
    return digest, encode_strict_json(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "request_digest": digest,
            "request": request,
        }
    )


def _require_deadline(deadline: float, owner: str, stage: str) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(f"{owner} deadline expired {stage}")
    return remaining


def _run_worker(payload: bytes, *, deadline: float, owner: str) -> object:
    remaining = _require_deadline(deadline, owner, "before DAG execution")

    def checkpoint() -> None:
        _require_deadline(deadline, owner, "during worker output decode")

    try:
        with TemporaryDirectory(
            prefix=f"jacobian-{owner.replace(' ', '-')}-"
        ) as worker_directory:
            return run_checked_worker_process(
                [sys.executable, "-I", str(_WORKER_PATH)],
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
                decode_result=lambda value: value,
                checkpoint=checkpoint,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(f"bounded {owner} worker could not be started") from exc


def _decode_response(
    response: object,
    *,
    expected_digest: str,
    owner: str,
    deadline: float,
) -> dict[str, object]:
    request_checkpoint(f"after {owner} worker output decode")
    _require_deadline(deadline, owner, "during worker output decode")
    if not isinstance(response, dict):
        raise RuntimeError(f"bounded {owner} worker returned malformed output")
    if response.get("protocol_version") != _PROTOCOL_VERSION:
        raise RuntimeError(f"bounded {owner} worker returned an unknown protocol")
    if response.get("request_digest") != expected_digest:
        raise RuntimeError(f"bounded {owner} worker returned an unbound result")
    return response


def _polynomial_from_payload(
    records: object,
    symbols: tuple[Any, ...],
    *,
    owner: str,
    deadline: float,
) -> Any:
    if not isinstance(records, list):
        raise ValueError("malformed cancelled polynomial")
    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
    for index, record in enumerate(records):
        if index % 256 == 0:
            request_checkpoint(f"during {owner} polynomial decode")
            _require_deadline(deadline, owner, "during polynomial decode")
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed cancelled polynomial")
        exponents = tuple(record[:variable_count])
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            raise ValueError("malformed cancelled polynomial")
        numerator, denominator = record[-2:]
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("malformed cancelled polynomial")
        numerator_value = parse_canonical_integer(numerator)
        denominator_value = parse_canonical_integer(denominator)
        if denominator_value <= 0 or exponents in coefficients:
            raise ValueError("malformed cancelled polynomial")
        coefficients[exponents] = Rational(numerator_value, denominator_value)
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def _decode_evaluation(
    response: dict[str, object],
    *,
    axis: tuple[str, ...],
    fraction_count: int,
    determinant_count: int,
    owner: str,
    deadline: float,
) -> DagEvaluation | DagDegeneracy:
    status = response.get("status")
    if status in ("singular", "undefined"):
        if set(response) != {"protocol_version", "request_digest", "status"}:
            raise RuntimeError(f"bounded {owner} worker returned malformed output")
        return DagDegeneracy(status)
    fractions = response.get("fractions")
    determinants = response.get("determinants")
    if (
        status != "ok"
        or set(response)
        != {
            "protocol_version",
            "request_digest",
            "status",
            "fractions",
            "determinants",
        }
        or not isinstance(fractions, list)
        or len(fractions) != fraction_count
        or not isinstance(determinants, list)
        or len(determinants) != determinant_count
    ):
        raise RuntimeError(f"bounded {owner} worker returned malformed output")
    symbols = symbols_for_variables(axis)
    components: list[RationalFunction] = []
    for record in fractions:
        request_checkpoint(f"before {owner} result decoding")
        if not isinstance(record, dict) or set(record) != {
            "numerator",
            "denominator",
        }:
            raise RuntimeError(f"bounded {owner} worker returned malformed output")
        numerator = _polynomial_from_payload(
            record["numerator"], symbols, owner=owner, deadline=deadline
        )
        denominator = _polynomial_from_payload(
            record["denominator"], symbols, owner=owner, deadline=deadline
        )
        if denominator.is_zero:
            raise RuntimeError(f"bounded {owner} worker returned a zero denominator")
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
            _polynomial_from_payload(
                record,
                symbols,
                owner=owner,
                deadline=deadline,
            ).monic(),
            axis,
            maximum_terms=256,
        )
        for record in determinants
    )
    return DagEvaluation(tuple(components), guards)


def evaluate_admitted_dag(
    nodes: Sequence[Node],
    axis: tuple[str, ...],
    *,
    fractions: Sequence[tuple[int, int]],
    determinants: Sequence[int],
    deadline: float,
    owner: str,
    undefined_numerators: Sequence[int] = (),
) -> DagEvaluation | DagDegeneracy:
    """Evaluate one admitted DAG through the sole bounded metric worker."""

    _require_deadline(deadline, owner, "before DAG payload encoding")
    request_checkpoint(f"before {owner} payload encoding")
    request = {
        "variables": list(axis),
        "nodes": [_node_payload(node) for node in nodes],
        "fractions": [list(pair) for pair in fractions],
        "determinants": list(determinants),
        "undefined_numerators": list(undefined_numerators),
    }
    digest, payload = _encode_request(request)
    request_checkpoint(f"after {owner} payload encoding")
    _require_deadline(deadline, owner, "after DAG payload encoding")
    response = _decode_response(
        _run_worker(payload, deadline=deadline, owner=owner),
        expected_digest=digest,
        owner=owner,
        deadline=deadline,
    )
    return _decode_evaluation(
        response,
        axis=axis,
        fraction_count=len(fractions),
        determinant_count=len(determinants),
        owner=owner,
        deadline=deadline,
    )


__all__ = ["DagDegeneracy", "DagEvaluation", "evaluate_admitted_dag"]
