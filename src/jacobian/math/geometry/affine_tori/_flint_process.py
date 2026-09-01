"""Killable python-flint process boundary for affine-torus fixed loci."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.geometry.affine_tori._bounds import (
    AffineTorusFixedLocusPlan,
    require_affine_torus_deadline,
)
from jacobian.math.geometry.affine_tori._kernel_types import (
    EmptyFixedLocusKernel,
    FixedLocusKernel,
    NonemptyFixedLocusKernel,
)
from jacobian.math.geometry.affine_tori.values import (
    MAX_AFFINE_TORUS_POINT_DIGITS,
    RationalAffineTorusMap,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_AFFINE_TORUS_WORKER = Path(__file__).resolve().with_name("_flint_worker.py")
_PROTOCOL_VERSION = 1
_WORKER_STDIN_LIMIT = 64 * 1024
_WORKER_STDERR_LIMIT = 64 * 1024
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0


def _positive_worker_allowance(deadline: float) -> float:
    """Reserve bounded parent decoding and result construction time."""

    require_affine_torus_deadline(deadline, "before launching the FLINT worker")
    remaining = deadline - monotonic()
    worker_allowance = remaining - _PARENT_FINALIZATION_SECONDS
    if worker_allowance <= 0:
        raise OperationExecutionTimeoutError(
            "affine-torus fixed-locus deadline has no worker execution allowance"
        )
    return worker_allowance


def _worker_input(source: RationalAffineTorusMap) -> bytes:
    payload = {
        "protocol_version": _PROTOCOL_VERSION,
        "dimension": source.torus.dimension,
        "linear_part": [list(row) for row in source.linear_part.entries],
        "translation": [
            coordinate.model_dump(mode="json")
            for coordinate in source.translation.coordinates
        ],
    }
    return encode_strict_json(payload)


def _strict_integer(value: Any, *, maximum_digits: int, positive: bool = False) -> int:
    if not isinstance(value, str) or len(value.lstrip("-")) > maximum_digits:
        raise ValueError("worker result integer exceeded its admitted digit bound")
    parsed = parse_canonical_integer(value)
    if value != format_canonical_integer(parsed) or (positive and parsed <= 0):
        raise ValueError("worker result integer is not canonical")
    return parsed


def _strict_fraction(value: Any, *, maximum_digits: int) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("worker result rational has invalid fields")
    numerator = _strict_integer(value["num"], maximum_digits=maximum_digits)
    denominator = _strict_integer(
        value["den"], maximum_digits=maximum_digits, positive=True
    )
    result = Fraction(numerator, denominator)
    if (
        value["num"] != format_canonical_integer(result.numerator)
        or value["den"] != format_canonical_integer(result.denominator)
        or not 0 <= result < 1
    ):
        raise ValueError("worker result rational is not canonical modulo one")
    return result


def _strict_integer_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    maximum_digits: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("worker result matrix has an invalid row count")
    result: list[tuple[int, ...]] = []
    for candidate_row in value:
        if not isinstance(candidate_row, list) or len(candidate_row) != columns:
            raise ValueError("worker result matrix has an invalid column count")
        result.append(
            tuple(
                _strict_integer(entry, maximum_digits=maximum_digits)
                for entry in candidate_row
            )
        )
    return tuple(result)


def _strict_fraction_vector(
    value: Any, *, length: int, maximum_digits: int
) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("worker result point has an invalid dimension")
    return tuple(
        _strict_fraction(entry, maximum_digits=maximum_digits) for entry in value
    )


def _strict_dimension(value: Any, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= maximum
    ):
        raise ValueError("worker result dimension is invalid")
    return value


def _decode_worker_projection(
    payload: Any,
    *,
    request_digest: str,
    source: RationalAffineTorusMap,
    plan: AffineTorusFixedLocusPlan,
) -> FixedLocusKernel:
    common_fields = {"protocol_version", "request_digest", "status"}
    if not isinstance(payload, dict) or not common_fields <= set(payload):
        raise ValueError("worker result is not an object with protocol fields")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != _PROTOCOL_VERSION
        or payload["request_digest"] != request_digest
    ):
        raise ValueError("worker result is not bound to its admitted request")
    dimension = source.torus.dimension
    maximum_integer_digits = max(
        len(format_canonical_integer(bounds.source_minor_height))
        for bounds in plan.rank_bounds
    )
    if payload["status"] == "EMPTY":
        if set(payload) != common_fields | {"character", "pairing"}:
            raise ValueError("empty worker result has invalid fields")
        character_value = payload["character"]
        if not isinstance(character_value, list) or len(character_value) != dimension:
            raise ValueError("worker obstruction has an invalid dimension")
        return EmptyFixedLocusKernel(
            character=tuple(
                _strict_integer(value, maximum_digits=maximum_integer_digits)
                for value in character_value
            ),
            pairing=_strict_fraction(
                payload["pairing"], maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS
            ),
        )
    if payload["status"] != "NONEMPTY" or set(payload) != common_fields | {
        "rank",
        "nullity",
        "base_point",
        "identity_embedding",
        "component_generators",
        "relation_matrix",
        "generator_orders",
        "invariant_factors",
        "component_count",
    }:
        raise ValueError("nonempty worker result has invalid fields")
    rank = _strict_dimension(payload["rank"], maximum=dimension)
    nullity = _strict_dimension(payload["nullity"], maximum=dimension)
    if rank + nullity != dimension:
        raise ValueError("worker result rank and nullity disagree")
    generators_value = payload["component_generators"]
    if not isinstance(generators_value, list) or len(generators_value) != rank:
        raise ValueError("worker component generators have an invalid count")
    orders_value = payload["generator_orders"]
    factors_value = payload["invariant_factors"]
    if not isinstance(orders_value, list) or len(orders_value) != rank:
        raise ValueError("worker component orders have an invalid count")
    if not isinstance(factors_value, list) or len(factors_value) > rank:
        raise ValueError("worker invariant factors have an invalid count")
    return NonemptyFixedLocusKernel(
        base_point=_strict_fraction_vector(
            payload["base_point"],
            length=dimension,
            maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
        ),
        identity_embedding=_strict_integer_matrix(
            payload["identity_embedding"],
            rows=dimension,
            columns=nullity,
            maximum_digits=maximum_integer_digits,
        ),
        component_generators=tuple(
            _strict_fraction_vector(
                generator,
                length=dimension,
                maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
            )
            for generator in generators_value
        ),
        relation_matrix=_strict_integer_matrix(
            payload["relation_matrix"],
            rows=rank,
            columns=rank,
            maximum_digits=maximum_integer_digits,
        ),
        generator_orders=tuple(
            _strict_integer(value, maximum_digits=maximum_integer_digits, positive=True)
            for value in orders_value
        ),
        invariant_factors=tuple(
            _strict_integer(value, maximum_digits=maximum_integer_digits, positive=True)
            for value in factors_value
        ),
        component_count=_strict_integer(
            payload["component_count"],
            maximum_digits=maximum_integer_digits,
            positive=True,
        ),
    )


def compute_fixed_locus_kernel(
    source: RationalAffineTorusMap,
    plan: AffineTorusFixedLocusPlan,
) -> FixedLocusKernel:
    """Run one admitted FLINT kernel in a bounded, killable child process."""

    input_bytes = _worker_input(source)
    # The private request omits the canonical source's domain and repeated
    # torus metadata, so the retained-source wire bound is conservative here.
    if (
        len(input_bytes) > plan.worker_input_bytes_upper_bound
        or len(input_bytes) > _WORKER_STDIN_LIMIT
    ):
        raise AssertionError("admitted affine-torus worker request exceeded its bound")
    request_digest = hashlib.sha256(input_bytes).hexdigest()
    require_affine_torus_deadline(plan.deadline, "before FLINT worker setup")
    try:
        with TemporaryDirectory(prefix="jacobian-affine-torus-flint-") as directory:
            allowance = _positive_worker_allowance(plan.deadline)
            completed = run_bounded_process(
                [sys.executable, str(_AFFINE_TORUS_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=allowance,
                environment=worker_environment(locale="C.UTF-8"),
                # The projection drops the complete retained source; that
                # omission dominates its small protocol and digest fields.
                stdout_limit=plan.result_bytes_upper_bound,
                stderr_limit=_WORKER_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(allowance)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("bounded affine-torus FLINT worker could not start") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "affine-torus fixed-locus computation cancelled during the FLINT worker"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "affine-torus FLINT worker exhausted its execution allowance"
        )
    require_affine_torus_deadline(plan.deadline, "after cleaning up the FLINT worker")
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded affine-torus FLINT worker did not establish a fixed locus"
        )
    try:
        decoded = json.loads(completed.stdout.decode("utf-8"))
        result = _decode_worker_projection(
            decoded,
            request_digest=request_digest,
            source=source,
            plan=plan,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded affine-torus FLINT worker returned malformed output"
        ) from exc
    require_affine_torus_deadline(plan.deadline, "after decoding the FLINT worker")
    return result


__all__ = ["compute_fixed_locus_kernel"]
