"""Bounded process owner for one rational cyclotomic kernel call."""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    request_checkpoint,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
)

_CYCLIC_PROFILE_WALL_SECONDS = 3_600.0
_CYCLIC_KERNEL_WORKER = Path(__file__).resolve().with_name("_kernel_worker.py")
_CYCLIC_KERNEL_WORKER_STDOUT_BYTES = 64 * 1024 * 1024
_CYCLIC_KERNEL_WORKER_STDERR_BYTES = 64 * 1024
_CYCLIC_KERNEL_CODEC_LIMITS = CanonicalLimits(
    max_input_bytes=_CYCLIC_KERNEL_WORKER_STDOUT_BYTES,
    max_output_bytes=_CYCLIC_KERNEL_WORKER_STDOUT_BYTES,
    max_depth=64,
)


def _encode_fraction(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _decode_fraction(value: Any) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise TypeError("malformed cyclotomic rational")
    numerator = value["num"]
    denominator = value["den"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise TypeError("malformed cyclotomic rational")
    if (
        max(len(numerator.lstrip("-")), len(denominator))
        > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise TypeError("cyclotomic rational exceeds the admitted digit bound")
    try:
        parsed_numerator = parse_canonical_integer(numerator)
        parsed_denominator = parse_canonical_integer(denominator)
        result = Fraction(parsed_numerator, parsed_denominator)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError("malformed cyclotomic rational") from exc
    if parsed_denominator <= 0 or (result.numerator, result.denominator) != (
        parsed_numerator,
        parsed_denominator,
    ):
        raise TypeError("noncanonical cyclotomic rational")
    return result


def _encode_requests(requests: tuple[tuple[int, int, Any, int], ...]) -> bytes:
    return encode_strict_json(
        [
            [
                format_canonical_integer(order),
                format_canonical_integer(degree),
                [
                    [
                        [_encode_fraction(coordinate) for coordinate in value]
                        for value in row
                    ]
                    for row in matrix_coordinates
                ],
                format_canonical_integer(common_denominator),
            ]
            for order, degree, matrix_coordinates, common_denominator in requests
        ],
        limits=_CYCLIC_KERNEL_CODEC_LIMITS,
    )


def _decode_requests(payload: bytes) -> tuple[tuple[int, int, Any, int], ...]:
    decoded = loads_strict_json(payload, limits=_CYCLIC_KERNEL_CODEC_LIMITS)
    if not isinstance(decoded, list):
        raise TypeError("cyclotomic kernel request must be a list")
    requests: list[tuple[int, int, Any, int]] = []
    for item in decoded:
        if not isinstance(item, list) or len(item) != 4:
            raise TypeError("malformed cyclotomic kernel request")
        order, degree, matrix, denominator = item
        if any(not isinstance(value, str) for value in (order, degree, denominator)):
            raise TypeError("malformed cyclotomic kernel dimensions")
        try:
            order = parse_canonical_integer(order)
            degree = parse_canonical_integer(degree)
            denominator = parse_canonical_integer(denominator)
        except ValueError as exc:
            raise TypeError("malformed cyclotomic kernel dimensions") from exc
        if not isinstance(matrix, list):
            raise TypeError("malformed cyclotomic kernel matrix")
        matrix_coordinates = tuple(
            tuple(
                tuple(_decode_fraction(coordinate) for coordinate in value)
                for value in row
            )
            for row in matrix
            if isinstance(row, list)
        )
        if len(matrix_coordinates) != len(matrix):
            raise TypeError("malformed cyclotomic kernel matrix row")
        requests.append((order, degree, matrix_coordinates, denominator))
    return tuple(requests)


def _encode_result(result: tuple[Any, ...]) -> list[Any]:
    rank, source_dimension, nonzero_minor, kernel_coords = result
    minor = None
    if nonzero_minor is not None:
        row_indices, pivot_columns, determinant = nonzero_minor
        minor = [
            list(row_indices),
            list(pivot_columns),
            [_encode_fraction(value) for value in determinant],
        ]
    return [
        rank,
        source_dimension,
        minor,
        [
            [
                [_encode_fraction(value) for value in coordinates]
                for coordinates in vector
            ]
            for vector in kernel_coords
        ],
    ]


def _encode_response(
    request_digest: bytes, results: tuple[tuple[Any, ...], ...]
) -> bytes:
    return encode_strict_json(
        {
            "digest": request_digest.hex(),
            "results": [_encode_result(result) for result in results],
        },
        limits=_CYCLIC_KERNEL_CODEC_LIMITS,
    )


def _decode_checkpoint(deadline: float | None, stage: str) -> None:
    request_checkpoint(stage)
    if deadline is not None and time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"cyclotomic kernel deadline expired {stage}"
        )


def _require_result_shape(encoded: Any, *, request: tuple[int, int, Any, int]) -> None:
    if not isinstance(encoded, list) or len(encoded) != 4:
        raise TypeError("malformed cyclotomic kernel result")
    rank, source_dimension, encoded_minor, encoded_kernel = encoded
    if type(rank) is not int or type(source_dimension) is not int:
        raise TypeError("malformed cyclotomic kernel dimensions")
    matrix = request[2]
    if not isinstance(matrix, tuple) or not matrix or not isinstance(matrix[0], tuple):
        raise TypeError("malformed cyclotomic request matrix")
    target_dimension = len(matrix)
    expected_source_dimension = len(matrix[0])
    degree = request[1]
    if (
        source_dimension != expected_source_dimension
        or rank < 0
        or rank > min(target_dimension, expected_source_dimension)
    ):
        raise TypeError("cyclotomic kernel dimensions do not match the request")
    if rank == 0:
        if encoded_minor is not None:
            raise TypeError("rank-zero cyclotomic result has a minor")
    else:
        if not isinstance(encoded_minor, list) or len(encoded_minor) != 3:
            raise TypeError("malformed cyclotomic rank minor")
        rows, columns, determinant = encoded_minor
        if (
            not isinstance(rows, list)
            or not isinstance(columns, list)
            or not isinstance(determinant, list)
            or len(rows) != rank
            or len(columns) != rank
            or len(determinant) != degree
            or any(type(value) is not int for value in (*rows, *columns))
            or rows != sorted(set(rows))
            or columns != sorted(set(columns))
            or any(value < 0 or value >= target_dimension for value in rows)
            or any(value < 0 or value >= expected_source_dimension for value in columns)
        ):
            raise TypeError("malformed cyclotomic rank minor")
    if (
        not isinstance(encoded_kernel, list)
        or len(encoded_kernel) != expected_source_dimension - rank
    ):
        raise TypeError("malformed cyclotomic kernel basis dimension")
    for vector in encoded_kernel:
        if not isinstance(vector, list) or len(vector) != expected_source_dimension:
            raise TypeError("malformed cyclotomic kernel vector")
        for coordinates in vector:
            if not isinstance(coordinates, list) or len(coordinates) != degree:
                raise TypeError("malformed cyclotomic kernel coordinates")


def _decode_response(
    payload: bytes,
    *,
    expected_digest: bytes,
    expected_count: int,
    expected_requests: tuple[tuple[int, int, Any, int], ...],
    deadline: float | None,
) -> tuple[tuple[Any, ...], ...]:
    _decode_checkpoint(deadline, "before cyclotomic response decoding")
    decoded = loads_strict_json(payload, limits=_CYCLIC_KERNEL_CODEC_LIMITS)
    _decode_checkpoint(deadline, "after cyclotomic response framing")
    if not isinstance(decoded, dict) or set(decoded) != {"digest", "results"}:
        raise TypeError("malformed cyclotomic kernel response")
    # Bind the response before interpreting any result projection.
    if decoded["digest"] != expected_digest.hex():
        raise TypeError("unbound cyclotomic kernel response")
    encoded_results = decoded["results"]
    if (
        not isinstance(encoded_results, list)
        or len(encoded_results) != expected_count
        or len(expected_requests) != expected_count
    ):
        raise TypeError("wrong cyclotomic kernel response count")
    results: list[tuple[Any, ...]] = []
    for encoded, request in zip(encoded_results, expected_requests, strict=True):
        _decode_checkpoint(deadline, "during cyclotomic response shape validation")
        _require_result_shape(encoded, request=request)
        rank, source_dimension, encoded_minor, encoded_kernel = encoded
        minor = None
        if encoded_minor is not None:
            rows, columns, determinant = encoded_minor
            minor = (
                tuple(rows),
                tuple(columns),
                tuple(_decode_fraction(value) for value in determinant),
            )
        kernel_coords_list: list[tuple[tuple[Fraction, ...], ...]] = []
        for vector in encoded_kernel:
            _decode_checkpoint(deadline, "during cyclotomic kernel decoding")
            kernel_coords_list.append(
                tuple(
                    tuple(_decode_fraction(value) for value in coordinates)
                    for coordinates in vector
                )
            )
        results.append((rank, source_dimension, minor, tuple(kernel_coords_list)))
    _decode_checkpoint(deadline, "after cyclotomic response decoding")
    return tuple(results)


def run_cyclotomic_kernels(
    requests: tuple[tuple[int, int, Any, int], ...],
    deadline: float | None,
) -> tuple[tuple[Any, ...], ...]:
    """Run all exact component kernels in one bounded child process."""

    input_data = _encode_requests(requests)
    request_digest = sha256(input_data).digest()
    remaining = (
        deadline - time.monotonic()
        if deadline is not None
        else _CYCLIC_PROFILE_WALL_SECONDS
    )
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "cyclotomic kernel subprocess exceeded the wall-time limit"
        )

    from jacobian.process import run_bounded_process, worker_environment

    try:
        with TemporaryDirectory(prefix="jacobian-cyclic-kernel-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_CYCLIC_KERNEL_WORKER)],
                input_bytes=input_data,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_CYCLIC_KERNEL_WORKER_STDOUT_BYTES,
                stderr_limit=_CYCLIC_KERNEL_WORKER_STDERR_BYTES,
                cwd=worker_directory,
            )
    except OSError as exc:
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "cyclotomic kernel subprocess was cancelled"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "cyclotomic kernel subprocess exceeded the wall-time limit"
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise OperationResourceExhaustedError(ExecutionResource.OUTPUT)
    if completed.returncode != 0:
        raise OperationBackendError(BackendFailureReason.ABNORMAL_EXIT)
    try:
        return _decode_response(
            completed.stdout,
            expected_digest=request_digest,
            expected_count=len(requests),
            expected_requests=requests,
            deadline=deadline,
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc


__all__ = ["run_cyclotomic_kernels"]
