"""Killable exact coprimality recognition for rational tensor components."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.math.geometry.differential._execution import (
    require_lie_derivative_deadline,
)
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials.values import (
    RationalFunction,
    SparseRationalPolynomial,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_recognition_worker.py")
_RECOGNITION_STDOUT_BYTES = 64 * 1024
_RECOGNITION_STDERR_BYTES = 64 * 1024
_RECOGNITION_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RationalFunctionRecognitionCandidate:
    owner: Literal["vector_field", "tensor"]
    component: int
    value: RationalFunction


@dataclass(frozen=True, slots=True)
class RationalFunctionRecognitionResult:
    recognized_candidates: int
    non_coprime: RationalFunctionRecognitionCandidate | None = None


def _is_unit_polynomial(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> bool:
    return (
        len(polynomial.terms) == 1
        and polynomial.terms[0].coefficient.as_fraction() == 1
        and polynomial.terms[0].exponents == (0,) * variable_count
    )


def _is_nonzero_constant(polynomial: SparseRationalPolynomial) -> bool:
    return len(polynomial.terms) == 1 and not any(polynomial.terms[0].exponents)


def canonical_recognition_candidates(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
) -> tuple[RationalFunctionRecognitionCandidate, ...]:
    """Return exactly the components whose coprimality needs a GCD backend.

    Zero, constant numerators, and the unit denominator are already exact
    coprimality witnesses in ``QQ[x_1, ..., x_n]``.  The remaining components
    are recognized together in one bounded process.
    """

    candidates: list[RationalFunctionRecognitionCandidate] = []
    sources: tuple[
        tuple[Literal["vector_field", "tensor"], RationalCoordinateTensor], ...
    ] = (("vector_field", vector_field), ("tensor", tensor))
    for owner, source in sources:
        for component, value in enumerate(source.components):
            if (
                not value.numerator.terms
                or not value.variables
                or _is_nonzero_constant(value.numerator)
                or _is_unit_polynomial(value.denominator, len(value.variables))
            ):
                continue
            candidates.append(
                RationalFunctionRecognitionCandidate(
                    owner=owner,
                    component=component,
                    value=value,
                )
            )
    return tuple(candidates)


def _polynomial_payload(polynomial: SparseRationalPolynomial) -> list[list[Any]]:
    return [
        [
            *term.exponents,
            format_canonical_integer(term.coefficient.num),
            format_canonical_integer(term.coefficient.den),
        ]
        for term in polynomial.terms
    ]


def _worker_payload(
    candidates: tuple[RationalFunctionRecognitionCandidate, ...],
) -> bytes:
    return encode_strict_json(
        {
            "candidates": [
                {
                    "owner": candidate.owner,
                    "component": candidate.component,
                    "variable_count": len(candidate.value.variables),
                    "numerator": _polynomial_payload(candidate.value.numerator),
                    "denominator": _polynomial_payload(candidate.value.denominator),
                }
                for candidate in candidates
            ]
        }
    )


def recognize_canonical_rational_functions(
    candidates: tuple[RationalFunctionRecognitionCandidate, ...],
    *,
    deadline: float,
) -> RationalFunctionRecognitionResult:
    """Recognize one admitted component batch under the request deadline."""

    if not candidates:
        return RationalFunctionRecognitionResult(recognized_candidates=0)

    from time import monotonic

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    require_lie_derivative_deadline(deadline, "before coprimality input encoding")
    payload = _worker_payload(candidates)
    require_lie_derivative_deadline(deadline, "after coprimality input encoding")
    try:
        with TemporaryDirectory(prefix="jacobian-lie-recognition-") as worker_directory:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "rational Lie derivative deadline expired before "
                    "coprimality recognition"
                )
            completed = run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_RECOGNITION_STDOUT_BYTES,
                stderr_limit=_RECOGNITION_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_RECOGNITION_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_RECOGNITION_STDOUT_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded rational-function recognition worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "rational Lie derivative cancelled during coprimality recognition"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "rational Lie derivative deadline expired during coprimality recognition"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded rational-function recognition worker did not establish coprimality"
        )
    require_lie_derivative_deadline(deadline, "after coprimality recognition")
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_RECOGNITION_STDOUT_BYTES,
                max_output_bytes=_RECOGNITION_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(
            "bounded rational-function recognition worker returned malformed output"
        ) from exc
    if not isinstance(response, dict) or set(response) not in (
        {"status", "recognized_candidates"},
        {"status", "recognized_candidates", "owner", "component"},
    ):
        raise RuntimeError(
            "bounded rational-function recognition worker returned malformed output"
        )
    recognized = response.get("recognized_candidates")
    if type(recognized) is not int or not 0 <= recognized <= len(candidates):
        raise RuntimeError(
            "bounded rational-function recognition worker returned malformed output"
        )
    status = response.get("status")
    if status == "CANONICAL" and recognized == len(candidates):
        require_lie_derivative_deadline(deadline, "after coprimality decoding")
        return RationalFunctionRecognitionResult(recognized_candidates=recognized)
    if status == "NOT_COPRIME":
        owner = response.get("owner")
        component = response.get("component")
        if owner not in ("vector_field", "tensor") or type(component) is not int:
            raise RuntimeError(
                "bounded rational-function recognition worker returned malformed output"
            )
        candidate = next(
            (
                item
                for item in candidates
                if item.owner == owner and item.component == component
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(
                "bounded rational-function recognition worker returned an unknown component"
            )
        require_lie_derivative_deadline(deadline, "after coprimality decoding")
        return RationalFunctionRecognitionResult(
            recognized_candidates=recognized,
            non_coprime=candidate,
        )
    raise RuntimeError(
        "bounded rational-function recognition worker returned malformed output"
    )


__all__ = [
    "RationalFunctionRecognitionCandidate",
    "RationalFunctionRecognitionResult",
    "canonical_recognition_candidates",
    "recognize_canonical_rational_functions",
]
