"""Process-boundary tests for rational-function coprimality recognition."""

from __future__ import annotations

from time import monotonic
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    recognize_canonical_rational_functions,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _polynomial(*terms: tuple[int, tuple[int, ...]]) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=str(coefficient), den="1"),
                exponents=exponents,
            )
            for coefficient, exponents in terms
        )
    )


def _candidate() -> RationalFunctionRecognitionCandidate:
    return RationalFunctionRecognitionCandidate(
        owner="vector_field",
        component=0,
        value=RationalFunction(
            variables=("x",),
            numerator=_polynomial((1, (1,))),
            denominator=_polynomial((1, (1,)), (1, (0,))),
        ),
    )


def test_recognition_worker_timeout_uses_the_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    observed: dict[str, Any] = {}

    def timed_out(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(process, "run_bounded_process", timed_out)
    started = monotonic()
    deadline = started + 5.0

    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during coprimality recognition",
        ),
    ):
        recognize_canonical_rational_functions((_candidate(),), deadline=deadline)

    assert 0 < observed["timeout_seconds"] <= 5.0
    resource_limits = observed["resource_limits"]
    assert isinstance(resource_limits, ProcessResourceLimits)
    assert resource_limits.cpu_seconds is not None
    assert resource_limits.address_space_bytes is not None
    assert resource_limits.file_size_bytes is not None
    assert str(observed["cwd"]).split("/")[-1].startswith("jacobian-lie-recognition-")
