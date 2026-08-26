"""Bounded killable backend for exact multivariate factorization.

``factor_list`` on an admitted sparse polynomial can expand an irreducible
factor whose support is exponentially large in the input (sparse-completing
inputs such as ``prod_i (x_i**64 - 1) + z * prod_i (x_i - 1)``), so the
kernel must not run in the engine process: a schema-valid request could
exhaust memory or hang before any typed outcome is reachable.  This module
runs one ``factor_list`` invocation in a resource-limited, killable child
process — the same isolation the Singular ideal backend uses — and reports
deadline stops, cancellations, and enforcement kills as typed execution
conditions instead of host exceptions.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from jacobian.math.polynomials.values import RationalPolynomial

__all__ = [
    "FACTOR_VERIFY_WALL_SECONDS",
    "FACTOR_WORK_WALL_SECONDS",
    "FactorBackendExhaustedError",
    "FactorBackendFailureError",
    "FactorBackendInterruptedError",
    "run_bounded_factorization",
]

_WORKER_PATH = Path(__file__).resolve().with_name("_factor_worker.py")

# Declared work budget for one exact factorization attempt.  The admitted
# request envelope does not statically separate inputs whose irreducible
# factors have exponentially many terms from ordinary ones, so the boundary
# is enforced by killing the worker; exhaustion is reported through the
# operation's typed outcome rather than a host exception.
FACTOR_WORK_WALL_SECONDS = 20.0
_FACTOR_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024
_FACTOR_STDOUT_LIMIT = 64 * 1024 * 1024
_FACTOR_STDERR_LIMIT = 1024 * 1024

# Explicit verification may replay a factorization that already succeeded once
# inside the operation budget, so it gets a wider bounded margin.
FACTOR_VERIFY_WALL_SECONDS = 60.0


class FactorBackendExhaustedError(Exception):
    """The exact factorization exceeded a declared output bound."""


class FactorBackendFailureError(Exception):
    """The worker reported that it could not perform the exact computation."""


class FactorBackendInterruptedError(Exception):
    """The worker was stopped by its deadline, cancellation, or an
    enforced resource cap such as the CPU or address-space budget.

    No factorization was obtained and nothing was established about the
    size of the exact output; this is a retryable execution condition,
    not a mathematical conclusion.
    """


def _serialized_request(polynomial: RationalPolynomial) -> bytes:
    return json.dumps(
        {
            "variables": list(polynomial.variables),
            "terms": [
                [*term.exponents, *term.coefficient.as_integer_ratio()]
                for term in polynomial.polynomial.terms
            ],
        }
    ).encode("utf-8")


def _sympy_decomposition(
    payload: dict[str, Any], variables: tuple[str, ...]
) -> tuple[Any, list[tuple[Any, int]]]:
    """Rebuild the ``factor_list`` shape parent-side without re-running it."""

    from sympy import QQ, Poly, Rational

    from jacobian.math.polynomials._conversions import symbols_for_variables

    coefficient = Rational(*payload["coefficient"])
    raw_factors = []
    for record in payload["factors"]:
        poly = Poly.from_dict(
            {
                tuple(entry[:-2]): Rational(int(entry[-2]), int(entry[-1]))
                for entry in record["terms"]
            },
            *symbols_for_variables(variables),
            domain=QQ,
        )
        raw_factors.append((poly, int(record["multiplicity"])))
    return coefficient, raw_factors


def run_bounded_factorization(
    polynomial: RationalPolynomial,
    *,
    wall_seconds: float | None = None,
) -> tuple[Any, list[tuple[Any, int]]]:
    """Run ``factor_list`` killably and return its raw SymPy decomposition.

    Raises :class:`FactorBackendInterruptedError` when the worker was
    stopped by its deadline, cancellation, or an enforced resource cap
    such as the CPU or address-space budget (a retryable execution
    condition establishing nothing about output size),
    :class:`FactorBackendExhaustedError` when the serialized exact
    factorization exceeded the declared transport output bound, and
    :class:`FactorBackendFailureError` when the worker reports a genuine
    computation failure.  The returned decomposition has the same shape as
    ``Poly.factor_list()`` so callers can reuse the monic-decomposition
    kernel unchanged.
    """

    from jacobian.process import (
        ProcessPlatformTools,
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    resolved_wall = float(
        FACTOR_WORK_WALL_SECONDS if wall_seconds is None else wall_seconds
    )
    prlimit = shutil.which("prlimit")
    # The address-space cap is the containment that keeps an admitted
    # sparse-completing input from exhausting host memory.  When util-linux
    # prlimit can wrap the launch, it installs every limit before exec; the
    # worker additionally self-applies the same cap portably on POSIX so
    # the bound survives platforms without the prlimit binary.  A platform
    # offering neither mechanism cannot run this kernel safely, so fail
    # closed instead of launching an unbounded child.
    if prlimit is None and os.name == "nt":
        raise FactorBackendFailureError(
            "no portable hard memory limit is available for the bounded "
            "factorization worker on this platform"
        )
    try:
        completed = run_bounded_process(
            [sys.executable, str(_WORKER_PATH)],
            input_bytes=_serialized_request(polynomial),
            timeout_seconds=resolved_wall,
            environment=worker_environment(
                locale="C.UTF-8",
                overrides={
                    "JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES": str(
                        _FACTOR_ADDRESS_SPACE_BYTES
                    )
                },
            ),
            stdout_limit=_FACTOR_STDOUT_LIMIT,
            stderr_limit=_FACTOR_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                # prlimit's CPU limit parses integer seconds only.
                cpu_seconds=max(1, int(resolved_wall)),
                address_space_bytes=_FACTOR_ADDRESS_SPACE_BYTES,
                file_size_bytes=8 * 1024 * 1024,
            ),
            platform_tools=ProcessPlatformTools(
                prlimit_executable=str(Path(prlimit).resolve()) if prlimit else None
            ),
        )
    except OSError as exc:
        # A launcher failure (missing helper, exec failure) produced no
        # worker run at all: a typed execution failure, never an
        # output-capacity conclusion.
        raise FactorBackendFailureError(
            "the bounded factorization worker could not be started"
        ) from exc
    if completed.timed_out or completed.cancelled:
        raise FactorBackendInterruptedError(
            "the bounded factorization worker was stopped by its deadline "
            "or cancellation before producing a factorization"
        )
    if completed.stdout_exceeded:
        raise FactorBackendExhaustedError(
            "the exact factorization output exceeds the declared transport bound"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    return _worker_outcome(
        payload, completed, polynomial, wrapped_with_prlimit=prlimit is not None
    )


def _worker_outcome(
    payload: Any,
    completed: Any,
    polynomial: RationalPolynomial,
    *,
    wrapped_with_prlimit: bool,
) -> tuple[Any, list[tuple[Any, int]]]:
    """Classify one finished worker run and decode its decomposition."""
    if isinstance(payload, dict) and payload.get("ok") is True:
        _require_worker_memory_limit(payload, wrapped_with_prlimit=wrapped_with_prlimit)
        try:
            return _sympy_decomposition(payload, polynomial.variables)
        except Exception as exc:
            # A syntactically valid success payload with a malformed
            # result shape is a worker defect or version skew, never an
            # exact decomposition.
            raise FactorBackendFailureError(
                "the bounded factorization worker produced a malformed result payload"
            ) from exc
    if isinstance(payload, dict) and payload.get("ok") is False:
        if payload.get("exhausted") is True:
            # Allocation failure under the enforced address-space cap is
            # an enforcement stop exactly like SIGXCPU or a deadline: it
            # proves only that this run's work envelope was too small,
            # never that the exact factors exceed any public bound, so
            # it stays a retryable execution condition.
            raise FactorBackendInterruptedError(
                "the bounded factorization worker exhausted its declared "
                "address-space budget before producing a factorization: "
                f"{payload.get('error')}"
            )
        raise FactorBackendFailureError(str(payload.get("error")))
    if isinstance(completed.returncode, int) and completed.returncode < 0:
        # Only a signal death conclusively attributable to an enforced
        # limit may authenticate the bounded outcome.  SIGXCPU is exactly
        # how the declared CPU limit terminates the worker, but CPU
        # exhaustion is a deadline-type execution condition: it
        # establishes nothing about output size and the verification
        # deadline differs from the producing one, so it must never
        # become OUTPUT_BUDGET_EXCEEDED.
        import signal

        sigxcpu = getattr(signal, "SIGXCPU", None)
        if sigxcpu is not None and completed.returncode == -int(sigxcpu):
            raise FactorBackendInterruptedError(
                "the bounded factorization worker hit its declared CPU budget"
            )
        raise FactorBackendFailureError(
            f"the bounded factorization worker was killed by unexpected "
            f"signal {-completed.returncode}"
        )
    # A clean or self-reported abnormal exit without parsable output is a
    # worker crash, not evidence about output size.
    raise FactorBackendFailureError(
        f"the bounded factorization worker exited with code "
        f"{completed.returncode} without producing a result"
    )


def _require_worker_memory_limit(
    payload: dict[str, Any],
    *,
    wrapped_with_prlimit: bool,
) -> None:
    """Require proof that the hard address-space cap was active."""

    if wrapped_with_prlimit:
        return
    if payload.get("as_limit_applied") is not True:
        raise FactorBackendFailureError(
            "the bounded factorization worker could not activate a hard "
            "memory limit on this platform"
        )


def primitive_content_fraction(polynomial: RationalPolynomial) -> Fraction:
    """Exact rational content of the source polynomial, without factoring."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    content = rational_polynomial_to_sympy(polynomial).primitive()[0]
    return Fraction(int(content.p), int(content.q))
