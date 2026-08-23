"""Bounded killable backend for exact multivariate factorization.

``factor_list`` on an admitted sparse polynomial can expand an irreducible
factor whose support is exponentially large in the input (sparse-completing
inputs such as ``prod_i (x_i**64 - 1) + z * prod_i (x_i - 1)``), so the
kernel must not run in the engine process: a schema-valid request could
exhaust memory or hang before any typed outcome is reachable.  This module
runs one ``factor_list`` invocation in a resource-limited, killable child
process — the same isolation the Singular ideal backend uses — and reports
work-budget exhaustion as a typed, replayable outcome instead.
"""

from __future__ import annotations

import json
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

# Result validation replays a factorization that already succeeded once
# inside the operation budget, so verification gets a wider margin.
FACTOR_VERIFY_WALL_SECONDS = 60.0


class FactorBackendExhaustedError(Exception):
    """The worker was killed before producing the exact factorization."""


class FactorBackendFailureError(Exception):
    """The worker reported that it could not perform the exact computation."""


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

    Raises :class:`FactorBackendExhaustedError` when the declared work budget is
    exhausted (timeout, resource-limit kill, or oversized transport output)
    and :class:`FactorBackendFailureError` when the worker reports a genuine
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
    completed = run_bounded_process(
        [sys.executable, str(_WORKER_PATH)],
        input_bytes=_serialized_request(polynomial),
        timeout_seconds=resolved_wall,
        environment=worker_environment(locale="C.UTF-8"),
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
    if completed.timed_out or completed.cancelled or completed.stdout_exceeded:
        raise FactorBackendExhaustedError(
            "the bounded factorization worker was stopped by its declared "
            "work budget"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and payload.get("ok") is True:
        return _sympy_decomposition(payload, polynomial.variables)
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise FactorBackendFailureError(str(payload.get("error")))
    # No parsable response with a clean exit status means the resource
    # limits killed the worker mid-computation.
    raise FactorBackendExhaustedError(
        f"the bounded factorization worker terminated with exit code "
        f"{completed.returncode} before producing a result"
    )


def primitive_content_fraction(polynomial: RationalPolynomial) -> Fraction:
    """Exact rational content of the source polynomial, without factoring."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    content = rational_polynomial_to_sympy(polynomial).primitive()[0]
    return Fraction(int(content.p), int(content.q))
