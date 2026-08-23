"""Bounded killable SymPy-kernel worker for source Gröbner bases.

No Gröbner attempt runs in the service process: an admitted hard system
can consume unbounded time or memory inside the kernel before any output
boundary is noticed.  Each attempt therefore runs here — killably, under
a wall clock and hard resource limits — and reports a tagged outcome:
``ok`` with the wire basis, ``exceeded`` when the complete reduced basis
of the submitted generating set leaves an output budget (sound evidence,
because that basis is unique), ``aborted`` when an intermediate prefix of
an incremental strategy left a budget (a work bound, never a conclusion,
since later generators can shrink an ideal), ``exhausted`` when the hard
address-space cap ended the attempt inside the kernel or during basis
materialization (bounded work, likewise never a mathematical conclusion),
or ``failed`` when the kernel or the wire codec raised for any other
reason inside the attempt — a broken adapter is a transport fault, so the
parent surfaces it as :class:`GroebnerWorkerExecutionError` instead of
projecting it into a domain status.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_TIMEOUT_SECONDS = 60.0
_CPU_SECONDS = 60
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_FILE_SIZE_BYTES = 1024 * 1024
_STDOUT_LIMIT = 128_000_000
_STDERR_LIMIT = 4096

_STRATEGIES = ("ascending", "descending", "complete")

_WORKER_PROGRAM = """\
import json
import sys

from pydantic import ValidationError
from sympy import Poly, QQ, Symbol, groebner

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

_EXHAUSTED_REPORT = '{"status": "exhausted"}\\n'


def _emit_exhausted() -> None:
    # The hard address-space cap ended this attempt.  Report the bounded
    # non-conclusion from a pre-serialized literal so the emission itself
    # cannot need a large allocation immediately after an allocation
    # failure; if even that write exhausts memory, the process exits
    # without a report and the parent raises its typed execution fault.
    try:
        sys.stdout.write(_EXHAUSTED_REPORT)
        sys.stdout.flush()
    except MemoryError:
        pass


def _leaves_output_budget(polys) -> bool:
    aggregate = 0
    bound = 10**MAX_CANONICAL_RATIONAL_DIGITS
    for poly in polys:
        terms = poly.terms()
        aggregate += len(terms)
        if aggregate > 1024:
            return True
        for monom, coefficient in terms:
            if any(exp > MAX_POLYNOMIAL_EXPONENT for exp in monom):
                return True
            if abs(int(coefficient.p)) >= bound or abs(int(coefficient.q)) >= bound:
                return True
    return False


def _kernel_attempt(generators, symbols, order):
    # Run one kernel Groebner call inside the attempt boundary.  Returns
    # ``(polys, over_budget, outcome)``: ``outcome`` is ``None`` when the
    # kernel concluded and the basis materialized, otherwise "exhausted"
    # when the address-space cap ended the attempt -- a work bound on this
    # attempt, never a mathematical claim -- or "failed" for any other
    # kernel exception, which is a broken adapter.
    try:
        basis = groebner(generators, *symbols, order=order, domain=QQ)
        polys = [Poly(expr, *symbols, domain=QQ) for expr in basis.exprs]
        over_budget = _leaves_output_budget(polys)
    except MemoryError:
        return None, None, "exhausted"
    except Exception:
        return None, None, "failed"
    return polys, over_budget, None


def _emit_ok(polys, variables) -> None:
    try:
        elements = [
            rational_polynomial_from_sympy(poly, tuple(variables)) for poly in polys
        ]
    except MemoryError:
        _emit_exhausted()
        return
    except ValidationError:
        print(json.dumps({"status": "exceeded"}))
        return
    except Exception:
        print(json.dumps({"status": "failed"}))
        return
    try:
        report = json.dumps({
            "status": "ok",
            "basis": [element.model_dump() for element in elements],
        })
    except MemoryError:
        _emit_exhausted()
        return
    sys.stdout.write(report + "\\n")


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    variables = tuple(payload["variables"])
    symbols = tuple(Symbol(name) for name in variables)
    strategy = payload["strategy"]
    order = payload["monomial_order"]
    try:
        generators = []
        for terms in payload["generators"]:
            poly = Poly.from_dict(
                {
                    tuple(term["exponents"]): QQ(int(term["num"]), int(term["den"]))
                    for term in terms
                },
                *symbols,
                domain=QQ,
            )
            generators.append(poly.as_expr())
    except Exception:
        print(json.dumps({"status": "failed"}))
        return

    if strategy == "complete":
        polys, over_budget, outcome = _kernel_attempt(generators, symbols, order)
        if outcome is not None:
            if outcome == "exhausted":
                _emit_exhausted()
            else:
                print(json.dumps({"status": "failed"}))
            return
        if over_budget:
            print(json.dumps({"status": "exceeded"}))
            return
        _emit_ok(polys, variables)
        return

    indices = range(len(generators))
    ordered = indices if strategy == "ascending" else reversed(indices)
    current = []
    last = len(generators) - 1
    polys = []
    for position, index in enumerate(ordered):
        current.append(generators[index])
        polys, over_budget, outcome = _kernel_attempt(current, symbols, order)
        if outcome is not None:
            if outcome == "exhausted":
                # The cap ended this prefix attempt: a work bound on the
                # strategy, never evidence about any basis of the ideal.
                _emit_exhausted()
            else:
                print(json.dumps({"status": "failed"}))
            return
        if not over_budget:
            continue
        if position == last:
            # The complete generating set was submitted: this basis is the
            # unique reduced basis of the ideal, so the overflow is real.
            print(json.dumps({"status": "exceeded"}))
            return
        print(json.dumps({"status": "aborted"}))
        return
    _emit_ok(polys, variables)


main()
"""


WorkerStatus = Literal["ok", "exceeded", "aborted", "exhausted"]

# Aggregate term boundary every returned basis must respect (mirrored by
# the result-model validators and by the worker's own budget check).
_MAX_OUTPUT_TERMS = 1_024


class GroebnerWorkerError(RuntimeError):
    """Base class for worker transport failures.

    These are execution/deployment faults, not mathematical conclusions: a
    caller cannot treat them as evidence about the ideal, so they surface as
    typed failures instead of being projected into a domain status.
    """


class GroebnerWorkerLaunchError(GroebnerWorkerError):
    """The worker process could not be launched (deployment failure)."""


class GroebnerWorkerCancelledError(GroebnerWorkerError):
    """The worker was cancelled before it could report any outcome."""


class GroebnerWorkerExecutionError(GroebnerWorkerError):
    """The worker ran but exited without a parseable, well-formed report."""


class GroebnerWorkerTimeoutError(GroebnerWorkerError):
    """The worker was killed by its wall-clock deadline without reporting.

    Host load or an unexpectedly slow machine can trigger this for work that
    would conclude elsewhere, so it must stay retryable and distinguishable
    from any mathematical conclusion.
    """


def _resolve_memory_enforcement() -> str | None:
    """Resolve the prlimit executable or fail closed without a hard cap."""
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        return str(Path(prlimit).resolve())
    import resource

    if not hasattr(resource, "prlimit"):
        raise GroebnerWorkerLaunchError(
            "the groebner worker's hard address-space cap cannot be "
            "enforced on this platform (no util-linux prlimit and no "
            "resource.prlimit); refusing to run unbounded-memory work"
        )
    return None


def _decode_report(raw: bytes) -> dict[str, Any]:
    """Decode one worker report, failing closed on protocol violations."""

    try:
        report = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise GroebnerWorkerExecutionError(str(error)) from None
    if not isinstance(report, dict):
        # A decoded payload with the wrong top-level shape is a broken
        # worker protocol, not evidence about the ideal.
        raise GroebnerWorkerExecutionError(
            f"groebner worker report is {type(report).__name__}, not a JSON object"
        )
    return report


def _basis_from_report(
    report: dict[str, Any], ideal: RationalPolynomialIdeal
) -> tuple[RationalPolynomial, ...]:
    """Convert a worker ``ok`` report to the canonical wire basis.

    The complete returned-basis contract is validated here, at the worker
    protocol layer, so a malformed nested response can never escape as a
    generic downstream result-model failure: every element must live in
    the submitted ideal's ordered ring, no element may be the zero
    polynomial, and the aggregate term count must stay inside the output
    boundary.  (An empty basis is admissible wire form for the zero
    ideal; the result-model replay validators enforce that correlation.)
    """

    from pydantic import ValidationError

    try:
        basis = tuple(
            RationalPolynomial.model_validate(element) for element in report["basis"]
        )
    except (ValidationError, TypeError, KeyError, ValueError) as error:
        # A declared basis outside the canonical contract is a worker fault,
        # not evidence about the ideal.
        raise GroebnerWorkerExecutionError(str(error)) from None
    if any(element.variables != ideal.variables for element in basis):
        raise GroebnerWorkerExecutionError(
            "groebner worker returned a basis polynomial outside the "
            "submitted ideal's ordered ring"
        )
    if any(not element.polynomial.terms for element in basis):
        raise GroebnerWorkerExecutionError(
            "groebner worker returned a zero polynomial inside its basis"
        )
    aggregate_terms = sum(len(element.polynomial.terms) for element in basis)
    if aggregate_terms > _MAX_OUTPUT_TERMS:
        raise GroebnerWorkerExecutionError(
            f"groebner worker returned a {aggregate_terms}-term basis beyond "
            f"the {_MAX_OUTPUT_TERMS}-term output boundary"
        )
    return basis


def complete_basis_in_worker(
    ideal: RationalPolynomialIdeal,
    monomial_order: str,
    strategy: str,
) -> tuple[WorkerStatus, tuple[RationalPolynomial, ...] | None]:
    """Run one bounded kernel attempt and report its outcome.

    ``strategy`` selects the generator order: ``ascending`` and
    ``descending`` run the guarded incremental construction over content-
    derived orders, while ``complete`` submits the whole generating set to
    one unguarded call where Buchberger sees every collapsing pair from
    the start.  Returns ``("ok", basis)`` when the attempt concluded
    within every output budget, ``("exceeded", None)`` for evidenced
    complete-basis overflow, ``("aborted", None)`` when an incremental
    strategy self-reports a bounded non-conclusion, and
    ``("exhausted", None)`` when the hard address-space cap ended the
    attempt — a work bound that decides nothing. A worker ``failed``
    report — a kernel or codec exception inside the attempt for any other
    reason — raises :class:`GroebnerWorkerExecutionError`: a broken
    worker/backend adapter is a transport fault, never work exhaustion.
    An ``ok`` basis that violates the returned-basis contract (foreign
    ordered ring, zero element, or aggregate term overflow) is likewise a
    broken worker protocol and raises :class:`GroebnerWorkerExecutionError`
    before any caller can consume it. Launch failures,
    cancellation, wall-clock timeouts, malformed worker exits, and
    reports with the wrong top-level shape raise their typed
    ``GroebnerWorkerError`` subclasses instead of collapsing into a domain
    status.
    """

    payload = {
        "variables": list(ideal.variables),
        "monomial_order": monomial_order,
        "strategy": strategy,
        "generators": [
            [
                {
                    "exponents": list(term.exponents),
                    "num": term.coefficient.num,
                    "den": term.coefficient.den,
                }
                for term in generator.polynomial.terms
            ]
            for generator in ideal.generators
        ],
    }
    prlimit = _resolve_memory_enforcement()
    try:
        completed = run_bounded_process(
            [sys.executable, "-c", _WORKER_PROGRAM],
            input_bytes=json.dumps(payload).encode("utf-8"),
            timeout_seconds=_TIMEOUT_SECONDS,
            environment=worker_environment(),
            stdout_limit=_STDOUT_LIMIT,
            stderr_limit=_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=_CPU_SECONDS,
                address_space_bytes=_ADDRESS_SPACE_BYTES,
                file_size_bytes=_FILE_SIZE_BYTES,
            ),
            platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
        )
    except OSError as error:
        raise GroebnerWorkerLaunchError(str(error)) from None
    if completed.cancelled:
        raise GroebnerWorkerCancelledError(
            "groebner worker was cancelled before reporting"
        )
    if completed.timed_out:
        raise GroebnerWorkerTimeoutError(
            "groebner worker exceeded its wall-clock deadline before "
            "reporting; the attempt may have been slowed by host load and "
            "is retryable"
        )
    if completed.returncode != 0:
        raise GroebnerWorkerExecutionError(
            f"groebner worker exited with returncode "
            f"{completed.returncode} without a report"
        )
    report = _decode_report(completed.stdout)
    status = report.get("status")
    if status in ("exceeded", "aborted", "exhausted"):
        return status, None
    if status == "failed":
        raise GroebnerWorkerExecutionError(
            "groebner worker reported an internal kernel or codec failure "
            "without any mathematical conclusion"
        )
    if status != "ok":
        raise GroebnerWorkerExecutionError(f"unknown worker status {status!r}")
    return "ok", _basis_from_report(report, ideal)


__all__ = [
    "GroebnerWorkerCancelledError",
    "GroebnerWorkerError",
    "GroebnerWorkerExecutionError",
    "GroebnerWorkerLaunchError",
    "GroebnerWorkerTimeoutError",
    "WorkerStatus",
    "complete_basis_in_worker",
]
