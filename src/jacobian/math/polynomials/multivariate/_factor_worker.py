"""Killable bounded SymPy ``factor_list`` worker.

This module runs as a standalone child process: it reads one JSON request on
stdin, performs the exact multivariate ``factor_list`` over ``QQ``, and writes
one JSON response on stdout.  It deliberately imports nothing from Jacobian so
the parent can launch it with a bare interpreter and kill it without leaking
engine state.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import QQ, Poly, Rational

    variables = payload["variables"]
    terms = payload["terms"]
    symbols = tuple(__import__("sympy").Symbol(name) for name in variables)
    source = Poly.from_dict(
        {
            tuple(entry[:-2]): Rational(int(entry[-2]), int(entry[-1]))
            for entry in terms
        },
        *symbols,
        domain=QQ,
    )
    coefficient, raw_factors = source.factor_list()
    return {
        "ok": True,
        "coefficient": [int(coefficient.p), int(coefficient.q)],
        "factors": [
            {
                "multiplicity": int(multiplicity),
                "terms": [
                    [*monomial, int(coeff.p), int(coeff.q)]
                    for monomial, coeff in factor.terms()
                ],
            }
            for factor, multiplicity in raw_factors
        ],
    }


def _apply_address_space_limit() -> bool:
    """Self-apply the declared address-space cap before any SymPy import.

    Portable POSIX containment: when the coordinator could not wrap the
    launch in ``prlimit``, the worker still bounds its own allocations
    before the factorization kernel runs.  Returns whether the hard limit
    is active so the coordinator can fail closed when no mechanism exists.
    """

    import os

    raw = os.environ.get("JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES")
    if not raw:
        return False
    try:
        import resource

        cap = int(raw)
        if cap <= 0:
            return False
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        return True
    except (ImportError, OSError, ValueError):
        return False


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.buffer.write(json.dumps(payload).encode("utf-8"))


def _fail(payload_message: str, *, exhausted: bool) -> int:
    _emit(
        {
            "ok": False,
            "error": payload_message,
            "exhausted": exhausted,
            "as_limit_applied": False,
        }
    )
    return 1


def main() -> int:
    as_limit_applied = _apply_address_space_limit()
    if not as_limit_applied:
        # Without an enforced address-space cap this process cannot safely
        # run the potentially explosive factorization at all: abort before
        # any allocation-heavy work instead of recording the failed limit
        # and proceeding anyway.
        return _fail(
            "no hard address-space limit could be applied to the bounded "
            "factorization worker",
            exhausted=False,
        )
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        response = dict(_run(payload))
    except MemoryError as exc:
        # Allocation failure under the address-space budget is resource
        # exhaustion, not a kernel bug: report it distinctly so the
        # coordinator can classify it as an enforcement stop (an
        # execution condition) rather than a kernel defect or a
        # mathematical capacity conclusion.
        _emit(
            {
                "ok": False,
                "error": repr(exc),
                "exhausted": True,
                "as_limit_applied": as_limit_applied,
            }
        )
        return 1
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "error": repr(exc),
                "as_limit_applied": as_limit_applied,
            }
        )
        return 1
    response["ok"] = True
    response["as_limit_applied"] = as_limit_applied
    sys.stdout.buffer.write(json.dumps(response).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
