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


def _run(payload: dict[str, object]) -> dict[str, object]:
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


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        response = json.dumps(_run(payload)).encode("utf-8")
    except Exception as exc:
        sys.stdout.buffer.write(
            json.dumps({"ok": False, "error": repr(exc)}).encode("utf-8")
        )
        return 1
    sys.stdout.buffer.write(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
