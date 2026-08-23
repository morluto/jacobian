"""Bounded killable SymPy-kernel worker for complete Gröbner bases.

Both guarded incremental strategies can abort while one unguarded kernel
call over the complete generating set still finishes quickly: Buchberger
sees every ideal-collapsing pair from the start, so it need never
materialize the exploding intermediate bases of any prefix.  The kernel
cannot bound itself, so this owner runs it killably in a subprocess with
a wall clock; a timeout, kill, or failure yields no mathematical
conclusion.
"""

from __future__ import annotations

import json
import sys

from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)
from jacobian.process import run_bounded_process, worker_environment

_TIMEOUT_SECONDS = 10.0
_STDOUT_LIMIT = 128_000_000
_STDERR_LIMIT = 4096

_WORKER_PROGRAM = """\
import json
import sys

from pydantic import ValidationError
from sympy import Poly, QQ, Symbol, groebner

from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    variables = tuple(payload["variables"])
    symbols = tuple(Symbol(name) for name in variables)
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
        basis = groebner(
            generators,
            *symbols,
            order=payload["monomial_order"],
            domain=QQ,
        )
        polys = [Poly(expr, *symbols, domain=QQ) for expr in basis.exprs]
        if sum(len(poly.terms()) for poly in polys) > 1024:
            print(json.dumps({"status": "exceeded"}))
            return
        elements = [
            rational_polynomial_from_sympy(poly, variables) for poly in polys
        ]
    except ValidationError:
        print(json.dumps({"status": "exceeded"}))
        return
    except Exception:
        print(json.dumps({"status": "failed"}))
        return
    print(json.dumps({
        "status": "ok",
        "basis": [element.model_dump() for element in elements],
    }))


main()
"""


def complete_basis_in_worker(
    ideal: RationalPolynomialIdeal,
    monomial_order: str,
) -> tuple[tuple[RationalPolynomial, ...] | None, bool]:
    """Run the unguarded kernel once and return its wire basis or evidence.

    Returns ``(basis, False)`` when the worker concluded within every
    output budget, ``(None, True)`` when it decided the complete reduced
    basis against those budgets — sound evidence for the typed budget
    outcome — and ``(None, False)`` on timeout, kill, or failure, where no
    conclusion exists.
    """

    from pydantic import ValidationError

    payload = {
        "variables": list(ideal.variables),
        "monomial_order": monomial_order,
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
    try:
        completed = run_bounded_process(
            [sys.executable, "-c", _WORKER_PROGRAM],
            input_bytes=json.dumps(payload).encode("utf-8"),
            timeout_seconds=_TIMEOUT_SECONDS,
            environment=worker_environment(),
            stdout_limit=_STDOUT_LIMIT,
            stderr_limit=_STDERR_LIMIT,
        )
    except Exception:
        return None, False
    if completed.timed_out or completed.cancelled or completed.returncode != 0:
        return None, False
    try:
        report = json.loads(completed.stdout.decode("utf-8"))
    except Exception:
        return None, False
    status = report.get("status")
    if status == "exceeded":
        return None, True
    if status != "ok":
        return None, False
    try:
        basis = tuple(
            RationalPolynomial.model_validate(element) for element in report["basis"]
        )
    except (ValidationError, TypeError, KeyError, ValueError):
        # A declared basis outside the canonical contract is evidence for
        # no conclusion.
        return None, False
    return basis, False


__all__ = ["complete_basis_in_worker"]
