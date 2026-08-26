"""Dispatch boundaries for bounded logic operations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def _positive_integer_query() -> dict[str, str]:
    return {
        "logic": "QF_LIA",
        "smtlib": (
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        ),
    }


def _contradictory_bounds_core() -> dict[str, str]:
    return {
        "logic": "QF_LIA",
        "smtlib": (
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 1))\n"
            "(assert (<= x 0))\n"
            "(check-sat)"
        ),
    }


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    [
        ("smt.solve", _positive_integer_query()),
        ("smt.unsat_core", _contradictory_bounds_core()),
    ],
)
def test_dispatch_types_parser_resource_failure_as_execution_unknown(
    monkeypatch, operation_id: str, payload: dict[str, str]
) -> None:
    """A fresh request validation cannot claim parser exhaustion is malformed.

    ``math.run`` revalidates every payload, so admission runs the backend
    parse first; a resource failure there must stay admissible and surface
    as the typed execution UNKNOWN instead of ``OperationRequestValidationError``.
    """

    import z3

    def exhausting_parser(_source: str, **_kwargs: object) -> object:
        raise z3.Z3Exception("out of memory")

    monkeypatch.setattr(z3, "parse_smt2_string", exhausting_parser)

    try:
        result = invoke_operation(operation_id, payload, Catalog.open())
    except OperationRequestValidationError as exc:
        raise AssertionError(
            f"{operation_id} rejected a parser resource failure as caller error"
        ) from exc

    assert result.operation_id == operation_id
    assert result.output["outcome"] == "UNKNOWN"


def test_dispatch_reports_memory_exhaustion_for_smt_solve(monkeypatch) -> None:
    """The smt.solve execution path names the exhausted budget it translated."""

    import z3

    def exhausting_parser(_source: str) -> object:
        raise z3.Z3Exception("out of memory")

    monkeypatch.setattr(z3, "parse_smt2_string", exhausting_parser)

    result = invoke_operation("smt.solve", _positive_integer_query(), Catalog.open())

    assert result.output["outcome"] == "UNKNOWN"
    assert result.output["exhausted"] == "memory"


def test_dispatch_preserves_the_cancelled_lpr_backend_outcome(monkeypatch) -> None:
    """Cancellation crosses the LPR adapter and dispatch without becoming ERROR."""

    from jacobian.math.logic import _sat as sat

    monkeypatch.setattr(sat.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sat, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        sat,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=True,
        ),
    )

    result = invoke_operation(
        "sat.refutation.check",
        {
            "cnf": {"variables": ["x"], "clauses": [[-1], [1]]},
            "refutation": {
                "steps": [
                    {
                        "kind": "addition",
                        "clause_id": 3,
                        "clause": [],
                        "at_hint_clause_ids": [1, 2],
                        "propagation_hints": [],
                    }
                ]
            },
        },
        Catalog.open(),
    )

    assert result.output["outcome"] == "CANCELLED"
