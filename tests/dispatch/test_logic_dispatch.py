"""Dispatch boundaries for bounded logic operations."""

from __future__ import annotations

import sys

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


def test_dispatch_types_unsat_core_initialization_failure_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable Z3 backend must not turn an accepted request into an import error."""

    monkeypatch.setitem(sys.modules, "z3", None)

    result = invoke_operation(
        "smt.unsat_core", _contradictory_bounds_core(), Catalog.open()
    )

    assert result.output["outcome"] == "UNKNOWN"
    assert result.output["core_indices"] == []
    assert result.output["detail"].startswith("the Z3 backend could not initialize:")
