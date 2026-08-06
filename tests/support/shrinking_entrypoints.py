"""Deliberately simple untrusted shrinking-plugin entrypoints."""

from __future__ import annotations

from typing import Any


def reduce_positive_value(request: dict[str, Any]) -> dict[str, Any]:
    value = request["target"]["value"]
    return {
        "response_version": "1",
        "current_objectives": {"value": str(value)},
        "reductions": (
            [
                {
                    "reducer": "decrement",
                    "payload": {"value": value - 1},
                    "objectives": {"value": str(value - 1)},
                }
            ]
            if value > 0
            else []
        ),
        "detail": "decrement integer candidate",
    }


def reduce_without_improvement(request: dict[str, Any]) -> dict[str, Any]:
    value = request["target"]["value"]
    return {
        "response_version": "1",
        "current_objectives": {"value": str(value)},
        "reductions": [
            {
                "reducer": "decrement",
                "payload": {"value": value},
                "objectives": {"value": str(value)},
            }
        ],
        "detail": "non-improving fixture proposal",
    }


def reduce_once_then_claim_complete(request: dict[str, Any]) -> dict[str, Any]:
    value = request["target"]["value"]
    return {
        "response_version": "1",
        "current_objectives": {"value": str(value)},
        "reductions": (
            [
                {
                    "reducer": "decrement",
                    "payload": {"value": value - 1},
                    "objectives": {"value": str(value - 1)},
                }
            ]
            if value > 2
            else []
        ),
        "detail": "claims completion after one reduction",
    }


def preserve_positive(request: dict[str, Any]) -> dict[str, Any]:
    reduced = request["reduced"]["payload"]["value"]
    return {
        "accepted": reduced >= 1,
        "conclusion": "FALSE" if reduced >= 1 else "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": "value remains positive" if reduced >= 1 else "value is not positive",
    }


def preserve_positive_except_failed_boundary(
    request: dict[str, Any],
) -> dict[str, Any]:
    if (
        request["original"]["payload"]["value"] == 2
        and request["reduced"]["payload"]["value"] == 1
    ):
        raise RuntimeError("fixture checker failed at the reduction boundary")
    return preserve_positive(request)
