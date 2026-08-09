"""Search-side support for bounded Erdős-Straus verification."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from jacobian.contracts.plugin_number_theory import (
    ErdosStrausCandidate,
    ErdosStrausCapabilityRequest,
    ErdosStrausClaim,
)


def _decompose(n: int) -> tuple[int, int, int] | None:
    """Find ordered positive denominators using exact rational arithmetic."""

    for x in range(n // 4 + 1, (3 * n) // 4 + 1):
        remaining = Fraction(4, n) - Fraction(1, x)
        if remaining <= 0:
            continue
        y_min = max(x, remaining.denominator // remaining.numerator + 1)
        y_max = (2 * remaining.denominator) // remaining.numerator
        for y in range(y_min, y_max + 1):
            last = remaining - Fraction(1, y)
            if last > 0 and last.denominator % last.numerator == 0:
                z = last.denominator // last.numerator
                return x, y, z
    return None


def _decomposition_table(
    lower: int,
    upper: int,
) -> tuple[list[dict[str, int]], int | None]:
    table: list[dict[str, int]] = []
    for n in range(lower, upper + 1):
        decomposition = _decompose(n)
        if decomposition is None:
            return table, n
        x, y, z = decomposition
        table.append({"n": n, "x": x, "y": y, "z": z})
    return table, None


def _evaluate_typed(
    claim: ErdosStrausClaim,
    candidate: ErdosStrausCandidate,
) -> dict[str, Any]:
    lower = candidate.lower_bound
    upper = candidate.upper_bound
    table, missing = _decomposition_table(lower, upper)
    complete = missing is None
    return {
        "response_version": "1",
        "conclusion": "TRUE" if complete else "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE" if complete else "BOUNDED_SEARCH",
        "coverage": "EXHAUSTIVE" if complete else "BOUNDED",
        "objectives": {
            "range_size": upper - lower + 1,
            "decompositions_found": len(table),
        },
        "features": {
            "lower_bound": lower,
            "upper_bound": upper,
        },
        "failure_classifications": ([] if complete else ["decomposition_not_found"]),
        "detail": (
            f"exact decompositions found for every n in [{lower}, {upper}]"
            if complete
            else f"search did not find a decomposition for n={missing}"
        ),
    }


def evaluate_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a bounded range without granting verification authority."""

    try:
        selected = ErdosStrausCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError("Erdős-Straus request does not match its contract") from exc
    return _evaluate_typed(selected.claim, selected.candidate)


def _find_witness_typed(
    claim: ErdosStrausClaim,
    candidate: ErdosStrausCandidate,
    role: str,
) -> dict[str, Any]:
    if role != "SUPPORTS_CLAIM":
        raise ValueError("erdos_straus_range supports only SUPPORTS_CLAIM witnesses")

    lower = candidate.lower_bound
    upper = candidate.upper_bound
    table, missing = _decomposition_table(lower, upper)
    if missing is not None:
        return {
            "response_version": "1",
            "status": "NOT_FOUND_WITHIN_SCOPE",
            "arithmetic": "EXACT_INTEGER",
            "coverage": "BOUNDED",
            "detail": (
                f"search found {len(table)} decompositions but none for n={missing}; "
                "this is not a counterexample"
            ),
        }
    return {
        "response_version": "1",
        "status": "FOUND",
        "witness": {"decompositions": table},
        "witness_format": "erdos_straus.decomposition_table",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "arithmetic": "EXACT_INTEGER",
        "coverage": "EXHAUSTIVE",
        "detail": f"complete proposed decomposition table for [{lower}, {upper}]",
    }


def find_witness_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Propose a complete bounded decomposition table as unverified evidence."""

    try:
        selected = ErdosStrausCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError("Erdős-Straus request does not match its contract") from exc
    return _find_witness_typed(
        selected.claim,
        selected.candidate,
        selected.witness_role,
    )
