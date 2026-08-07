"""Independent exact checker for bounded Erdős-Straus evidence."""

from __future__ import annotations

from typing import Any, TypeGuard

_MIN_N = 2
_MAX_N = 10_000


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _claim_view(payload: dict[str, Any]) -> dict[str, Any]:
    predicate = payload.get("predicate")
    if not isinstance(predicate, dict):
        return payload
    parameters = predicate.get("parameters", {})
    return {
        "predicate": predicate.get("name"),
        **(parameters if isinstance(parameters, dict) else {}),
    }


def _parse_range(payload: dict[str, Any]) -> tuple[int, int] | None:
    lower = payload.get("lower_bound")
    upper = payload.get("upper_bound")
    if (
        not _is_int(lower)
        or not _is_int(upper)
        or lower < _MIN_N
        or upper < lower
        or upper > _MAX_N
    ):
        return None
    return lower, upper


def _validated_range(
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[tuple[int, int] | None, str | None]:
    claim_range = _parse_range(claim)
    candidate_range = _parse_range(candidate)
    if claim_range is None or candidate_range is None:
        return None, "claim or candidate range is malformed"
    if claim_range != candidate_range:
        return None, "candidate range does not exactly match the claim"
    return claim_range, None


def _witness_reject_detail(
    witness: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if witness.get("witness_format") != "erdos_straus.decomposition_table":
        return "unexpected witness format"
    if witness.get("format_version") != "1":
        return "unsupported witness format version"
    if witness.get("role") != "SUPPORTS_CLAIM":
        return "witness role does not support the claim"
    if witness.get("bindings") != expected_bindings:
        return "witness bindings do not match the request"
    return None


def _parse_decomposition_table(
    table: list[Any],
) -> tuple[dict[int, tuple[int, int, int]], str | None]:
    parsed: dict[int, tuple[int, int, int]] = {}
    for row in table:
        if not isinstance(row, dict) or set(row) != {"n", "x", "y", "z"}:
            return parsed, "decomposition row is malformed"
        n, x, y, z = row["n"], row["x"], row["y"], row["z"]
        if not all(_is_int(value) for value in (n, x, y, z)):
            return parsed, "decomposition values must be integers"
        if x <= 0 or y <= 0 or z <= 0:
            return parsed, "decomposition denominators must be positive"
        if n in parsed:
            return parsed, "decomposition table contains duplicate n values"
        if 4 * x * y * z != n * (x * y + x * z + y * z):
            return parsed, f"decomposition identity fails for n={n}"
        parsed[n] = (x, y, z)
    return parsed, None


def check_decomposition_table(request: dict[str, Any]) -> dict[str, Any]:
    """Check one exact decomposition for every integer in the declared range."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = _claim_view(request["claim"]["payload"])
        candidate = request["candidate"]["payload"]
        witness = request["witness"]["payload"]
        if claim.get("predicate") != "erdos_straus_range":
            return _reject("unsupported claim predicate")
        claim_range, detail = _validated_range(claim, candidate)
        if detail is not None:
            return _reject(detail)
        if claim_range is None:
            return _reject("claim or candidate range is malformed")
        detail = _witness_reject_detail(witness, request["expected_bindings"])
        if detail is not None:
            return _reject(detail)

        table = witness.get("payload", {}).get("decompositions")
        if not isinstance(table, list):
            return _reject("decomposition table must be a list")
        parsed, detail = _parse_decomposition_table(table)
        if detail is not None:
            return _reject(detail)

        lower, upper = claim_range
        expected = set(range(lower, upper + 1))
        if set(parsed) != expected:
            return _reject("decomposition table is incomplete or outside the range")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                f"checked exact three-unit-fraction decompositions for every n "
                f"in [{lower}, {upper}]"
            ),
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")
