from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/path-dependent-limit"
CONCLUSION = "ALL_LINEAR_PATHS_ZERO_BUT_LIMIT_DOES_NOT_EXIST"
SCOPE = "all straight lines through (0,0) and submitted nonlinear rational paths"
MAX_EVIDENCE_BYTES = 1_048_576


def _exact_int(value: object) -> bool:
    return type(value) is int


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[+-]?(?:[0-9]+/[0-9]+|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value
    ):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _source_is_bound() -> bool:
    if not workspace_input_is_bound(visible_path=WORKSPACE / "input.json", tests=TESTS):
        return False
    try:
        hidden = (TESTS / "input.json").read_bytes()
        return json.loads(hidden)["source"]["revision"] == (
            "d4e9f8ca877552f4491a9c2d52e0d230c0fca620"
        )
    except (OSError, ValueError, KeyError):
        return False


def _monomial(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict) or set(value) != {"x_power", "y_power"}:
        return None
    x_power = value["x_power"]
    y_power = value["y_power"]
    if not _exact_int(x_power) or not _exact_int(y_power):
        return None
    if x_power < 0 or y_power < 0:
        return None
    return x_power, y_power


def _function_certificate(value: object, p: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "numerator_x_power",
        "numerator_y_power",
        "denominator_terms",
    }:
        return False
    if not all(
        _exact_int(value[field]) for field in ("numerator_x_power", "numerator_y_power")
    ):
        return False
    terms = value["denominator_terms"]
    if not isinstance(terms, list) or len(terms) != 2:
        return False
    normalized = [_monomial(term) for term in terms]
    if any(term is None for term in normalized):
        return False
    return (
        value["numerator_x_power"] == 2 * p
        and value["numerator_y_power"] == 1
        and sorted(term for term in normalized if term is not None)
        == [(0, 2), (4 * p, 0)]
    )


def _line_certificate(value: object, p: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "axes_zero",
        "numerator_order",
        "denominator_leading_order",
        "quotient_order",
        "arbitrary_nonzero_slope_limit",
    }:
        return False
    if value["axes_zero"] is not True:
        return False
    if not all(
        _exact_int(value[field])
        for field in ("numerator_order", "denominator_leading_order", "quotient_order")
    ):
        return False
    return value == {
        "axes_zero": True,
        "numerator_order": 2 * p + 1,
        "denominator_leading_order": 2,
        "quotient_order": 2 * p - 1,
        "arbitrary_nonzero_slope_limit": "0",
    }


def _path_certificate(value: object) -> tuple[Fraction, Fraction, int] | None:
    if not isinstance(value, dict) or set(value) != {"c", "y_x_power", "limit"}:
        return None
    c = _fraction(value["c"])
    limit = _fraction(value["limit"])
    y_x_power = value["y_x_power"]
    if c is None or c == 0 or limit is None or not _exact_int(y_x_power):
        return None
    return c, limit, y_x_power


def _nonlinear_paths(value: object, p: int) -> bool:
    if not isinstance(value, list) or not 3 <= len(value) <= 8:
        return False
    seen: set[Fraction] = set()
    for path in value:
        certificate = _path_certificate(path)
        if certificate is None:
            return False
        c, limit, y_x_power = certificate
        if c in seen or y_x_power != 2 * p or limit != c / (1 + c * c):
            return False
        if limit == 0:
            return False
        seen.add(c)
    return True


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "exponent_p",
        "function",
        "origin_value",
        "line_certificate",
        "nonlinear_paths",
    }:
        return False
    p = value["exponent_p"]
    if not _exact_int(p) or not 1 <= p <= 5 or value["origin_value"] != "0":
        return False
    return (
        _function_certificate(value["function"], p)
        and _line_certificate(value["line_certificate"], p)
        and _nonlinear_paths(value["nonlinear_paths"], p)
    )


def _evaluate(submission: object) -> dict[str, float]:
    data = submission if isinstance(submission, dict) else {}
    input_binding = workspace_input_is_bound()
    math_ok = bool(_source_is_bound() and _result(data.get("result")))
    reward = float(input_binding and submission is not None and math_ok)
    return {
        "correctness": float(math_ok),
        "input_binding": float(input_binding),
        "reward": reward,
    }


def main() -> None:
    result = _evaluate(load_submission())
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True) + "\n")
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
