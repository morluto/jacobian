import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

TESTS = Path("/tests")
LIMITATION = "The certificate refutes the published singleton claim and proves sufficiency for its submitted family member; it does not independently prove necessity for every possible trip."
RESULT_FIELDS = frozenset({"alpha", "beta", "even_offset", "odd_offset", "trace"})
TRACE_FIELDS = frozenset({"n", "x", "y", "value", "floor"})


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _result(value: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or not RESULT_FIELDS.issubset(value):
        return False
    alpha, beta = (_fraction(value["alpha"]), _fraction(value["beta"]))
    even, odd = (_fraction(value["even_offset"]), _fraction(value["odd_offset"]))
    if None in {alpha, beta, even, odd} or not (alpha > beta > 0 and alpha + beta == 2):
        return False
    if even != 0 or odd != (alpha - beta) / 2 or (not 0 <= odd < 1):
        return False
    trace = value["trace"]
    length, band = (frozen.get("trace_length"), frozen.get("band"))
    if (
        type(length) is not int
        or type(band) is not int
        or (not isinstance(trace, list))
        or (len(trace) != length)
    ):
        return False
    if any(
        not isinstance(item, dict)
        or not TRACE_FIELDS.issubset(item)
        or any(type(item[field]) is not int for field in ("n", "x", "y", "floor"))
        or (_fraction(item["value"]) is None)
        for item in trace
    ):
        return False
    for n in range(length):
        x, y = ((n + 1) // 2, n // 2)
        exact = x * alpha + y * beta
        item = trace[n]
        if (
            item["n"] != n
            or item["x"] != x
            or item["y"] != y
            or (_fraction(item["value"]) != exact)
            or (item["floor"] != exact.numerator // exact.denominator)
        ):
            return False
        if abs(x - y) >= band:
            return False
    return all(item["floor"] == item["n"] for item in trace)


def _result_protocol_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        return False
    if not all(
        _fraction(value[name]) is not None
        for name in ("alpha", "beta", "even_offset", "odd_offset")
    ):
        return False
    trace = value["trace"]
    return bool(
        isinstance(trace, list)
        and len(trace) == 16
        and all(
            isinstance(item, dict)
            and set(item) == TRACE_FIELDS
            and all(type(item[field]) is int for field in ("n", "x", "y", "floor"))
            and (_fraction(item["value"]) is not None)
            for item in trace
        )
    )


def main() -> None:
    submission = load_submission_raw(require_input_binding=False)
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    contract = bool(
        submission_matches_public_schema(submission)
        and _result_protocol_valid(data.get("result"))
    )
    math_correct = _result(data.get("result"), _load())
    correct = input_bound and contract and math_correct
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "protocol_compliance": float(contract),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
