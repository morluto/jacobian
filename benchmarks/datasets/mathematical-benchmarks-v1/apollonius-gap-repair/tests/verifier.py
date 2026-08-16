import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = (Path("/app"), Path("/tests"))


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _q(value: object) -> Fraction | None:
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


def _qs(value: object) -> list[Fraction] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    parsed = [_q(item) for item in value]
    return None if any(item is None for item in parsed) else parsed


def _result(value: object, frozen: dict[str, Any]) -> bool:
    fields = {
        "k",
        "c",
        "p",
        "q",
        "center",
        "radius",
        "circle_coefficients",
        "distance_coefficients",
        "multiplier",
    }
    if (
        not isinstance(value, dict)
        or not fields.issubset(value)
        or frozen.get("human_score") != 0
    ):
        return False
    k, c, p, q, center, radius, multiplier = (
        _q(value[name])
        for name in ("k", "c", "p", "q", "center", "radius", "multiplier")
    )
    if None in {k, c, p, q, center, radius, multiplier} or not (
        k > 0 and c > 0 and (k != 1)
    ):
        return False
    if (
        p != k * c / (k + 1)
        or q != k * c / (k - 1)
        or center != (p + q) / 2
        or (radius != abs(q - p) / 2)
        or (radius <= 0)
    ):
        return False
    circle, distance = (
        _qs(value["circle_coefficients"]),
        _qs(value["distance_coefficients"]),
    )
    expected_circle = [
        Fraction(1),
        Fraction(1),
        -2 * center,
        center * center - radius * radius,
    ]
    expected_distance = [1 - k * k, 1 - k * k, 2 * k * k * c, -k * k * c * c]
    return (
        circle == expected_circle
        and distance == expected_distance
        and (multiplier == 1 - k * k)
        and (distance == [multiplier * x for x in circle])
    )


def _result_values_protocol_valid(value: object) -> bool:
    fields = {
        "k",
        "c",
        "p",
        "q",
        "center",
        "radius",
        "circle_coefficients",
        "distance_coefficients",
        "multiplier",
    }
    if not isinstance(value, dict) or not fields.issubset(value):
        return False
    scalar_names = fields - {"circle_coefficients", "distance_coefficients"}
    scalars = {name: _q(value[name]) for name in scalar_names}
    if any(parsed is None for parsed in scalars.values()):
        return False
    if not (
        scalars["k"] > 0
        and scalars["k"] != 1
        and (scalars["c"] > 0)
        and (scalars["radius"] > 0)
    ):
        return False
    return all(
        isinstance(value[name], list)
        and len(value[name]) == 4
        and all(_q(item) is not None for item in value[name])
        for name in ("circle_coefficients", "distance_coefficients")
    )


def _result_protocol_valid(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "k",
            "c",
            "p",
            "q",
            "center",
            "radius",
            "circle_coefficients",
            "distance_coefficients",
            "multiplier",
        }
        and _result_values_protocol_valid(value)
    )


def _encode_certificate_lines(expected: list[str]) -> tuple[bytes, ...] | None:
    try:
        return tuple(line.encode() for line in expected)
    except UnicodeError:
        return None


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    contract = bool(
        isinstance(submission, dict) and _result_protocol_valid(data.get("result"))
    )
    math_correct = _result(data.get("result"), _load())
    correct = input_bound and contract and math_correct
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": float(correct)})
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
