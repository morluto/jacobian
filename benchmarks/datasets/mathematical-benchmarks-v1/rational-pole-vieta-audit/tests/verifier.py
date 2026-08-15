import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    workspace_input_is_bound,
)

TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 64 * 1024


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON recursively without Python's bool/int coercion."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) is int or type(right) is int:
        return type(left) is type(right) and left == right
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _add(a: list[int], b: list[int]) -> list[int]:
    out = [0] * max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    return out


def _canonical_fraction(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


def _evidence_valid(value: object, result: object) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    evidence = read_evidence_json(
        value[0],
        expected_path="evidence/pole-vieta-certificate.json",
        max_bytes=MAX_EVIDENCE_BYTES,
    )
    return bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == "jacobian/rational-pole-vieta-audit"
        and _json_equal(evidence.get("result"), result)
    )


def _result_valid(result: object) -> bool:
    required = {
        "denominator_coefficients",
        "combined_numerator_coefficients",
        "cleared_polynomial_coefficients",
        "pole_square_residuals",
        "root_sum",
        "diagnosis",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    denominator = [1]
    for k in range(1, 5):
        denominator = _mul(denominator, [-k, 0, 1])
    numerator = [0]
    residuals = []
    for k in range(1, 5):
        quotient = [1]
        residual = k
        for j in range(1, 5):
            if j != k:
                quotient = _mul(quotient, [-j, 0, 1])
                residual *= k - j
        numerator = _add(numerator, [k * value for value in quotient])
        residuals.append({"k": k, "residual": residual})
    cleared = _add(numerator, _mul([4, -2010], denominator))
    root_sum = -Fraction(cleared[-2], cleared[-1])
    return bool(
        result["denominator_coefficients"] == denominator
        and result["combined_numerator_coefficients"] == numerator
        and result["cleared_polynomial_coefficients"] == cleared
        and result["pole_square_residuals"] == residuals
        and all(item["residual"] != 0 for item in residuals)
        and _canonical_fraction(result["root_sum"]) == root_sum
        and result["diagnosis"]
        == "POLES_ARE_PLUS_MINUS_SQUARE_ROOTS_NOT_DENOMINATOR_PARAMETERS"
    )


def main() -> None:
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(_result_valid(result))
    ev_ok = bool(
        isinstance(submission, dict)
        and _evidence_valid(submission.get("witness"), result)
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=bool(input_binding and submission is not None),
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
