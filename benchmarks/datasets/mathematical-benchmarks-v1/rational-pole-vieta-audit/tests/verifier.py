import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 64 * 1024


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
        and (result["cleared_polynomial_coefficients"] == cleared)
        and (result["pole_square_residuals"] == residuals)
        and all(item["residual"] != 0 for item in residuals)
        and (_canonical_fraction(result["root_sum"]) == root_sum)
        and (
            result["diagnosis"]
            == "POLES_ARE_PLUS_MINUS_SQUARE_ROOTS_NOT_DENOMINATOR_PARAMETERS"
        )
    )


def main() -> None:
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(_result_valid(result))
    reward = float(math_ok)
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
