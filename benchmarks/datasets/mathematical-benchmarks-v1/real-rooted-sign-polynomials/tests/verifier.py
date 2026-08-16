import itertools
import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W, E = (Path("/app"), Path("/tests"))


def _load_frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (
            (W / "input.json").is_symlink()
            or (E / "input.json").is_symlink()
            or (W / "input.json").read_bytes() != raw
        ):
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _discriminant(coefficients):
    degree = len(coefficients) - 1
    if degree == 1:
        return None
    if degree == 2:
        c, b, a = coefficients
        return b * b - 4 * a * c
    d, c, b, a = coefficients
    return (
        b * b * c * c
        - 4 * a * c**3
        - 4 * b**3 * d
        - 27 * a * a * d * d
        + 18 * a * b * c * d
    )


def _all_cases():
    cases = []
    for degree in range(1, 4):
        for coefficients in itertools.product((-1, 1), repeat=degree + 1):
            discriminant = _discriminant(coefficients)
            cases.append(
                {
                    "coefficients": list(coefficients),
                    "degree": degree,
                    "discriminant": discriminant,
                    "all_roots_real": discriminant is None or discriminant >= 0,
                }
            )
    return cases


def _result_ok(result, frozen):
    required = {
        "normalized_second_coefficient",
        "second_power_sum",
        "root_product_square",
        "maximum_degree",
        "candidate_audit",
        "classified_polynomials",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or frozen.get("coefficient_set") != [-1, 1]
    ):
        return False
    expected = _all_cases()
    expected_by_coefficients = {tuple(case["coefficients"]): case for case in expected}
    submitted_audit = result["candidate_audit"]
    if not isinstance(submitted_audit, list):
        return False
    try:
        submitted_by_coefficients = {
            tuple(case["coefficients"]): case for case in submitted_audit
        }
        submitted_classification = {
            tuple(coefficients) for coefficients in result["classified_polynomials"]
        }
    except (KeyError, TypeError):
        return False
    accepted = {
        tuple(case["coefficients"]) for case in expected if case["all_roots_real"]
    }
    scalar_ok = (
        type(result["normalized_second_coefficient"]) is int
        and result["normalized_second_coefficient"] == -1
        and (type(result["second_power_sum"]) is int)
        and (result["second_power_sum"] == 3)
        and (type(result["root_product_square"]) is int)
        and (result["root_product_square"] == 1)
        and (type(result["maximum_degree"]) is int)
        and (result["maximum_degree"] == 3)
    )
    audit_ok = all(
        isinstance(case, dict)
        and set(case) == {"coefficients", "degree", "discriminant", "all_roots_real"}
        and (type(case["degree"]) is int)
        and (case["discriminant"] is None or type(case["discriminant"]) is int)
        and (type(case["all_roots_real"]) is bool)
        for case in submitted_audit
    )
    return bool(
        scalar_ok
        and audit_ok
        and (len(submitted_by_coefficients) == len(submitted_audit) == len(expected))
        and (submitted_by_coefficients == expected_by_coefficients)
        and (len(submitted_classification) == len(result["classified_polynomials"]))
        and (submitted_classification == accepted)
    )


def main():
    submission_path = W / "submission.json"
    try:
        symlinked = submission_path.is_symlink()
    except OSError:
        symlinked = True
    submission = None if symlinked else load_submission()
    frozen = _load_frozen()
    math_correct = bool(submission and _result_ok(submission.get("result"), frozen))
    correct = math_correct
    reward = float(correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
