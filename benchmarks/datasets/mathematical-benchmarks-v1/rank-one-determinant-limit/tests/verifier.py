import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
FROZEN_LIMITATIONS = [
    "NO_FLOATING_POINT_SPECTRAL_EVIDENCE",
    "NO_GENERAL_MATRIX_FAMILY_BEYOND_THE_FROZEN_RANK_ONE_FORM",
]


def rat(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    n, d = value["numerator"], value["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError
    return Fraction(n, d)


def frozen_valid():
    try:
        workspace, frozen = W / "input.json", E / "input.json"
        data = json.loads(frozen.read_text())
        return (
            not workspace.is_symlink()
            and not frozen.is_symlink()
            and workspace.read_bytes() == frozen.read_bytes()
            and data["source"]["row_sha256"]
            == "sha256:d3c4493e5fdd5b76a9af3c9833ffa12071e4f9e3b79f4d4b622918348b33c7eb"
        )
    except (OSError, ValueError, KeyError):
        return False


def sample_valid(sample):
    try:
        if not isinstance(sample, dict) or set(sample) != {
            "n",
            "diagonal_product",
            "reciprocal_sum",
            "determinant_constant",
            "determinant_linear",
            "lambda",
        }:
            return False
        n = sample["n"]
        if type(n) is not int or not 2 <= n <= 30:
            return False
        product = math.prod(i**3 - i for i in range(2, n + 1))
        reciprocal_sum = sum(
            (Fraction(1, i**3 - i) for i in range(2, n + 1)), Fraction()
        )
        linear = -product * reciprocal_sum
        return bool(
            type(sample["diagonal_product"]) is int
            and sample["diagonal_product"] == product
            and type(sample["determinant_constant"]) is int
            and sample["determinant_constant"] == product
            and linear.denominator == 1
            and type(sample["determinant_linear"]) is int
            and sample["determinant_linear"] == linear.numerator
            and rat(sample["reciprocal_sum"]) == reciprocal_sum
            and rat(sample["lambda"]) == 1 / reciprocal_sum
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def certificate_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "partial_fraction_coefficients",
        "samples",
        "tail_gap",
        "limit",
    }:
        return False
    try:
        a, b, c = [rat(x) for x in result["partial_fraction_coefficients"]]
        partial = a + b + c == 0 and a - c == 0 and -b == 1
        samples = result["samples"]
        if not isinstance(samples, list):
            return False
        # Validate each sample is a dict before calling .get() so a scalar
        # element does not raise AttributeError outside the except clause.
        ns = []
        for sample in samples:
            if not isinstance(sample, dict):
                return False
            ns.append(sample.get("n"))
        samples_ok = (
            6 <= len(samples) <= 12
            and len(set(ns)) == len(ns)
            and all(sample_valid(sample) for sample in samples)
        )
        tail = result["tail_gap"]
        shifts = tail["affine_shifts"]
        tail_ok = (
            type(tail["numerator"]) is int
            and tail["numerator"] == 8
            and isinstance(shifts, list)
            and len(shifts) == 2
            and all(type(x) is int for x in shifts)
            and sum(shifts) == 1
            and math.prod(shifts) == -2
            and all(
                rat(sample["lambda"]) - 4
                == Fraction(8, (sample["n"] + shifts[0]) * (sample["n"] + shifts[1]))
                for sample in samples
            )
        )
        return partial and samples_ok and tail_ok and rat(result["limit"]) == 4
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def evidence_valid(evidence, result):
    expected_path = "evidence/determinant-certificate.json"
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not evidence_list_is_bound(evidence, expected_path=expected_path)
    ):
        return False
    target = resolve_evidence(evidence[0], expected_path=expected_path)
    try:
        return target is not None and json.loads(target.read_text()) == result
    except (OSError, ValueError):
        return False


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    correctness = bool(contract and frozen_valid() and certificate_valid(result))
    evidence = bool(contract and evidence_valid(submission.get("evidence"), result))
    limitations_ok = bool(
        contract
        and isinstance(submission.get("limitations"), list)
        and submission.get("limitations") == FROZEN_LIMITATIONS
    )
    scope = bool(
        contract
        and submission.get("scope") == expected["required_scope"]
        and submission.get("completeness") == "COMPLETE"
        and limitations_ok
    )
    assurance = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0.0
        if not correctness or not evidence or false_certification
        else 0.8 + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correctness),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier") / "reward.json")


if __name__ == "__main__":
    main()
