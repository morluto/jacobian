import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
LIMITATION = "The certificate repairs the annotated coordinate identity; it does not independently formalize every theorem or endpoint convention in the full geometry proof."


def _load() -> dict[str, Any]:
    try:
        a, b = WORKSPACE / "input.json", TESTS / "input.json"
        if a.is_symlink() or b.is_symlink() or a.read_bytes() != b.read_bytes():
            return {}
        value = json.loads(b.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _q(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


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
        or set(value) != fields
        or frozen.get("human_score") != 0
    ):
        return False
    k, c, p, q, center, radius, multiplier = (
        _q(value[name])
        for name in ("k", "c", "p", "q", "center", "radius", "multiplier")
    )
    if None in {k, c, p, q, center, radius, multiplier} or not (
        k > 0 and c > 0 and k != 1
    ):
        return False
    if (
        p != k * c / (k + 1)
        or q != k * c / (k - 1)
        or center != (p + q) / 2
        or radius != abs(q - p) / 2
        or radius <= 0
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
        and multiplier == 1 - k * k
        and distance == [multiplier * x for x in circle]
    )


def _evidence(value: object) -> bool:
    if (
        not evidence_list_is_bound(value, expected_path="evidence/answer.txt")
        or not isinstance(value, list)
        or len(value) != 1
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    try:
        return path is not None and "polynomial" in path.read_text().lower()
    except (OSError, UnicodeError):
        return False


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result(data.get("result"), _load()))
    evidence_valid = bool(math_correct and _evidence(data.get("evidence")))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and LIMITATION in data.get("limitations", [])
        and not false_certification
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
