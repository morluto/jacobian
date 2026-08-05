import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The certificate refutes the published singleton claim and proves sufficiency for its submitted family member; it does not independently prove necessity for every possible trip."


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _submission() -> dict[str, Any] | None:
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed


def _result(value: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "alpha",
        "beta",
        "even_offset",
        "odd_offset",
        "trace",
    }:
        return False
    alpha, beta = _fraction(value["alpha"]), _fraction(value["beta"])
    even, odd = _fraction(value["even_offset"]), _fraction(value["odd_offset"])
    if None in {alpha, beta, even, odd} or not (alpha > beta > 0 and alpha + beta == 2):
        return False
    if even != 0 or odd != (alpha - beta) / 2 or not (0 <= odd < 1):
        return False
    trace = value["trace"]
    length, band = frozen.get("trace_length"), frozen.get("band")
    if (
        type(length) is not int
        or type(band) is not int
        or not isinstance(trace, list)
        or len(trace) != length
    ):
        return False
    if any(
        not isinstance(item, dict)
        or set(item) != {"n", "x", "y", "value", "floor"}
        or any(type(item[field]) is not int for field in ("n", "x", "y", "floor"))
        or not isinstance(item["value"], str)
        for item in trace
    ):
        return False
    for n in range(length):
        x, y = (n + 1) // 2, n // 2
        exact = x * alpha + y * beta
        item = trace[n]
        if (
            item["n"] != n
            or item["x"] != x
            or item["y"] != y
            or _fraction(item["value"]) != exact
            or item["floor"] != exact.numerator // exact.denominator
        ):
            return False
        if abs(x - y) >= band:
            return False
    return all(item["floor"] == item["n"] for item in trace)


def _evidence(value: object, result: object) -> bool:
    if (
        not evidence_list_is_bound(value, expected_path="evidence/answer.txt")
        or not isinstance(value, list)
        or len(value) != 1
    ):
        return False
    if not isinstance(result, dict) or not isinstance(result.get("trace"), list):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return False
    trace_digest = hashlib.sha256(
        json.dumps(result["trace"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return lines == [
        "emerald-path-family-certificate-v1",
        f"alpha: {result.get('alpha')}",
        f"beta: {result.get('beta')}",
        f"even_offset: {result.get('even_offset')}",
        f"odd_offset: {result.get('odd_offset')}",
        f"trace_sha256: {trace_digest}",
    ]


def main() -> None:
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = _result(data.get("result"), _load())
    evidence_valid = _evidence(data.get("evidence"), data.get("result"))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        input_bound
        and contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and LIMITATION in data.get("limitations", [])
        and not false_certification
    )
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "input_binding": float(input_bound),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
