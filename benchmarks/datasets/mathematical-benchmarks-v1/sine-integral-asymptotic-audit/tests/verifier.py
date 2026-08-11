import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    _public_submission_is_valid,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The checker replays an exact formal tail identity and bound under standard calculus lemmas; it does not machine-prove those lemmas or arbitrary transcendental asymptotics."


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


def _term_map(value: object) -> dict[tuple[str, int], int] | None:
    if not isinstance(value, list) or len(value) != 5:
        return None
    result: dict[tuple[str, int], int] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "function",
            "power",
            "coefficient",
        }:
            return None
        function, power, coefficient = (
            item["function"],
            item["power"],
            item["coefficient"],
        )
        if (
            type(function) is not str
            or function not in {"SIN", "COS"}
            or type(power) is not int
            or type(coefficient) is not int
            or power < 1
        ):
            return None
        key = (function, power)
        if key in result:
            return None
        result[key] = coefficient
    return result


def _formal_tail_identity(terms: dict[tuple[str, int], int], remainder: object) -> bool:
    if not isinstance(remainder, dict) or set(remainder) != {
        "integrand",
        "power",
        "coefficient",
    }:
        return False
    if (
        remainder.get("integrand") != "COS"
        or type(remainder.get("power")) is not int
        or remainder.get("power") != 6
        or type(remainder.get("coefficient")) is not int
    ):
        return False
    derivative: defaultdict[tuple[str, int], int] = defaultdict(int)
    for (function, power), coefficient in terms.items():
        if function == "COS":
            derivative[("SIN", power)] -= coefficient
            derivative[("COS", power + 1)] -= power * coefficient
        else:
            derivative[("COS", power)] += coefficient
            derivative[("SIN", power + 1)] -= power * coefficient
    derivative[("COS", 6)] -= remainder["coefficient"]
    return {key: value for key, value in derivative.items() if value} == {
        ("SIN", 1): -1
    }


def _result(value: object, frozen: dict[str, Any]) -> bool:
    required = {
        "tail_terms",
        "tail_remainder",
        "si_terms",
        "si_remainder",
        "absolute_remainder_bound",
        "published_sine_coefficient",
        "corrected_sine_coefficient",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    tail = _term_map(value["tail_terms"])
    si = _term_map(value["si_terms"])
    tr, sr, bound = (
        value["tail_remainder"],
        value["si_remainder"],
        value["absolute_remainder_bound"],
    )
    published = frozen.get("published_expansion")
    if tail is None or si is None or not _formal_tail_identity(tail, tr):
        return False
    if set(si) != set(tail) or any(
        si[key] != -coefficient for key, coefficient in tail.items()
    ):
        return False
    if (
        not isinstance(tr, dict)
        or not isinstance(sr, dict)
        or type(sr.get("power")) is not int
        or type(sr.get("coefficient")) is not int
        or sr != {**tr, "coefficient": -tr["coefficient"]}
    ):
        return False
    if (
        not isinstance(bound, dict)
        or set(bound) != {"numerator", "power", "domain"}
        or type(bound.get("numerator")) is not int
        or type(bound.get("power")) is not int
    ):
        return False
    if (
        bound != {"numerator": abs(sr["coefficient"]) // 5, "power": 5, "domain": "x>0"}
        or abs(sr["coefficient"]) % 5
    ):
        return False
    published_sine = (
        next(
            (
                item.get("coefficient")
                for item in published
                if isinstance(item, dict)
                and item.get("function") == "SIN"
                and item.get("power") == 2
            ),
            None,
        )
        if isinstance(published, list)
        else None
    )
    corrected = si.get(("SIN", 2))
    return (
        type(value["published_sine_coefficient"]) is int
        and type(value["corrected_sine_coefficient"]) is int
        and value["published_sine_coefficient"] == published_sine == 1
        and value["corrected_sine_coefficient"] == corrected == -1
        and published_sine != corrected
    )


def _evidence(value: object, result: object) -> bool:
    if (
        not evidence_list_is_bound(
            value, expected_path="evidence/answer.txt", max_bytes=4096
        )
        or not isinstance(value, list)
        or len(value) != 1
    ):
        return False
    if not isinstance(result, dict):
        return False
    path = resolve_evidence(
        value[0], expected_path="evidence/answer.txt", max_bytes=4096
    )
    if path is None:
        return False
    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return False
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return lines == [
        "sine-integral-certificate-v1",
        f"result_sha256: {digest}",
        f"published_sine_coefficient: {result.get('published_sine_coefficient')}",
        f"corrected_sine_coefficient: {result.get('corrected_sine_coefficient')}",
    ]


def main() -> None:
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    expected = json.loads((TESTS / "expected.json").read_text())
    envelope_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        completeness="COMPLETE",
        verification_record="forbidden",
    )
    contract = bool(envelope_contract and _public_submission_is_valid(submission))
    math_correct = _result(data.get("result"), _load())
    evidence_valid = _evidence(data.get("evidence"), data.get("result"))
    scope_correct = data.get("scope") == expected["required_scope"]
    assurance_correct = data.get("claimed_assurance") == expected["maximum_assurance"]
    limitations = data.get("limitations")
    limitations_correct = isinstance(limitations, list) and LIMITATION in limitations
    completeness_correct = data.get("completeness") == "COMPLETE"
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
        and completeness_correct
        and limitations_correct
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
                "completeness_accuracy": float(completeness_correct),
                "limitations_accuracy": float(limitations_correct),
                "protocol_compliance": float(contract),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
