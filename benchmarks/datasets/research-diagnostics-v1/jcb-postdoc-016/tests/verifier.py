import json
import re
from pathlib import Path

from verifier_support import (
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

E = Path("/tests")
INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
ALLOWED = frozenset({"COMPUTED"})


def _positive_integer(value):
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError
    if INTEGER.fullmatch(value) is None or int(value) < 1 or str(int(value)) != value:
        raise ValueError
    return int(value)


def _factor(value):
    remaining = value
    prime = 2
    factors = []
    while prime * prime <= remaining:
        power = 0
        while remaining % prime == 0:
            remaining //= prime
            power += 1
        if power:
            factors.append({"prime": str(prime), "power": power})
        prime = 3 if prime == 2 else prime + 2
    if remaining > 1:
        factors.append({"prime": str(remaining), "power": 1})
    return factors


def _value_decisions(rows, expected_values):
    if not isinstance(rows, list) or len(rows) != len(expected_values):
        raise ValueError
    decisions = {}
    for row, integer in zip(rows, expected_values, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "value",
            "factors",
            "is_powerful",
            "violating_primes",
        }:
            raise ValueError
        parsed = _positive_integer(row["value"])
        factors = _factor(integer)
        violating = [item["prime"] for item in factors if item["power"] < 2]
        powerful = not violating
        if parsed != integer or row != {
            "value": str(integer),
            "factors": factors,
            "is_powerful": powerful,
            "violating_primes": violating,
        }:
            raise ValueError
        decisions[integer] = powerful
    return decisions


def _triple_rows_are_exact(checks, starts, decisions):
    if not isinstance(checks, list) or len(checks) != len(starts):
        return False
    for check, start in zip(checks, starts, strict=True):
        members = [start, start + 1, start + 2]
        witnesses = [member for member in members if not decisions[member]]
        if (
            check
            != {
                "start": str(start),
                "members": [str(member) for member in members],
                "non_powerful_witnesses": [str(member) for member in witnesses],
                "all_powerful": False,
            }
            or not witnesses
        ):
            return False
    return True


def _finite_window(value, frozen, expected):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "definition",
        "values",
        "triple_checks",
        "bounded_reference",
        "global_conjecture_status",
    }:
        return False
    if (
        value["schema_version"] != "1"
        or value["definition"] != frozen.get("definition")
        or value["bounded_reference"]
        != {"largest_member_at_most": "100000000000000", "replayed": False}
        or value["global_conjecture_status"] != "OPEN"
    ):
        return False
    expected_values = list(range(expected["first"], expected["last"] + 1))
    try:
        decisions = _value_decisions(value["values"], expected_values)
    except (KeyError, TypeError, ValueError):
        return False
    return _triple_rows_are_exact(
        value["triple_checks"], expected["triple_starts"], decisions
    )


def _result(value):
    return value == {
        "evidence_path": "evidence/powerful-window.json",
        "checked_values": [str(value) for value in range(8, 17)],
        "checked_triple_starts": [str(value) for value in range(8, 15)],
        "bounded_reference_replayed": False,
        "global_conjecture_status": "OPEN",
    }


def main():
    submission = load_submission()
    frozen = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    evidence = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/powerful-window.json"
        )
        if contract
        else None
    )
    evidence_valid = bool(evidence is not None)
    math_correct = bool(
        contract
        and _result(submission["result"])
        and _finite_window(evidence, frozen, expected)
    )
    scope = bool(contract and submission["scope"] == expected["scope"])
    assurance = bool(
        contract and submission["claimed_assurance"] == expected["maximum_assurance"]
    )
    limitations = bool(
        contract
        and submission["limitations"]
        == [
            "PUBLIC_10E14_ARTIFACT_NOT_REPLAYED",
            "FINITE_WINDOW_DOES_NOT_PROVE_UNBOUNDED_CONJECTURE",
        ]
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0
        if (
            not math_correct
            or not evidence_valid
            or not scope
            or not assurance
            or not limitations
            or false_certification
        )
        else 0.7
        + 0.1 * evidence_valid
        + 0.1 * scope
        + 0.05 * assurance
        + 0.05 * limitations
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "limitation_accuracy": float(limitations),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
