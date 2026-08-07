import json
import re
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    read_evidence_json,
    strict_submission_contract,
)

E = Path("/tests")
INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
ALLOWED = frozenset({"COMPUTED"})
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


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
    keyed = []
    for row in rows:
        if not isinstance(row, dict) or "value" not in row:
            raise ValueError
        keyed.append((_positive_integer(row["value"]), row))
    normalized = [row for _, row in sorted(keyed, key=lambda item: item[0])]
    if len(normalized) != len({key for key, _ in keyed}):
        raise ValueError
    decisions = {}
    for row, integer in zip(normalized, sorted(expected_values), strict=True):
        if set(row) != {
            "value",
            "factors",
            "is_powerful",
            "violating_primes",
        }:
            raise ValueError
        if (
            type(row["is_powerful"]) is not bool
            or not isinstance(row["factors"], list)
            or not isinstance(row["violating_primes"], list)
            or any(
                not isinstance(factor, dict)
                or set(factor) != {"prime", "power"}
                or type(factor["prime"]) is not str
                or type(factor["power"]) is not int
                or isinstance(factor["power"], bool)
                for factor in row["factors"]
            )
            or any(type(prime) is not str for prime in row["violating_primes"])
        ):
            raise ValueError
        parsed = _positive_integer(row["value"])
        factors = _factor(integer)
        violating = [item["prime"] for item in factors if item["power"] < 2]
        powerful = not violating
        # Normalize nested factor collections: the instruction and public
        # evidence schema do not specify an ordering, so compare by prime
        # and reject duplicates rather than requiring the verifier's
        # sorted presentation.
        submitted_factors = sorted(
            row["factors"], key=lambda f: _positive_integer(f["prime"])
        )
        if len(submitted_factors) != len({f["prime"] for f in submitted_factors}):
            raise ValueError
        submitted_violating = sorted(row["violating_primes"], key=_positive_integer)
        if len(submitted_violating) != len(set(submitted_violating)):
            raise ValueError
        if (
            parsed != integer
            or row["value"] != str(integer)
            or submitted_factors != factors
            or row["is_powerful"] != powerful
            or submitted_violating != violating
        ):
            raise ValueError
        decisions[integer] = powerful
    return decisions


def _triple_rows_are_exact(checks, starts, decisions):
    if not isinstance(checks, list) or len(checks) != len(starts):
        return False
    keyed = []
    for check in checks:
        if not isinstance(check, dict) or "start" not in check:
            return False
        try:
            keyed.append((_positive_integer(check["start"]), check))
        except (TypeError, ValueError):
            return False
    normalized = [check for _, check in sorted(keyed, key=lambda item: item[0])]
    if len(normalized) != len({key for key, _ in keyed}):
        return False
    for check, start in zip(normalized, sorted(starts), strict=True):
        if set(check) != {
            "start",
            "members",
            "non_powerful_witnesses",
            "all_powerful",
        }:
            return False
        if (
            type(check["all_powerful"]) is not bool
            or not isinstance(check["members"], list)
            or not all(type(m) is str for m in check["members"])
            or not isinstance(check["non_powerful_witnesses"], list)
            or not all(type(w) is str for w in check["non_powerful_witnesses"])
        ):
            return False
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
        or not isinstance(value["bounded_reference"], dict)
        or set(value["bounded_reference"])
        != {
            "largest_member_at_most",
            "replayed",
        }
        or type(value["bounded_reference"]["largest_member_at_most"]) is not str
        or type(value["bounded_reference"]["replayed"]) is not bool
        or value["bounded_reference"]
        != {"largest_member_at_most": "100000000000000", "replayed": False}
        or value["global_conjecture_status"] != "OPEN"
    ):
        return False
    expected_values = list(range(expected["first"], expected["last"] + 1))
    try:
        decisions = _value_decisions(value["values"], expected_values)
    except (KeyError, TypeError, ValueError, OverflowError):
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


def _safe_evidence(submission):
    """Read digest-bound evidence independently of protocol compliance."""

    if not isinstance(submission, dict):
        return None
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return None
    return read_evidence_json(
        evidence[0],
        expected_path="evidence/powerful-window.json",
        max_bytes=MAX_EVIDENCE_BYTES,
    )


def main():
    submission = load_submission_raw()
    frozen = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    evidence = _safe_evidence(submission)
    finite_window_valid = bool(
        evidence is not None and _finite_window(evidence, frozen, expected)
    )
    evidence_valid = bool(evidence is not None and finite_window_valid)
    math_correct = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("result"), dict)
        and _result(submission["result"])
        and finite_window_valid
    )
    scope = bool(
        isinstance(submission, dict) and submission.get("scope") == expected["scope"]
    )
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(
        isinstance(submission, dict)
        and submission.get("limitations")
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
            not contract
            or not math_correct
            or not evidence_valid
            or not scope
            or not assurance
            or not limitations
            or false_certification
        )
        else 0.8 + 0.1 * scope + 0.05 * assurance + 0.05 * limitations
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
