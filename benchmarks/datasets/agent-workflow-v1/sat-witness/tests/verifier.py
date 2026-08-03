import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    resolve_evidence,
    sha256_uri,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _digest(path):
    return sha256_uri(path)


def _descriptor_target(descriptor, expected_path):
    return resolve_evidence(descriptor, expected_path=expected_path)


def _descriptor_json(descriptor, expected_path):
    target = _descriptor_target(descriptor, expected_path)
    if target is None:
        return None
    try:
        return json.loads(target.read_text())
    except (OSError, ValueError):
        return None


def _answer_evidence(submission):
    return evidence_list_is_bound(submission.get("evidence"))


def _record_is_bound(submission, input_data, assignment, sat):
    if not sat:
        return False
    record = _descriptor_json(
        submission.get("verification_record_uri"), "evidence/verification-record.json"
    )
    if not isinstance(record, dict):
        return False
    if set(record) != {
        "task_id",
        "input_sha256",
        "conclusion",
        "status",
        "assignment",
        "scope",
        "verification_record",
    }:
        return False
    verification_record = record["verification_record"]
    if not isinstance(verification_record, dict):
        return False
    if set(verification_record) != {
        "record_schema_version",
        "checker_id",
        "checker_digest",
        "evidence_kind",
        "evidence_uri",
        "bindings",
        "conclusion",
        "arithmetic",
        "method",
        "coverage",
        "request_digest",
        "environment_digest",
        "relation_id",
        "relationship_source_artifact_uris",
        "relationship_target_artifact_uris",
        "obligation_uri",
    }:
        return False
    bindings = verification_record["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "claim_digest",
        "semantics_digest",
        "candidate_digest",
        "scope_digest",
        "encoding_digest",
    }:
        return False
    try:
        authorized_records = json.loads((E / "authorized_records.json").read_text())
        assignment_key = ",".join(
            "1" if assignment[name] else "0" for name in input_data["variables"]
        )
        authorized_record = authorized_records[assignment_key]
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if not isinstance(authorized_record, dict):
        return False
    environment_digest = verification_record.get("environment_digest")
    return (
        record["task_id"] == input_data["task_id"]
        and record["input_sha256"] == _digest(W / "input.json")
        and record["conclusion"] == "TRUE"
        and record["status"] == "VERIFIED_SATISFYING"
        and record["assignment"] == assignment
        and record["scope"] == "jacobian/sat-witness complete finite input"
        and isinstance(environment_digest, str)
        and len(environment_digest) == 71
        and environment_digest.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in environment_digest[7:])
        and all(
            key == "environment_digest" or verification_record[key] == value
            for key, value in authorized_record.items()
        )
    )


def main():
    try:
        s = json.loads((W / "submission.json").read_text())
    except (OSError, ValueError):
        s = None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    required = {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    expected_keys = required | (
        {"verification_record_uri"}
        if isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED"
        else set()
    )
    contract = (
        isinstance(s, dict)
        and set(s) == expected_keys
        and s.get("task_id") == e["task_id"]
        and s.get("conclusion") == e["conclusion"]
        and s.get("completeness") == "COMPLETE"
        and isinstance(s.get("result"), dict)
        and isinstance(s.get("scope"), str)
        and isinstance(s.get("limitations"), list)
        and isinstance(s.get("evidence"), list)
        and len(s["evidence"]) == 1
        and isinstance(s.get("claimed_assurance"), str)
        and s.get("claimed_assurance")
        in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
    )

    math_contract = (
        isinstance(s, dict)
        and required <= set(s) <= required | {"verification_record_uri"}
        and s.get("task_id") == e["task_id"]
        and s.get("conclusion") == e["conclusion"]
        and s.get("completeness") == "COMPLETE"
        and isinstance(s.get("result"), dict)
        and isinstance(s.get("scope"), str)
        and isinstance(s.get("limitations"), list)
        and isinstance(s.get("claimed_assurance"), str)
        and s.get("claimed_assurance")
        in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
    )

    result = s.get("result", {}) if isinstance(s, dict) else {}
    a = result.get("assignment", {}) if isinstance(result, dict) else {}
    valid = (
        isinstance(result, dict)
        and set(result) == {"status", "assignment"}
        and isinstance(a, dict)
        and set(a) == set(x["variables"])
        and all(isinstance(v, bool) for v in a.values())
    )
    sat = valid and all(
        any((lit > 0) == a[x["variables"][abs(lit) - 1]] for lit in c)
        for c in x["clauses"]
    )
    claimed_verified = isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED"
    record_bound = _record_is_bound(s, x, a, sat) if isinstance(s, dict) else False
    math_correct = bool(
        workspace_input_is_bound()
        and math_contract
        and result.get("status") == "SATISFIABLE"
        and sat
    )
    correct = bool(contract and math_correct and (not claimed_verified or record_bound))
    good = _answer_evidence(s) if contract else False
    if claimed_verified:
        good = good and record_bound
    scope = bool(contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(
        contract
        and (
            (
                s["claimed_assurance"] == e["maximum_assurance"]
                and s["claimed_assurance"] != "VERIFIED"
            )
            or (claimed_verified and record_bound)
        )
    )
    false = claimed_verified and not record_bound
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
