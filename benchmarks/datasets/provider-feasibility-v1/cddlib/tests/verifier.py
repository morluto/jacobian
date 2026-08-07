from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verifier_support import load_submission, read_evidence_json

expected = json.loads(Path("/tests/expected.json").read_text())
submission = load_submission()
report = None
if isinstance(submission, dict) and isinstance(submission.get("evidence"), list):
    report = read_evidence_json(
        submission["evidence"][0] if len(submission["evidence"]) == 1 else None,
        expected_path="evidence/provider-report.json",
    )
result = submission.get("result") if isinstance(submission, dict) else None


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _execution_bound(report: object) -> bool:
    """Reject reports that only restate public spike success literals."""

    if not isinstance(report, dict):
        return False
    if report.get("contract") != expected["contract"]:
        return False
    if report.get("status") != "COMPLETED":
        return False
    if report.get("conclusion") != expected["report_conclusion"]:
        return False
    if report.get("assurance") != expected["report_assurance"]:
        return False
    provider = report.get("provider")
    reproduction = report.get("reproduction")
    frozen = expected["reproduction"]
    if not isinstance(provider, dict) or not isinstance(reproduction, dict):
        return False
    if not isinstance(frozen, dict):
        return False
    if reproduction.get("scope") != frozen["scope"]:
        return False
    runtime = provider.get("runtime")
    versions = provider.get("versions")
    if (
        not isinstance(runtime, dict)
        or not runtime
        or versions != frozen["mathematical_output"]["versions"]
    ):
        return False
    exact_arithmetic = reproduction.get("exact_arithmetic")
    if not isinstance(exact_arithmetic, dict):
        return False
    if exact_arithmetic.get("large_fraction_input") != frozen["exact_arithmetic_probe"]:
        return False
    if (
        exact_arithmetic.get("large_fraction_output")
        != frozen["exact_arithmetic_probe"]
    ):
        return False
    cases = reproduction.get("cases")
    if cases != frozen["mathematical_output"]["cases"]:
        return False
    mathematical_output = {
        "contract": expected["contract"],
        "provider": expected["provider"],
        "versions": versions,
        "exact_arithmetic": exact_arithmetic,
        "cases": cases,
    }
    if mathematical_output != frozen["mathematical_output"]:
        return False
    mathematical_digest = _sha256_bytes(_canonical_json(mathematical_output))
    if mathematical_digest != frozen["expected_mathematical_output_sha256"]:
        return False
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        return False
    return set(report) > {"contract", "status", "conclusion", "assurance"}


valid = bool(
    isinstance(submission, dict)
    and set(submission)
    == {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    and submission.get("task_id") == expected["task_id"]
    and submission.get("conclusion") == "FEASIBLE"
    and submission.get("claimed_assurance") == "COMPUTED"
    and submission.get("completeness") == "COMPLETE"
    and isinstance(result, dict)
    and result
    == {
        "provider": expected["provider"],
        "contract": expected["contract"],
        "status": "COMPLETED",
        "pin_sha256": expected["pin_sha256"],
    }
    and _execution_bound(report)
)
target = Path("/logs/verifier/reward.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(
        {
            "reward": 1.0 if valid else 0.0,
            "pin_fidelity": float(valid),
            "provider_outcome": float(valid),
            "false_certification": bool(
                isinstance(submission, dict)
                and submission.get("claimed_assurance") == "VERIFIED"
            ),
        },
        sort_keys=True,
    )
)
