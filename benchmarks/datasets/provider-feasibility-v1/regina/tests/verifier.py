from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

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


def _report_header_ok(report: dict) -> bool:
    if report.get("contract") != expected["contract"]:
        return False
    if report.get("status") != "COMPLETED":
        return False
    if report.get("conclusion") != expected["report_conclusion"]:
        return False
    return report.get("assurance") == expected["report_assurance"]


def _mathematical_output_bound(
    reproduction: dict, frozen: dict, provider: dict
) -> bool:
    if reproduction.get("scope") != frozen["scope"]:
        return False
    runtime = provider.get("runtime")
    distribution_version = provider.get("distribution_version")
    if not isinstance(runtime, dict) or not runtime:
        return False
    cases = reproduction.get("cases")
    normal_surfaces = reproduction.get("normal_surfaces")
    expected_output = frozen["expected_provider_output"]
    if cases != expected_output["cases"]:
        return False
    if normal_surfaces != expected_output["normal_surfaces"]:
        return False
    mathematical_output = {
        "contract": expected["contract"],
        "provider": expected["provider"],
        "distribution_version": distribution_version,
        "cases": cases,
        "normal_surfaces": normal_surfaces,
    }
    if mathematical_output != expected_output:
        return False
    mathematical_digest = _sha256_bytes(_canonical_json(mathematical_output))
    return mathematical_digest == frozen["expected_mathematical_output_sha256"]


def _execution_bound(report: object) -> bool:
    """Reject reports that only restate public spike success literals."""

    if not isinstance(report, dict) or not _report_header_ok(report):
        return False
    provider = report.get("provider")
    reproduction = report.get("reproduction")
    frozen = expected["reproduction"]
    if not isinstance(provider, dict) or not isinstance(reproduction, dict):
        return False
    if not isinstance(frozen, dict):
        return False
    if not _mathematical_output_bound(reproduction, frozen, provider):
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


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"out-of-range JSON number: {value}")
    return parsed


def _raw_claimed_verified() -> bool:
    """Detect a VERIFIED claim in the raw submission without accepting it."""

    try:
        path = Path("/app/submission.json")
        if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
            return False
        raw = json.loads(
            path.read_text(),
            parse_constant=_reject_nonfinite_json,
            parse_float=_finite_json_float,
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return False
    return isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"


target = Path("/logs/verifier/reward.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(
        {
            "reward": 1.0 if valid else 0.0,
            "pin_fidelity": float(valid),
            "provider_outcome": float(valid),
            "false_certification": _raw_claimed_verified(),
        },
        sort_keys=True,
    )
)
normalize_reward_file(target)
