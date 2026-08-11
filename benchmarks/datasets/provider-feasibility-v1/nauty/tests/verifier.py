from __future__ import annotations

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


def _digest_ok(value: object) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _report_header_ok(report: dict) -> bool:
    if report.get("contract") != expected["contract"]:
        return False
    if report.get("status") != "COMPLETED":
        return False
    if report.get("conclusion") != expected["report_conclusion"]:
        return False
    return report.get("assurance") == expected["report_assurance"]


def _graph6_count_bound(reproduction: dict, frozen: dict) -> bool:
    expected_graph6 = reproduction.get("expected_graph6")
    if expected_graph6 != frozen["expected_graph6"]:
        return False
    if not isinstance(expected_graph6, list) or not expected_graph6:
        return False
    observed_count = reproduction.get("observed_count")
    if type(observed_count) is not int or observed_count != frozen["observed_count"]:
        return False
    return observed_count == len(expected_graph6)


def _reproduction_digest_bound(reproduction: dict, frozen: dict) -> bool:
    expected_digest = reproduction.get("expected_output_sha256")
    observed_digest = reproduction.get("observed_output_sha256")
    if expected_digest != frozen["expected_output_sha256"]:
        return False
    if not _digest_ok(expected_digest) or not _digest_ok(observed_digest):
        return False
    return observed_digest == expected_digest


def _reproduction_graph_bound(reproduction: dict, frozen: dict, provider: dict) -> bool:
    executables = provider.get("executables")
    if not isinstance(executables, dict) or not executables:
        return False
    if not _graph6_count_bound(reproduction, frozen):
        return False
    return _reproduction_digest_bound(reproduction, frozen)


def _canonicalization_bound(report: dict) -> bool:
    canonicalization = report.get("canonicalization")
    frozen_canonicalization = expected["canonicalization"]
    if not isinstance(canonicalization, dict) or not isinstance(
        frozen_canonicalization, dict
    ):
        return False
    if canonicalization.get("command") != frozen_canonicalization["command"]:
        return False
    if (
        canonicalization.get("expected_output_graph6")
        != frozen_canonicalization["expected_output_graph6"]
    ):
        return False
    if canonicalization.get("input_graph6") != frozen_canonicalization["input_graph6"]:
        return False
    canonical_digest = canonicalization.get("expected_output_sha256")
    if canonical_digest != frozen_canonicalization["expected_output_sha256"]:
        return False
    if not _digest_ok(canonical_digest):
        return False
    if canonicalization.get("observed_output_sha256") != canonical_digest:
        return False
    return canonicalization.get("isomorphic_inputs_converged") is True


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
    if not _reproduction_graph_bound(reproduction, frozen, provider):
        return False
    if not _canonicalization_bound(report):
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
