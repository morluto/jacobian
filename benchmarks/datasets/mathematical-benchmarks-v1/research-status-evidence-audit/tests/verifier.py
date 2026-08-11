import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission_raw,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
MAX_SUBMISSION_BYTES = 1_048_576

RULES = {
    "workshop-equation-status": {
        "classification": "PROBLEM_LISTED_STATUS_UNSUPPORTED",
        "required": {"W2007"},
        "unsupported": {"NEGATIVE_SEARCH_PROVES_CURRENT_STATUS"},
    },
    "vanishing-conjecture-status": {
        "classification": "HISTORICALLY_OPEN_CURRENT_STATUS_NOT_ESTABLISHED",
        "required": {"Z2008"},
        "unsupported": {"HISTORICAL_OPENNESS_PROVES_2026_OPENNESS"},
    },
    "binomial-equation-resolution": {
        "classification": "RESOLVED_BY_CITED_WORK",
        "required": {"BMSS2019"},
        "unsupported": set(),
    },
    "lebesgue-nagell-progress": {
        "classification": "PARTIAL_PROGRESS_NO_FULL_RESOLUTION",
        "required": {"B2004", "KP2025"},
        "unsupported": {
            "RESOLVED_RANGES_CHARACTERIZED_AS_SMALL_AND_SUFFICIENTLY_LARGE"
        },
    },
}


def load_submission_bounded():
    """Load a bounded regular submission without crashing on deep JSON."""
    path = W / "submission.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return None
        return load_submission_raw(path)
    except (OSError, RecursionError):
        return None


def load_frozen():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if (
            workspace.is_symlink()
            or frozen.is_symlink()
            or not workspace.is_file()
            or not frozen.is_file()
            or workspace.stat().st_size > MAX_SUBMISSION_BYTES
            or frozen.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def citation_semantics_are_present(case):
    texts = {item["id"]: item["text"].lower() for item in case["citations"]}
    cid = case["case_id"]
    if cid == "workshop-equation-status":
        return "problem 1" in texts.get("W2007", "")
    if cid == "vanishing-conjecture-status":
        return (
            "open problems" in texts.get("Z2008", "")
            and max(item["year"] for item in case["citations"]) < 2026
        )
    if cid == "binomial-equation-resolution":
        return "all integer solutions" in texts.get(
            "BMSS2019", ""
        ) and "not including" in texts.get("SW1999", "")
    if cid == "lebesgue-nagell-progress":
        return "does not claim a complete resolution" in texts.get(
            "KP2025", ""
        ) and "ranges" in texts.get("B2004", "")
    return False


def cases_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {"cases"}:
        return False
    if not frozen or "cases" not in frozen:
        return False
    submitted = result.get("cases")
    if not isinstance(submitted, list) or len(submitted) != len(RULES):
        return False
    frozen_by_id = {case["case_id"]: case for case in frozen["cases"]}
    seen = set()
    for item in submitted:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "classification",
            "selected_evidence_ids",
            "unsupported_inferences",
        }:
            return False
        cid = item.get("case_id")
        if not isinstance(cid, str) or cid in seen or cid not in RULES:
            return False
        seen.add(cid)
        rule = RULES[cid]
        selected = item.get("selected_evidence_ids")
        unsupported = item.get("unsupported_inferences")
        if (
            not isinstance(selected, list)
            or any(type(value) is not str for value in selected)
            or len(selected) != len(set(selected))
            or not isinstance(unsupported, list)
            or any(type(value) is not str for value in unsupported)
            or len(unsupported) != len(set(unsupported))
        ):
            return False
        available = {citation["id"] for citation in frozen_by_id[cid]["citations"]}
        selected_set = set(selected)
        valid_selection = selected_set == rule["required"] or (
            cid == "lebesgue-nagell-progress" and selected_set == {"KP2025"}
        )
        if (
            item.get("classification") != rule["classification"]
            or not valid_selection
            or not selected_set.issubset(available)
            or set(unsupported) != rule["unsupported"]
            or not citation_semantics_are_present(frozen_by_id[cid])
        ):
            return False
    return seen == set(RULES)


def evidence_valid(evidence, result):
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    # Bound the evidence file before the digest-bound resolution hashes it, so
    # an oversized malformed artifact is a zero-scored non-conclusion rather
    # than a timeout that prevents writing reward.json.
    candidate = W / "evidence" / "answer.txt"
    try:
        if candidate.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
    except OSError:
        return False
    try:
        if not evidence_list_is_bound(evidence):
            return False
        target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    except RecursionError:
        return False
    if target is None:
        return False
    try:
        text = target.read_text()
        markers = [
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        prose = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("RESULT_JSON:")
        ]
        # Build the semantic text from prose only. The RESULT_JSON: marker
        # carries the classifications, case IDs, and selected evidence IDs,
        # which already encode the resolution, partial-progress, historical
        # openness, and problem-listing terms; including it would let a
        # submission whose prose supports none of the classifications pass
        # every semantic_claims check and earn full evidence validity.
        lowered = "\n".join(prose).lower()
        semantic_claims = (
            (
                any(
                    term in lowered
                    for term in ("resolved", "solves", "all integer solutions")
                )
                and any(term in lowered for term in ("binomial", "bmss2019"))
            ),
            (
                any(
                    term in lowered
                    for term in ("partial", "further results", "incomplete")
                )
                and any(term in lowered for term in ("lebesgue", "kp2025"))
            ),
            (
                any(
                    term in lowered
                    for term in ("historical", "open problem", "historically open")
                )
                and any(term in lowered for term in ("vanishing", "z2008"))
            ),
            (
                any(
                    term in lowered
                    for term in (
                        "problem listed",
                        "listed problem",
                        "problem listing",
                        "problem statement",
                    )
                )
                and any(term in lowered for term in ("workshop", "w2007"))
            ),
        )
        return bool(
            len(markers) == 1
            and json.loads(markers[0]) == result
            and prose
            and sum(map(len, prose)) >= 20
            and all(semantic_claims)
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False


def main():
    submission = load_submission_bounded()
    frozen = load_frozen()
    expected = json.loads((E / "expected.json").read_text())
    claimed_assurance = (
        submission.get("claimed_assurance") if isinstance(submission, dict) else None
    )
    if isinstance(claimed_assurance, str):
        mathematical_contract = strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=expected["conclusion"],
            verification_record="optional",
        )
        public_contract = strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=expected["conclusion"],
            allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
            verification_record="forbidden",
        )
    else:
        mathematical_contract = False
        public_contract = False
    result = submission.get("result") if isinstance(submission, dict) else {}
    math_correct = bool(isinstance(result, dict) and cases_valid(result, frozen))
    evidence = bool(
        mathematical_contract
        and isinstance(result, dict)
        and evidence_valid(submission.get("evidence"), result)
    )
    scope = bool(
        mathematical_contract and submission.get("scope") == expected["required_scope"]
    )
    assurance = bool(
        mathematical_contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = isinstance(claimed_assurance, str) and claimed_assurance in {
        "VERIFIED",
        "CHECKED",
    }
    reward = (
        0.0
        if not public_contract
        or not math_correct
        or not evidence
        or false_certification
        else 0.8 + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
