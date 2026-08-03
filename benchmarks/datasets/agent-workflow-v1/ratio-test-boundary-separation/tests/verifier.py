from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/ratio-test-boundary-separation"
CONCLUSION = "RATIO_BOUNDARY_INCONCLUSIVE"
SCOPE = "positive rational series indexed by n >= 1"

_PROOF_ASSISTANT_RE = re.compile(r"proof[ -]assistant", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|neither|nor|doesn'?t|isn'?t|wasn'?t|aren'?t|"
    r"won'?t|don'?t|does\s+not|is\s+not|was\s+not)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_VERIFICATION_RE = re.compile(
    r"\b(?:perform(?:ed)?|confirm(?:ed)?|complet(?:ed)?|proven|proved|verified|"
    r"established|done|carried\s+out)\b",
    re.IGNORECASE,
)


def _negated_proof_assistant_limitation(item: object) -> bool:
    """A limitation that negates proof-assistant verification (whole-word match)."""

    if not isinstance(item, str) or not _PROOF_ASSISTANT_RE.search(item):
        return False
    return bool(_NEGATION_RE.search(item))


def _affirmative_proof_assistant_claim(item: object) -> bool:
    """An un-negated limitation asserting proof-assistant verification happened."""

    if not isinstance(item, str) or not _PROOF_ASSISTANT_RE.search(item):
        return False
    if _NEGATION_RE.search(item):
        return False
    return bool(_AFFIRMATIVE_VERIFICATION_RE.search(item))


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value
    ):
        return None
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if str(result) == value else None


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != hidden:
            return False
        source = json.loads(hidden)
    except (OSError, ValueError):
        return False
    return bool(
        source.get("source", {}).get("revision")
        == "339937d75342072a31903739b1bbbe72e1b40c21"
        and source.get("source", {}).get("rows") == [1066, 1069]
    )


def _divergent(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "term",
        "ratio",
        "ratio_error",
        "blocks",
    }:
        return False
    if (value["term"], value["ratio"], value["ratio_error"]) != (
        "1/n",
        "n/(n+1)",
        "1/(n+1)",
    ):
        return False
    blocks = value["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 9:
        return False
    for block, level in zip(blocks, range(2, 11), strict=True):
        if not isinstance(block, dict) or set(block) != {
            "level",
            "start",
            "end",
            "count",
            "term_lower_bound",
            "block_lower_bound",
        }:
            return False
        start = 2**level
        count = 2**level
        lower = Fraction(1, 2 ** (level + 1))
        if block != {
            "level": level,
            "start": start,
            "end": 2 * start - 1,
            "count": count,
            "term_lower_bound": str(lower),
            "block_lower_bound": "1/2",
        }:
            return False
        if count * lower != Fraction(1, 2):
            return False
    return True


def _convergent(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "term",
        "telescoping_identity",
        "ratio",
        "ratio_error",
        "checkpoints",
    }:
        return False
    if (
        value["term"],
        value["telescoping_identity"],
        value["ratio"],
        value["ratio_error"],
    ) != ("1/(n*(n+1))", "1/n-1/(n+1)", "n/(n+2)", "2/(n+2)"):
        return False
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list) or not 4 <= len(checkpoints) <= 12:
        return False
    seen: set[int] = set()
    for item in checkpoints:
        if not isinstance(item, dict) or set(item) != {"N", "partial_sum", "tail"}:
            return False
        n = item["N"]
        if type(n) is not int or n < 1 or n in seen:
            return False
        seen.add(n)
        partial = _fraction(item["partial_sum"])
        tail = _fraction(item["tail"])
        if partial != Fraction(n, n + 1) or tail != Fraction(1, n + 1):
            return False
        if partial + tail != 1:
            return False
    return True


def _result(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"ratio_limit", "divergent_witness", "convergent_witness"}
        and value["ratio_limit"] == "1"
        and _divergent(value["divergent_witness"])
        and _convergent(value["convergent_witness"])
    )


def _evidence(value: object, result: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line[12:].strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound = json.loads(markers[0])
    except (ValueError, RecursionError):
        return False
    prose = "\n".join(
        line for line in text.splitlines() if not line.startswith("RESULT_JSON:")
    )
    folded = prose.casefold()
    return bool(
        bound == result
        and all(
            word in folded
            for word in ("ratio", "diverge", "converge", "dyadic", "telescop")
        )
    )


def _evaluate(submission: object) -> dict[str, float | bool]:
    protocol = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    limitations = (
        submission.get("limitations", []) if isinstance(submission, dict) else []
    )
    has_negated_limitation = isinstance(limitations, list) and any(
        _negated_proof_assistant_limitation(item) for item in limitations
    )
    has_affirmative_claim = isinstance(limitations, list) and any(
        _affirmative_proof_assistant_claim(item) for item in limitations
    )
    correct = bool(
        protocol
        and isinstance(submission, dict)
        and _source_is_bound()
        and submission.get("scope") == SCOPE
        and submission.get("claimed_assurance") == "COMPUTED"
        and _result(submission.get("result"))
        and _evidence(submission.get("evidence"), submission.get("result"))
        and has_negated_limitation
        and not has_affirmative_claim
    )
    score = 1.0 if correct and not false_certification else 0.0
    return {
        "correctness": score,
        "evidence_validity": score,
        "scope_accuracy": score,
        "assurance_calibration": score,
        "reward": score,
        "false_certification": false_certification,
    }


def main() -> None:
    result = _evaluate(load_submission())
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
