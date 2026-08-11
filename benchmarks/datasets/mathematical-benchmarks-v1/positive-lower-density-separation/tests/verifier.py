import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    aggregate_reward,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATION = "Eight exact levels replay the general formula but do not machine-prove the infinite limit or the Erdős problem."


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(term in text for term in ("eight", "8", "finite", "exact levels"))
        and any(term in text for term in ("infinite limit", "erdős", "erdos"))
        and any(
            term in text
            for term in ("not prove", "do not prove", "does not prove", "not machine")
        )
    )


# The published prose obligation is structural, not verbatim: the explanation
# must affirmatively state the separation and its limitation, accept equivalent
# phrasing, and reject contradictory or unrelated text.
_LOWER_DENSITY_POSITIVE = (
    "lower density is positive",
    "lower density positive",
    "positive lower density",
)
_DIFFERENT_LIMITS = (
    "different limits",
    "limits differ",
    "endpoint subsequences have different",
    "subsequential limits differ",
    "different subsequential limits",
)
_NATURAL_DENSITY_ABSENT = (
    "natural density does not exist",
    "natural density does not",
    "density does not exist",
    "no natural density",
    "natural density is absent",
    "natural density doesn't exist",
)
_FINITE_REPLAY = (
    "finite levels replay",
    "replay instances",
    "instances of the general formula",
    "eight finite cases",
)
_NOT_GENERAL_PROOF = (
    "rather than proving",
    "not a proof",
    "not proving every",
    "not machine-prove",
    "do not prove",
    "not prove every",
)
_CONCEPT_GROUPS = (
    _LOWER_DENSITY_POSITIVE,
    _DIFFERENT_LIMITS,
    _NATURAL_DENSITY_ABSENT,
    _FINITE_REPLAY,
    _NOT_GENERAL_PROOF,
)
# Direct contradictions of the certified claims; the opposite affirmative
# phrasing also fails the concept-group presence check.
_NEGATIONS = (
    "limits agree",
    "limits are equal",
    "limits coincide",
    "same limit",
    "lower density is zero",
    "lower density vanishes",
    "lower density is not positive",
    "proves every infinite",
    "machine-proves the infinite",
    "proves the general limit",
)


def q(text):
    if (
        not isinstance(text, str)
        or re.fullmatch(r"(?:0|1|[1-9][0-9]*/[1-9][0-9]*)", text) is None
    ):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "base",
        "family",
        "count_formula",
        "levels",
        "lower_density",
        "upper_density",
        "lower_density_positive",
        "natural_density_exists",
        "semantic_relation",
    }:
        return False
    b = result.get("base")
    if (
        type(b) is not int
        or b not in range(2, 10)
        or result.get("family") != "ALTERNATING_GEOMETRIC_BLOCKS"
        or result.get("count_formula") != "(b^(2m+2)-1)/(b+1)"
    ):
        return False
    expected = []
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        expected.append(
            {
                "level": m,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": str(Fraction(count, high)),
                "excluded_density": str(Fraction(count, low)),
            }
        )
    levels = result.get("levels")
    exact_integer_levels = bool(
        isinstance(levels, list)
        and len(levels) == 8
        and all(
            isinstance(row, dict)
            and all(
                type(row.get(field)) is int
                for field in (
                    "level",
                    "included_endpoint",
                    "excluded_endpoint",
                    "cumulative_count",
                )
            )
            for row in levels
        )
    )
    return (
        exact_integer_levels
        and sorted(result.get("levels"), key=lambda row: row["level"]) == expected
        and q(result.get("lower_density")) == Fraction(1, b + 1)
        and q(result.get("upper_density")) == Fraction(b, b + 1)
        and result.get("lower_density_positive") is True
        and result.get("natural_density_exists") is False
        and result.get("semantic_relation") == "FORMALIZED_PREDICATE_STRICTLY_STRONGER"
    )


def _explanation_is_valid(path: Path) -> bool:
    """Stream the explanation, requiring each fact and no contradiction."""

    matched = [False] * len(_CONCEPT_GROUPS)
    contradicted = False
    carry = ""
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                contradicted = contradicted or any(
                    negation in window for negation in _NEGATIONS
                )
                for index, group in enumerate(_CONCEPT_GROUPS):
                    if not matched[index] and any(phrase in window for phrase in group):
                        matched[index] = True
                carry = window[-256:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return not contradicted and all(matched)


def evidence_ok(evidence):
    # The structured endpoint certificate is replayed independently.  The
    # public evidence contract requires one digest-bound text explanation
    # whose content affirmatively states the separation and its limitation;
    # equivalent phrasing is accepted and contradictory or unrelated text is
    # rejected.  The digest and prose are streamed without a hidden size cap.
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    return _explanation_is_valid(path)


def raw_submission():
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    raw = raw_submission()
    e = json.loads((T / "expected.json").read_text())
    input_binding = workspace_input_is_bound()
    s = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    r = raw.get("result") if isinstance(raw, dict) else None
    math_ok = valid_result(r)
    ev = bool(isinstance(raw, dict) and evidence_ok(raw.get("evidence")))
    scope = bool(
        isinstance(raw, dict)
        and raw.get("scope")
        == "parameterized-geometric-block-family-with-eight-replayed-levels"
        and raw.get("completeness") == "COMPLETE"
        and _limitations_valid(raw.get("limitations"))
    )
    assurance = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED"
    )
    false = false_verified_claim(raw, verification_record_bound=False)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_ok),
                "evidence_validity": float(ev),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": aggregate_reward(
                    correctness=math_ok,
                    evidence_validity=ev,
                    scope_accuracy=scope,
                    assurance_calibration=assurance,
                    false_certification=false,
                    protocol_ok=bool(contract and input_binding),
                    soft_assurance=True,
                ),
                "false_certification": false,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
