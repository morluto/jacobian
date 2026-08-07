import json
import re
from decimal import Decimal, DecimalException
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

E = Path("/tests")

_ASSURANCE_ORDER = {
    "UNVERIFIED": 0,
    "COMPUTED": 1,
    "CHECKED": 2,
    "VERIFIED": 3,
}


def _is_integer(value):
    """Accept any schema-valid integral JSON number while rejecting booleans.

    JSON Schema's ``integer`` type accepts numbers with a zero fractional part
    (e.g. ``12.0``), so the verifier must validate mathematical integrality
    rather than requiring Python's ``int`` representation. Booleans are still
    rejected because ``False == 0`` would otherwise spoof a zero element.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, Decimal):
        return value == value.to_integral_value()
    return False


def _is_int_list(value):
    return isinstance(value, list) and all(_is_integer(item) for item in value)


def _is_int_matrix(value):
    return bool(
        isinstance(value, list)
        and all(isinstance(row, list) for row in value)
        and all(_is_integer(item) for row in value for item in row)
    )


def _fits_small_integer(value, maximum):
    if not _is_integer(value):
        return False
    try:
        return -maximum <= value <= maximum
    except (DecimalException, OverflowError):
        return False


def _bounded_index(value, maximum):
    if not _is_integer(value):
        return None
    try:
        if value < 0 or value > maximum:
            return None
        return int(value)
    except (DecimalException, OverflowError):
        return None


def _valid_cover(result, bounds):
    n = result.get("modulus")
    step = result.get("subgroup_step")
    if (
        not _is_integer(n)
        or not _is_integer(step)
        or not bounds["minimum_modulus"] <= n <= bounds["maximum_modulus"]
        or not _fits_small_integer(n, bounds["maximum_modulus"])
        or not _fits_small_integer(step, bounds["maximum_modulus"])
    ):
        return False
    n = int(n)
    step = int(step)
    if (
        step < bounds["minimum_cosets"]
        or n % step
        or n // step < bounds["minimum_coset_size"]
    ):
        return False
    subgroup = list(range(0, n, step))
    representatives = list(range(step))
    cosets = [
        sorted((representative + value) % n for value in subgroup)
        for representative in representatives
    ]
    submitted_subgroup = result.get("subgroup")
    submitted_representatives = result.get("representatives")
    submitted_cosets = result.get("cosets")
    if not (
        _is_int_list(submitted_subgroup)
        and _is_int_list(submitted_representatives)
        and _is_int_matrix(submitted_cosets)
    ):
        return False
    # The published schema requires only unique integer elements, so the
    # subgroup and each coset are compared as unordered collections. The coset
    # list order is still fixed because covering-part references and the
    # duplicate index pair address cosets by position.
    return bool(
        sorted(submitted_subgroup) == sorted(subgroup)
        and submitted_representatives == representatives
        and [sorted(coset) for coset in submitted_cosets] == cosets
        and len({value for coset in cosets for value in coset}) == n
        and sum(len(coset) for coset in cosets) == n
    )


def _valid_predicates(result):
    artifact = result.get("part_artifact")
    references = result.get("covering_part_references")
    cosets = result.get("cosets")
    pair = result.get("duplicate_indices")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"id", "kind", "elements"}
        or not _is_integer(artifact.get("id"))
        or artifact.get("id") != 0
        or artifact.get("kind") != "SUBGROUP"
        or not _is_int_list(artifact.get("elements"))
        or sorted(artifact.get("elements")) != sorted(result.get("subgroup"))
        or not isinstance(references, list)
        or not isinstance(cosets, list)
        or len(references) != len(cosets)
        or any(not _is_integer(reference) or reference != 0 for reference in references)
    ):
        return False
    if not isinstance(pair, list) or len(pair) != 2:
        return False
    left = _bounded_index(pair[0], len(references) - 1)
    right = _bounded_index(pair[1], len(references) - 1)
    if left is None or right is None:
        return False
    # The submission schema only requires two distinct in-range integers, so the
    # exposing pair is accepted in either order; the remaining checks are
    # symmetric in left/right.
    if not (
        0 <= left < len(references) and 0 <= right < len(references) and left != right
    ):
        return False
    unique_parts = list(dict.fromkeys(references))
    sizes = {part: len(cosets[references.index(part)]) for part in unique_parts}
    range_pairwise = all(
        sizes[a] != sizes[b]
        for i, a in enumerate(unique_parts)
        for b in unique_parts[i + 1 :]
    )
    indexed_pairwise = all(
        len(cosets[i]) != len(cosets[j])
        for i in range(len(cosets))
        for j in range(i + 1, len(cosets))
    )
    return bool(
        result.get("range_pairwise_truth") is range_pairwise is True
        and result.get("indexed_pairwise_truth") is indexed_pairwise is False
        and references[left] == references[right] == artifact["id"]
        and len(cosets[left]) == len(cosets[right])
    )


def _valid_result(result, source):
    required = {
        "modulus",
        "subgroup_step",
        "subgroup",
        "representatives",
        "cosets",
        "part_artifact",
        "covering_part_references",
        "range_pairwise_truth",
        "indexed_pairwise_truth",
        "duplicate_indices",
    }
    return bool(
        isinstance(result, dict)
        and set(result) == required
        and _valid_cover(result, source["bounds"])
        and _valid_predicates(result)
    )


_CONJECTURE_KINDS = ("open conjecture", "underlying conjecture")
_DISCLAIMERS = (
    "does not settle",
    "does not resolve",
    "does not prove",
    "does not establish",
    "not settle",
    "not resolve",
    "not prove",
    "not establish",
    "doesn't settle",
    "doesn't resolve",
    "is insufficient to resolve",
    "is insufficient to settle",
)
# An affirmative settlement claim ("settles the open conjecture", etc.) that is
# not preceded by a negation. The fixed-width negative lookbehinds exclude
# "not settle ..." and "n't settle ..." while still matching bare overclaims.
_AFFIRMATIVE_SETTLEMENT = re.compile(
    r"(?<!not )(?<!n't )"
    r"(?:settle|settles|resolve|resolves|prove|proves|establish|establishes)"
    r" the (?:open|underlying) conjecture"
)


def _limitation_disclaims_conjecture(text):
    """Require a limitation to disclaim settling the open conjecture.

    The task forbids presenting the finite audit as settling the underlying
    conjecture, so a valid limitation must reference the open/underlying
    conjecture, explicitly disclaim settling it, and must not affirm that it is
    settled. Mere keyword presence is not enough.
    """
    lowered = text.casefold()
    if not any(kind in lowered for kind in _CONJECTURE_KINDS):
        return False
    if _AFFIRMATIVE_SETTLEMENT.search(lowered):
        return False
    return any(disclaimer in lowered for disclaimer in _DISCLAIMERS)


def _load_exact_submission():
    path = Path("/app/submission.json")
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
            return None
        return json.loads(path.read_text(), parse_float=Decimal)
    except (
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
        DecimalException,
    ):
        return None


def _evidence_descriptor_ok(descriptor):
    return (
        isinstance(descriptor, dict)
        and set(descriptor) == {"path", "sha256"}
        and descriptor.get("path") == "evidence/answer.txt"
        and isinstance(descriptor.get("sha256"), str)
    )


def _evidence_file_stat_ok(target):
    try:
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size > 1_048_576
        ):
            return False
    except OSError:
        return False
    return True


def _evidence_target_text(target):
    try:
        if target.stat().st_size > 1_048_576:
            return None
        return target.read_text().casefold()
    except (OSError, UnicodeError):
        return None


def _evidence_content_ok(text, result):
    if not all(term in text for term in ("exact cover", "set.range", "vacuously")):
        return False
    modulus = result.get("modulus") if isinstance(result, dict) else None
    step = result.get("subgroup_step") if isinstance(result, dict) else None
    if type(modulus) is not int or type(step) is not int:
        return False
    if not (
        re.search(rf"\bz\s*/\s*{modulus}\s*z\b", text)
        or re.search(rf"\b(?:modulus|n)\s*(?:=|is|:)\s*{modulus}\b", text)
    ):
        return False
    subgroup = result.get("subgroup") if isinstance(result, dict) else None
    subgroup_pattern = None
    if isinstance(subgroup, list) and all(type(value) is int for value in subgroup):
        values = r"\s*,\s*".join(str(value) for value in sorted(subgroup))
        subgroup_pattern = rf"\{{\s*{values}\s*\}}"
    if not (
        re.search(rf"\b(?:subgroup\s+)?step\s*(?:=|is|:)\s*{step}\b", text)
        or (subgroup_pattern is not None and re.search(subgroup_pattern, text))
    ):
        return False
    # Reject evidence that contradicts the submitted conclusion. The audit
    # asserts an exact cover with a vacuously-true Set.range predicate, so any
    # negation of those claims invalidates the evidence regardless of keyword
    # presence.
    contradictions = (
        "not an exact cover",
        "not a exact cover",
        "no exact cover",
        "is not an exact cover",
        "set.range is not vacuously",
        "not vacuously true",
        "not vacuously",
        "vacuously false",
    )
    return not any(contradiction in text for contradiction in contradictions)


def _evidence_valid(evidence, result):
    # The published submission schema caps the evidence array at one descriptor,
    # so require exactly one before awarding evidence validity.
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    descriptor = evidence[0]
    if not _evidence_descriptor_ok(descriptor):
        return False
    target = Path("/app") / "evidence" / "answer.txt"
    if target is None:
        return False
    if not _evidence_file_stat_ok(target):
        return False
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(descriptor, expected_path="evidence/answer.txt")
    if target is None:
        return False
    text = _evidence_target_text(target)
    if text is None:
        return False
    return _evidence_content_ok(text, result)


def main():
    submission = _load_exact_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(
        workspace_input_is_bound()
        and contract
        and _valid_result(submission.get("result"), source)
    )
    evidence_valid = bool(
        contract
        and _evidence_valid(submission.get("evidence"), submission.get("result"))
    )
    scope_correct = bool(
        contract and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract
        and isinstance(submission.get("limitations"), list)
        and submission.get("limitations")
        and all(
            _limitation_disclaims_conjecture(item)
            for item in submission.get("limitations", [])
            if isinstance(item, str)
        )
        and all(isinstance(item, str) for item in submission.get("limitations", []))
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    claimed = (
        submission.get("claimed_assurance") if isinstance(submission, dict) else None
    )
    ceiling = expected["maximum_assurance"]
    assurance_within_ceiling = bool(
        contract
        and isinstance(claimed, str)
        and isinstance(ceiling, str)
        and claimed in _ASSURANCE_ORDER
        and ceiling in _ASSURANCE_ORDER
        and _ASSURANCE_ORDER[claimed] <= _ASSURANCE_ORDER[ceiling]
    )
    correct = bool(
        contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_within_ceiling
        and limitations_correct
        and not false_certification
    )
    reward = 0.0 if not correct else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
