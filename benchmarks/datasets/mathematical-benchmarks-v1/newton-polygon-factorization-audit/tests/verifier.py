"""Exact verifier for the author-corrected Newton-polygon lemma audit."""

import json
import math
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = (
    "Dumas's theorem and the corrected general lemma are not machine-formalized."
)
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return any(
        term in text for term in ("dumas", "corrected lemma", "newton polygon")
    ) and any(
        term in text
        for term in ("not formal", "not machine", "not verified", "without formal")
    )


_EXPLANATION_FACTS = {
    "newton_polygon": (
        re.compile(r"\bnewton\s+polygon\b"),
        re.compile(r"\blower\s+hull\b"),
    ),
    "old_right_edge_holds": (
        re.compile(
            r"\bold\b.{0,64}\b(?:right[- ]edge\s+)?hypotheses?\b"
            r".{0,48}\b(?:hold|satisf)"
        ),
        re.compile(
            r"\bright[- ]edge\b.{0,64}\b(?:old\s+)?hypotheses?\b"
            r".{0,48}\b(?:hold|satisf)"
        ),
    ),
    "factor_constant_valuations": (
        re.compile(r"\bfactor\b.{0,96}\bconstant[- ]terms?\b.{0,64}\bvaluation"),
        re.compile(r"\bconstant[- ]term\s+valuations?\b.{0,96}\b(?:factor|both)\b"),
        re.compile(
            r"\bfactor\s+constants?\b.{0,128}\bvaluations?\b"
            r".{0,128}\b(?:non[- ]?zero|neither\b.{0,32}\bzero|not\b.{0,32}\bzero)\b"
        ),
        re.compile(
            r"\bfactor\b.{0,128}\bvaluations?\b.{0,128}"
            r"\b(?:positive|greater\s+than\s+zero|>\s*0)\b"
        ),
    ),
    "corrected_left_edge_fails": (
        re.compile(
            r"\bcorrect(?:ed|ion)?\b.{0,96}\bleft[- ]edge\b"
            r".{0,96}\b(?:fail|does\s+not\s+hold|not\s+satisf)"
        ),
        re.compile(
            r"\bleft[- ]edge\b.{0,96}\b(?:primitiv|gcd|condition)\b"
            r".{0,64}\b(?:fail|greater\s+than\s+one|>\s*1)"
        ),
    ),
}
_EXPLANATION_CONTRADICTIONS = (
    re.compile(r"\bold\s+hypotheses?\b.{0,32}\b(?:fail|do\s+not\s+hold)\b"),
    re.compile(r"\bconstant[- ]term\s+valuations?\b.{0,32}\b(?:zero|nonpositive)\b"),
    re.compile(
        r"\bfactor\s+constants?\b.{0,64}\bvaluations?\b.{0,64}"
        r"\b(?:one\b.{0,16}\bzero|includes?\s+zero)\b"
    ),
    re.compile(r"\bcorrected?\s+left[- ]edge\b.{0,32}\bconditions?\s+hold\b"),
    re.compile(r"(?<!not )(?<!n't )\brefutes?\b.{0,32}\b(?:repair|correction)\b"),
)


def _load_frozen_input() -> dict:
    try:
        frozen = TESTS / "input.json"
        if frozen.is_symlink():
            return {}
        payload = frozen.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, math.isqrt(value) + 1)
    )


def _coefficients(value: object) -> list[int] | None:
    if not isinstance(value, list) or not 3 <= len(value) <= 10:
        return None
    result = []
    for entry in value:
        if not isinstance(entry, str) or len(entry) > 30:
            return None
        try:
            parsed = int(entry)
        except ValueError:
            return None
        if str(parsed) != entry:
            return None
        result.append(parsed)
    return result if result[0] and result[-1] else None


def _multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _valuation(value: int, prime: int) -> int | None:
    if value == 0:
        return None
    result = 0
    value = abs(value)
    while value % prime == 0:
        value //= prime
        result += 1
    return result


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _lower_hull(values: list[int | None]) -> list[tuple[int, int]]:
    hull: list[tuple[int, int]] = []
    for point in (
        (index, value) for index, value in enumerate(values) if value is not None
    ):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull


def _right_edge_hypotheses(values: list[int | None], ell: int, j: int) -> bool:
    v_ell, v_j = values[ell], values[j]
    if v_ell is None or v_ell <= 0 or v_j != 0 or math.gcd(v_ell, j - ell) != 1:
        return False
    slope_size = Fraction(v_ell, j - ell)
    for index in range(1, j):
        if index == ell:
            continue
        value = values[index]
        if value is not None and not slope_size < Fraction(value, j - index):
            return False
    return True


def _corrected_left_conditions(values: list[int | None], ell: int) -> bool:
    v_zero, v_ell = values[0], values[ell]
    if v_zero is None or v_ell is None or v_zero < v_ell:
        return False
    difference = v_zero - v_ell
    if math.gcd(difference, ell) != 1:
        return False
    left_slope = Fraction(difference, ell)
    for index in range(1, ell):
        value = values[index]
        if value is None:
            continue
        if not Fraction(v_zero - value, index) < left_slope:
            return False
    return True


def _certificate_valid(result: object, source: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "prime",
        "factor_left",
        "factor_right",
        "ell",
        "j",
    }:
        return False
    prime, ell, j = result["prime"], result["ell"], result["j"]
    if (
        not isinstance(prime, int)
        or isinstance(prime, bool)
        or prime > 19
        or not _is_prime(prime)
    ):
        return False
    if (
        not isinstance(ell, int)
        or isinstance(ell, bool)
        or not isinstance(j, int)
        or isinstance(j, bool)
    ):
        return False
    left, right = (
        _coefficients(result["factor_left"]),
        _coefficients(result["factor_right"]),
    )
    if left is None or right is None or len(left) < 3 or len(right) < 3:
        return False
    product = _multiply(left, right)
    degree = len(product) - 1
    if degree < 6 or not 2 <= ell < j <= degree:
        return False
    values = [_valuation(coefficient, prime) for coefficient in product]
    if not _right_edge_hypotheses(values, ell, j):
        return False
    if _valuation(left[0], prime) in (None, 0) or _valuation(right[0], prime) in (
        None,
        0,
    ):
        return False
    hull = _lower_hull(values)
    edges = list(pairwise(hull))
    if ((ell, values[ell]), (j, 0)) not in edges:
        return False
    negative_edges = [(a, b) for a, b in edges if b[1] < a[1]]
    if len(negative_edges) < 2 or not any(
        b[0] <= ell for a, b in negative_edges if b[0] != j
    ):
        return False
    if _corrected_left_conditions(values, ell):
        return False
    return bool(
        source.get("old_conclusion") and source.get("corrected_left_edge_conditions")
    )


def _newton_explanation_valid(path: Path) -> bool:
    """Stream evidence and require every documented Newton-polygon fact."""

    matched = dict.fromkeys(_EXPLANATION_FACTS, False)
    contradicted = False
    carry = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                contradicted = contradicted or any(
                    pattern.search(window) for pattern in _EXPLANATION_CONTRADICTIONS
                )
                for name, alternatives in _EXPLANATION_FACTS.items():
                    if not matched[name] and any(
                        pattern.search(window) for pattern in alternatives
                    ):
                        matched[name] = True
                carry = window[-384:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return not contradicted and all(matched.values())


def _raw_submission() -> dict | None:
    """Parse the bounded submission JSON without full schema validation."""
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _evidence_valid(evidence: object) -> bool:
    # The polynomial witness and repair boundary are independently replayed.
    # The public evidence contract promises one bound text artifact that
    # must contain a meaningful Newton-polygon explanation.
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(
        evidence[0], expected_path="evidence/answer.txt", max_bytes=None
    )
    if target is None:
        return False
    return _newton_explanation_valid(target)


def main() -> None:
    input_binding = workspace_input_is_bound()
    raw = _raw_submission()
    submission = load_submission(require_input_binding=False)
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    math_correct = bool(_certificate_valid(result, source))
    evidence_valid = bool(
        isinstance(raw, dict) and _evidence_valid(raw.get("evidence"))
    )
    scope_correct = bool(
        contract
        and isinstance(raw, dict)
        and raw.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        isinstance(raw, dict)
        and isinstance(raw.get("claimed_assurance"), str)
        and raw.get("claimed_assurance") in ALLOWED_ASSURANCES
    )
    limitations_correct = bool(
        contract
        and isinstance(raw, dict)
        and _limitations_valid(raw.get("limitations"))
    )
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    correct = (
        math_correct
        and input_binding
        and contract
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
