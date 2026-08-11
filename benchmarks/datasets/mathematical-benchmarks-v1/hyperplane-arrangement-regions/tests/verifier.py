from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from functools import reduce
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/hyperplane-arrangement-regions"
CONCLUSION = "REGION_COUNT_CERTIFIED"
SCOPE = "the ten frozen cube and tetrahedron face planes"
PLANES = {
    "cube_x0": (1, 0, 0, 0),
    "cube_x1": (1, 0, 0, 1),
    "cube_y0": (0, 1, 0, 0),
    "cube_y1": (0, 1, 0, 1),
    "cube_z0": (0, 0, 1, 0),
    "cube_z1": (0, 0, 1, 1),
    "tetra_A1C1D1": (0, 0, 1, 1),
    "tetra_BC1D1": (0, 1, -1, 0),
    "tetra_BA1D1": (1, 0, 1, 1),
    "tetra_BA1C1": (1, -1, 1, 1),
}
PLANE_LABELS = tuple(PLANES)
_NEGATED_PROOF_ASSISTANT = re.compile(
    r"\b(?:not|no|without|doesn['']?t|does\s+not|cannot|never)\b"
    r"[^.;\n]{0,80}\b(?:verif(?:y|ied|ies)|formaliz(?:e|ed|es)|"
    r"prove(?:s|d)|certif(?:y|ied|ies))\b[^.;\n]{0,80}"
    r"\b(?:proof assistant|lean)\b|"
    r"\b(?:not|no|without|doesn['']?t|does\s+not|cannot|never)\b"
    r"[^.;\n]{0,30}\b(?:proof assistant|lean)\b",
    re.I,
)
_AFFIRMATIVE_PROOF_ASSISTANT = re.compile(
    r"\b(?:verif(?:y|ied|ies)|formaliz(?:e|ed|es)|prove(?:s|d)|certif(?:y|ied|ies))\b"
    r"[^.;\n]{0,80}\b(?:proof assistant|lean)\b|"
    r"\b(?:proof assistant|lean)\b[^.;\n]{0,80}"
    r"\b(?:verif(?:y|ied|ies)|formaliz(?:e|ed|es)|prove(?:s|d)|certif(?:y|ied|ies))\b",
    re.I,
)
_EVIDENCE_STEMS = ("duplic", "restrict", "line", "region")


def _limitation_is_valid(value: str) -> bool:
    """Accept limitations that genuinely negate proof-assistant verification.

    The negation must appear within 30 chars before the proof-assistant phrase
    in the same clause, so an unrelated negation elsewhere does not rescue an
    affirmative proof-assistant claim.
    """

    normalized = re.sub(r"[-_\s]+", " ", value.casefold()).strip()
    if "proof assistant" not in normalized and "lean" not in normalized:
        return False
    for clause in re.split(r"[.;\n]", normalized):
        if _AFFIRMATIVE_PROOF_ASSISTANT.search(clause) and not (
            _NEGATED_PROOF_ASSISTANT.search(clause)
        ):
            return False
    return _NEGATED_PROOF_ASSISTANT.search(normalized) is not None


def _canonical(values: tuple[int, ...]) -> tuple[int, ...] | None:
    if not any(values[:-1]):
        return None
    divisor = reduce(math.gcd, (abs(v) for v in values if v), 0)
    normalized = tuple(v // divisor for v in values)
    first = next(v for v in normalized if v)
    return tuple(-v for v in normalized) if first < 0 else normalized


def _rational_line(
    values: tuple[Fraction, Fraction, Fraction],
) -> tuple[int, int, int] | None:
    a, b, c = values
    if a == b == 0:
        return None
    lcm = math.lcm(a.denominator, b.denominator, c.denominator)
    raw = (int(a * lcm), int(b * lcm), int(c * lcm))
    return _canonical(raw)


def _restriction(
    current: tuple[int, ...], prior: tuple[int, ...]
) -> tuple[int, int, int] | None:
    normal = current[:3]
    pivot = next(i for i, value in enumerate(normal) if value)
    free = [i for i in range(3) if i != pivot]
    p = Fraction(normal[pivot])
    q = prior[:3]
    coefficients = tuple(
        Fraction(q[index]) - Fraction(q[pivot] * normal[index], p) for index in free
    )
    rhs = Fraction(prior[3]) - Fraction(q[pivot] * current[3], p)
    return _rational_line((coefficients[0], coefficients[1], rhs))


def _line_regions(lines: set[tuple[int, int, int]]) -> int:
    regions = 1
    previous: list[tuple[int, int, int]] = []
    for a, b, c in sorted(lines):
        points: set[tuple[Fraction, Fraction]] = set()
        for d, e, f in previous:
            determinant = a * e - b * d
            if determinant:
                points.add(
                    (
                        Fraction(c * e - b * f, determinant),
                        Fraction(a * f - c * d, determinant),
                    )
                )
        regions += len(points) + 1
        previous.append((a, b, c))
    return regions


def _increments(order: list[tuple[str, tuple[int, ...]]]) -> list[int]:
    unique: list[tuple[int, ...]] = []
    increments: list[int] = []
    for _, plane in order:
        if plane in unique:
            increments.append(0)
            continue
        lines = {
            line for prior in unique if (line := _restriction(plane, prior)) is not None
        }
        increments.append(_line_regions(lines))
        unique.append(plane)
    return increments


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        source = json.loads(hidden)
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and source["source"]["revision"]
            == "c705198ae1043810b1e1693bd879250b51a7a523"
            and source["source"]["row"] == 20
        )
    except (OSError, ValueError, KeyError):
        return False


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "regions",
        "ordered_planes",
        "duplicate_groups",
    }:
        return False
    entries = value["ordered_planes"]
    if not isinstance(entries, list) or len(entries) != 10:
        return False
    order: list[tuple[str, tuple[int, ...]]] = []
    declared: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "label",
            "coefficients",
            "increment",
        }:
            return False
        label = entry["label"]
        coefficients = entry["coefficients"]
        increment = entry["increment"]
        if (
            not isinstance(label, str)
            or label not in PLANES
            or not isinstance(coefficients, list)
            or len(coefficients) != 4
            or any(type(item) is not int for item in coefficients)
            or type(increment) is not int
        ):
            return False
        canonical = _canonical(tuple(coefficients))
        if canonical != PLANES[label]:
            return False
        order.append((label, canonical))
        declared.append(increment)
    if {label for label, _ in order} != set(PLANES) or len(
        {label for label, _ in order}
    ) != 10:
        return False
    actual = _increments(order)
    duplicate = value["duplicate_groups"]
    valid_group = {"cube_z1", "tetra_A1C1D1"}
    return bool(
        type(value["regions"]) is int
        and declared == actual
        and value["regions"] == 1 + sum(actual) == 64
        and isinstance(duplicate, list)
        and len(duplicate) == 1
        and isinstance(duplicate[0], list)
        and len(duplicate[0]) == 2
        and all(type(member) is str for member in duplicate[0])
        and set(duplicate[0]) == valid_group
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
        markers = [
            line[12:].strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        bound = json.loads(markers[0]) if len(markers) == 1 else None
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False
    prose = "\n".join(
        line for line in text.splitlines() if not line.startswith("RESULT_JSON:")
    ).strip()
    return bound == result and len(prose) >= 32


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
    envelope = isinstance(submission, dict)
    mathematical = bool(
        envelope and _source_is_bound() and _result(submission.get("result"))
    )
    evidence_valid = bool(
        envelope and _evidence(submission.get("evidence"), submission.get("result"))
    )
    scope_accurate = bool(
        protocol
        and envelope
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and any(
            _limitation_is_valid(item)
            for item in submission.get("limitations", [])
            if isinstance(item, str)
        )
    )
    assurance_calibrated = bool(
        envelope
        and submission.get("claimed_assurance") == "COMPUTED"
        and not false_certification
    )
    correct_score = 1.0 if mathematical else 0.0
    evidence_score = 1.0 if evidence_valid else 0.0
    scope_score = 1.0 if scope_accurate else 0.0
    assurance_score = 1.0 if assurance_calibrated else 0.0
    reward = (
        min(correct_score, evidence_score, scope_score, assurance_score)
        if protocol
        else 0.0
    )
    return {
        "correctness": correct_score,
        "evidence_validity": evidence_score,
        "scope_accuracy": scope_score,
        "assurance_calibration": assurance_score,
        "reward": reward,
        "false_certification": false_certification,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
