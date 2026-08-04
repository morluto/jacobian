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

ORDER = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
TARGET = [[0, 0, 0, 0, 1, -2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, -3]]


def _integer(value):
    return type(value) is int


def _canonical_fraction(value):
    if not isinstance(value, str):
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fraction if str(fraction) == value else None


def _valid_vector(vector):
    return (
        isinstance(vector, list)
        and len(vector) == 10
        and all(_integer(value) for value in vector)
    )


def _valid_potential(potential):
    if not isinstance(potential, dict) or set(potential) != {"terms"}:
        return False
    terms = potential["terms"]
    if not isinstance(terms, list):
        return False
    seen = set()
    for term in terms:
        if not isinstance(term, dict) or set(term) != {
            "coefficient",
            "x_power",
            "y_power",
        }:
            return False
        coefficient = _canonical_fraction(term["coefficient"])
        x_power, y_power = term["x_power"], term["y_power"]
        if (
            coefficient is None
            or not _integer(x_power)
            or not _integer(y_power)
            or not 0 <= x_power <= 4
            or not 0 <= y_power <= 4
            or coefficient == 0
            or (x_power, y_power) in seen
        ):
            return False
        seen.add((x_power, y_power))
    return True


def _valid_scope(scope):
    if not isinstance(scope, str):
        return False
    normalized = " ".join(scope.casefold().split())
    required = ("closed polynomial one-form", "swapped", "degree at most three", "r2")
    if not all(term in normalized for term in required):
        return False
    return not re.search(
        r"(?:\b(?:not|no|without|excluding)\b[^.]*\bclosed polynomial one-form\b|"
        r"\bclosed polynomial one-form\b[^.]*\b(?:not|no|without|excluding)\b)",
        normalized,
    )


def _valid_limitations(limitations):
    if not isinstance(limitations, list) or not limitations:
        return False
    normalized = " ".join(
        " ".join(item.casefold().split())
        for item in limitations
        if isinstance(item, str)
    )
    return bool(
        normalized
        and re.search(r"poincare lemma[^.]*\bnot\b[^.]*\bcheck", normalized)
        and re.search(r"arbitrary smooth forms[^.]*\bnot\b[^.]*\bcheck", normalized)
    )


def _valid_derivation_evidence(evidence):
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    if len(lines) != 6:
        return False
    fields = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if not separator or key in fields or not value.strip():
            return False
        fields[key] = " ".join(value.casefold().split())
    if set(fields) != {
        "CHAIN_RULE",
        "CONSTRAINTS",
        "RANK",
        "DIMENSION",
        "POTENTIALS",
        "LIMITATION",
    }:
        return False
    compact = {key: value.replace(" ", "") for key, value in fields.items()}
    return (
        "d/dxf(y,x)" in compact["CHAIN_RULE"]
        and "f_y(y,x)" in compact["CHAIN_RULE"]
        and all(
            term in compact["CONSTRAINTS"]
            for term in ("a_11-2*a_02=0", "a_21-3*a_03=0")
        )
        and fields["RANK"] == "2"
        and fields["DIMENSION"] == "8"
        and "every" in fields["POTENTIALS"]
        and "f_x=f(x,y)" in compact["POTENTIALS"]
        and "f_y=f(y,x)" in compact["POTENTIALS"]
        and all(
            term in fields["LIMITATION"]
            for term in ("poincare lemma", "arbitrary smooth forms", "not")
        )
    )


def rank(matrix):
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for i in range(len(rows)):
            if i != pivot_row and rows[i][column]:
                scale = rows[i][column]
                rows[i] = [
                    value - scale * p
                    for value, p in zip(rows[i], rows[pivot_row], strict=True)
                ]
        pivot_row += 1
    return pivot_row


def derivative(terms, axis):
    result = {}
    for term in terms:
        coefficient = Fraction(term["coefficient"])
        x_power, y_power = term["x_power"], term["y_power"]
        power = x_power if axis == 0 else y_power
        if not power:
            continue
        key = (x_power - 1, y_power) if axis == 0 else (x_power, y_power - 1)
        result[key] = result.get(key, Fraction()) + coefficient * power
    return {key: value for key, value in result.items() if value}


def valid_result(result):
    try:
        if not isinstance(result, dict) or set(result) != {
            "constraints",
            "rank",
            "dimension",
            "basis",
            "potentials",
        }:
            return False
        constraints, basis = result["constraints"], result["basis"]
        potentials = result["potentials"]
        if (
            not isinstance(constraints, list)
            or len(constraints) != 2
            or not all(_valid_vector(row) for row in constraints)
            or not isinstance(basis, list)
            or len(basis) != 8
            or not all(_valid_vector(vector) for vector in basis)
            or not isinstance(potentials, list)
            or len(potentials) != 8
            or not all(_valid_potential(potential) for potential in potentials)
            or not _integer(result["rank"])
            or not _integer(result["dimension"])
        ):
            return False
        valid = (
            result["rank"] == 2
            and result["dimension"] == 8
            and rank(constraints) == 2
            and rank(constraints + TARGET) == 2
            and rank(basis) == 8
            and all(
                all(sum(row[i] * vector[i] for i in range(10)) == 0 for row in TARGET)
                for vector in basis
            )
        )
        for vector, potential in zip(basis, potentials, strict=True):
            expected = {
                ORDER[i]: Fraction(value) for i, value in enumerate(vector) if value
            }
            valid = valid and derivative(potential["terms"], 0) == expected
            valid = valid and derivative(potential["terms"], 1) == {
                (y, x): value for (x, y), value in expected.items()
            }
        return valid
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/closed-one-form-polynomial-classification",
        conclusion="SOURCE_CHAIN_RULE_REPAIRED",
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(valid_result(data.get("result")))
    evidence_valid = bool(
        evidence_list_is_bound(
            data.get("evidence"), expected_path="evidence/answer.txt"
        )
        and _valid_derivation_evidence(data.get("evidence"))
    )
    scope_correct = bool(contract and _valid_scope(data.get("scope")))
    limitations_correct = _valid_limitations(data.get("limitations"))
    assurance_correct = (
        data.get("claimed_assurance") == "COMPUTED" and limitations_correct
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
    )
    reward = 1.0 if correct else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": bool(false_certification),
            }
        )
    )


if __name__ == "__main__":
    main()
