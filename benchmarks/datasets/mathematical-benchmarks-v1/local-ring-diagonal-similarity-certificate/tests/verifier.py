import json
import math
from itertools import permutations
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
LIMITATION = "The verifier certifies only the frozen matrix certificate, not the general local-ring theorem."


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(
            term in text
            for term in (
                "frozen",
                "six-dimensional",
                "6-dimensional",
                "matrix certificate",
            )
        )
        and any(
            term in text for term in ("local-ring", "local ring", "general theorem")
        )
        and any(term in text for term in ("not general", "not the general", "only"))
    )


# The published prose obligation is structural, not verbatim: the explanation
# must affirmatively relate the three certified facts, accept equivalent
# phrasing, and reject contradictory or unrelated text.  Each concept group
# lists affirmative phrases; ``_NEGATIONS`` flags direct contradictions.
_PRODUCTS_AGREE = (
    "products agree",
    "modular products agree",
    "pa and bp coincide",
    "pa and bp agree",
    "pa=bp",
    "pa = bp",
    "pa equals bp",
    "pa and bp match",
    "pa and bp are equal",
    "modular products coincide",
    "modular products are equal",
    "matrix products coincide",
)
_DETERMINANT_UNIT = (
    "determinant is a unit",
    "is a unit",
    "determinant is invertible",
    "is invertible",
    "det is a unit",
    "det is invertible",
    "det(p) is a unit",
    "det(p) is invertible",
    "determinant belongs to the units",
)
_DIAGONAL_MATCH = (
    "unit entries",
    "unit entry",
    "diagonal entries match",
    "matched diagonal pair",
    "diagonal pair",
    "matched pairs agree",
    "diagonal pairs agree",
    "matched diagonal pairs agree",
    "diagonal matching",
    "matches the diagonal",
    "match the diagonal",
)
_CONCEPT_GROUPS = (_PRODUCTS_AGREE, _DETERMINANT_UNIT, _DIAGONAL_MATCH)
_NEGATIONS = (
    "not a unit",
    "not invertible",
    "non-unit",
    "nonunit",
    "do not agree",
    "does not agree",
    "do not match",
    "does not match",
    "do not coincide",
    "does not coincide",
)


def sign(p):
    return -1 if sum(p[i] > p[j] for i in range(6) for j in range(i + 1, 6)) % 2 else 1


def product_diag(matrix, d, left):
    return [
        [(matrix[i][j] * (d[i] if left else d[j])) % 125 for j in range(6)]
        for i in range(6)
    ]


def valid(r, d):
    if not isinstance(r, dict) or set(r) != {
        "modulus",
        "PA",
        "BP",
        "determinant_modulus",
        "unit_permutation",
        "permutation_sign",
        "signed_term_modulus",
        "matched_pairs",
    }:
        return False
    matrix = d["P"]
    pa = product_diag(matrix, d["a_diagonal"], False)
    bp = product_diag(matrix, d["b_diagonal"], True)
    det = (
        sum(
            sign(p) * math.prod(matrix[i][p[i]] for i in range(6))
            for p in permutations(range(6))
        )
        % 125
    )
    perm = r.get("unit_permutation")
    if (
        not isinstance(perm, list)
        or any(type(value) is not int for value in perm)
        or sorted(perm) != list(range(6))
    ):
        return False
    entries = [matrix[i][perm[i]] for i in range(6)]
    pairs = [
        {
            "row": i,
            "column": perm[i],
            "b_value": d["b_diagonal"][i],
            "a_value": d["a_diagonal"][perm[i]],
            "unit_entry": entries[i],
        }
        for i in range(6)
    ]
    return (
        r.get("modulus") == 125
        and r.get("PA") == pa
        and r.get("BP") == bp
        and pa == bp
        and r.get("determinant_modulus") == det
        and math.gcd(det, 125) == 1
        and all(math.gcd(x, 125) == 1 for x in entries)
        and r.get("permutation_sign") == sign(perm)
        and r.get("signed_term_modulus") == sign(perm) * math.prod(entries) % 125
        and isinstance(r.get("matched_pairs"), list)
        and len(r["matched_pairs"]) == 6
        and all(
            isinstance(pair, dict)
            and set(pair) == {"row", "column", "b_value", "a_value", "unit_entry"}
            and all(type(value) is int for value in pair.values())
            for pair in r["matched_pairs"]
        )
        and sorted(r["matched_pairs"], key=lambda pair: pair.get("row", -1)) == pairs
        and all(x["a_value"] == x["b_value"] for x in pairs)
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


def evidence_ok(e):
    # The typed matrix certificate is replayed independently.  The public
    # evidence contract requires one digest-bound text explanation whose
    # content affirmatively states the certified relationships; equivalent
    # phrasing is accepted and contradictory or unrelated text is rejected.
    if not isinstance(e, list) or len(e) != 1:
        return False
    path = resolve_evidence(e[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    return _explanation_is_valid(path)


def _reject_nonfinite(token):
    raise ValueError(f"non-finite JSON number: {token}")


def raw_submission():
    """Parse the bounded submission JSON before envelope validation.

    Used to preserve independently evaluable diagnostics (mathematical
    correctness, evidence, assurance, and unauthorized ``VERIFIED`` claims)
    when a non-result envelope field violates the schema.
    """
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def main():
    raw = raw_submission()
    e = json.loads((T / "expected.json").read_text())
    d = json.loads((T / "input.json").read_text())
    input_binding = workspace_input_is_bound()
    s = load_submission(W / "submission.json", require_input_binding=False)
    c = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    # Mathematical correctness and evidence are evaluated from the bounded raw
    # payload so a malformed non-result field cannot erase them; protocol and
    # scope remain tied to the schema-valid envelope.
    r = raw.get("result") if isinstance(raw, dict) else None
    m = valid(r, d)
    ev = bool(isinstance(raw, dict) and evidence_ok(raw.get("evidence")))
    a = bool(isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED")
    sc = bool(
        isinstance(s, dict)
        and s.get("scope") == "frozen-six-dimensional-Z-mod-125-certificate"
        and s.get("completeness") == "COMPLETE"
        and _limitations_valid(s.get("limitations"))
    )
    f = false_verified_claim(raw, verification_record_bound=False)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(c)),
                "input_binding": float(input_binding),
                "correctness": float(m),
                "evidence_validity": float(ev),
                "scope_accuracy": float(sc),
                "assurance_calibration": float(a),
                "reward": aggregate_reward(
                    correctness=m,
                    evidence_validity=ev,
                    scope_accuracy=sc,
                    assurance_calibration=a,
                    false_certification=f,
                    protocol_ok=bool(c and input_binding),
                    soft_assurance=True,
                ),
                "false_certification": f,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
