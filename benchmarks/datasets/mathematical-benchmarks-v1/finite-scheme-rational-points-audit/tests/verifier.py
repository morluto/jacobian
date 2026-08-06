import itertools
import json
import re
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
P = 5
LIMITATION = "The verifier certifies one finite affine countermodel over F_5, not a general theorem about schemes or functors of points."


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(term in text for term in ("f_5", "f5", "finite affine", "countermodel"))
        and any(term in text for term in ("scheme", "functor", "rational point"))
        and any(
            term in text
            for term in ("not general", "not a general", "only", "one finite")
        )
    )


# Published reward-bearing prose obligations for evidence/answer.txt.  The
# verifier checks that the evidence text states all three facts, accepting
# mathematically equivalent phrasing while rejecting unrelated or empty text.
_EVIDENCE_FACTS = {
    "rational_points": (re.compile(r"\brational\s+points?\b"),),
    "same_nonempty_set": (
        re.compile(r"\b(?:same|equal)\b.{0,64}\brational\s+points?\b"),
        re.compile(r"\b(?:bijection|bijective)\b"),
        re.compile(r"\beach\b.{0,32}\bthree\s+rational\s+points?\b"),
    ),
    "induced_map": (
        re.compile(r"\binduced\s+map\b"),
        re.compile(r"\bpullback\b"),
    ),
    "nonzero_nilpotent": (
        re.compile(r"\bnon[- ]?zero\b.{0,64}\bnilpotent\b"),
        re.compile(r"\bnilpotent\b.{0,64}\bnon[- ]?zero\b"),
    ),
    "order_three": (
        re.compile(r"\border[- ](?:three|3)\b"),
        re.compile(r"\bcube\b.{0,32}\bzero\b"),
        re.compile(r"\bthird\s+power\b.{0,32}\bzero\b"),
    ),
    "b_reduced": (
        re.compile(r"\bb\b.{0,32}\breduced\b"),
        re.compile(r"\bsecond\s+(?:algebra|scheme)\b.{0,32}\breduced\b"),
    ),
    "not_isomorphic": (
        re.compile(r"\bnot\s+isomorphic\b"),
        re.compile(r"\bnon[- ]?isomorphic\b"),
        re.compile(r"\b(?:cannot|can't)\s+be\s+isomorphic\b"),
        re.compile(r"\bno\s+isomorphism\b"),
    ),
}
EXPECTED_COLUMNS = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]


def tensor(dimension: int, products: dict[tuple[int, int], int]):
    value = [[[0] * dimension for _ in range(dimension)] for _ in range(dimension)]
    for (left, right), output in products.items():
        value[left][right][output] = 1
        value[right][left][output] = 1
    return value


def canonical_a():
    return tensor(5, {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 3): 3, (0, 4): 4, (3, 3): 4})


def canonical_b():
    return tensor(3, {(0, 0): 0, (1, 1): 1, (2, 2): 2})


def multiply(left, right, table):
    dimension = len(left)
    return [
        sum(
            left[i] * right[j] * table[i][j][k]
            for i in range(dimension)
            for j in range(dimension)
        )
        % P
        for k in range(dimension)
    ]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True)) % P


def algebra_maps(table, unit):
    dimension = len(unit)
    found = []
    for candidate in itertools.product(range(P), repeat=dimension):
        if dot(candidate, unit) != 1:
            continue
        if all(
            dot(candidate, table[i][j]) == candidate[i] * candidate[j] % P
            for i in range(dimension)
            for j in range(dimension)
        ):
            found.append(list(candidate))
    return found


def _is_int_vector(value, length, lo, hi):
    """Validate an exact-length integer vector before sorting or arithmetic."""

    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(x) is int and lo <= x <= hi for x in value)
    )


def evidence_text_is_valid(path: Path) -> bool:
    """Stream evidence and require every published mathematical fact."""

    matched = dict.fromkeys(_EVIDENCE_FACTS, False)
    carry = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                for name, alternatives in _EVIDENCE_FACTS.items():
                    if not matched[name] and any(
                        pattern.search(window) for pattern in alternatives
                    ):
                        matched[name] = True
                carry = window[-256:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return all(matched.values())


def valid_morphism(columns, a_table, b_table, a_unit, b_unit):
    if not isinstance(columns, list) or len(columns) != 3:
        return False
    image_unit = [
        sum(b_unit[j] * columns[j][i] for j in range(3)) % P for i in range(5)
    ]
    if image_unit != a_unit:
        return False
    for i in range(3):
        for j in range(3):
            image_product = [
                sum(b_table[i][j][k] * columns[k][m] for k in range(3)) % P
                for m in range(5)
            ]
            if multiply(columns[i], columns[j], a_table) != image_product:
                return False
    return True


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "field_prime",
        "a_unit",
        "b_unit",
        "a_multiplication",
        "b_multiplication",
        "morphism_columns",
        "a_points",
        "b_points",
        "induced_point_map",
        "nilpotent",
        "b_reduced",
    }:
        return False
    a_table, b_table = canonical_a(), canonical_b()
    a_unit, b_unit = [1, 1, 1, 0, 0], [1, 1, 1]
    a_points, b_points = algebra_maps(a_table, a_unit), algebra_maps(b_table, b_unit)
    if (
        result["field_prime"] != P
        or result["a_unit"] != a_unit
        or result["b_unit"] != b_unit
    ):
        return False
    if result["a_multiplication"] != a_table or result["b_multiplication"] != b_table:
        return False
    columns = result["morphism_columns"]
    if columns != EXPECTED_COLUMNS or not valid_morphism(
        columns, a_table, b_table, a_unit, b_unit
    ):
        return False
    a_pts = result["a_points"]
    b_pts = result["b_points"]
    if not (
        isinstance(a_pts, list)
        and isinstance(b_pts, list)
        and all(_is_int_vector(p, 5, 0, P - 1) for p in a_pts)
        and all(_is_int_vector(p, 3, 0, P - 1) for p in b_pts)
    ):
        return False
    if sorted(a_pts) != sorted(a_points) or sorted(b_pts) != sorted(b_points):
        return False
    induced = []
    for point in a_pts:
        pullback = [dot(point, column) for column in columns]
        if pullback not in b_pts:
            return False
        induced.append(b_pts.index(pullback))
    if result["induced_point_map"] != induced or sorted(induced) != [0, 1, 2]:
        return False
    witness = result["nilpotent"]
    vector = witness.get("vector") if isinstance(witness, dict) else None
    if not _is_int_vector(vector, 5, 0, P - 1) or vector == [0] * 5:
        return False
    power2 = multiply(vector, vector, a_table)
    power3 = multiply(power2, vector, a_table)
    if witness != {
        "vector": vector,
        "power2": power2,
        "power3": power3,
        "exact_order": 3,
    }:
        return False
    if power2 == [0] * 5 or power3 != [0] * 5:
        return False
    b_has_nilpotent = any(
        multiply(
            multiply(list(v), list(v), b_table),
            multiply(list(v), list(v), b_table),
            b_table,
        )
        == [0] * 3
        for v in itertools.product(range(P), repeat=3)
        if any(v)
    )
    return result["b_reduced"] is True and not b_has_nilpotent


def evidence_ok(evidence):
    # The finite-algebra certificate is independently replayed from the typed
    # result.  The public evidence contract requires one bound text file that
    # states the three published mathematical obligations.
    if not evidence_list_is_bound(evidence):
        return False
    resolved = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if resolved is None:
        return False
    return evidence_text_is_valid(resolved)


def raw_submission():
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    raw = raw_submission()
    expected = json.loads((TESTS / "expected.json").read_text())
    input_binding = workspace_input_is_bound()
    submission = load_submission(
        WORKSPACE / "submission.json", require_input_binding=False
    )
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    mathematical = valid_result(result)
    evidence = bool(isinstance(raw, dict) and evidence_ok(raw.get("evidence")))
    scope = bool(
        isinstance(raw, dict)
        and raw.get("scope") == "finite-affine-schemes-over-F5-certificate"
        and raw.get("completeness") == "COMPLETE"
        and _limitations_valid(raw.get("limitations"))
    )
    assurance = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED"
    )
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    accepted = bool(
        contract
        and input_binding
        and mathematical
        and evidence
        and scope
        and not false_certification
    )
    (Path("/logs/verifier")).mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": 0 if not accepted else 0.9 + 0.1 * assurance,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
