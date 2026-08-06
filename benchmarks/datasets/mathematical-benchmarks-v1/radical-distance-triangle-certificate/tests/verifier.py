import json
import math
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["TRIANGLE_INEQUALITY_TRUSTED", "NO_PROOF_ASSISTANT_REPLAY"]
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _json_equal(left, right):
    """Compare two JSON values without Python's bool/int coercion.

    Python treats ``True == 1`` as equal, so a certificate that replaces an
    integer ``1`` with boolean ``true`` would pass ``==`` despite not being an
    exact copy. Serializing both values to canonical JSON distinguishes them.
    """

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _integer_value(value):
    """Accept any schema-valid integral JSON number while rejecting booleans.

    JSON Schema's ``integer`` type accepts numbers with a zero fractional part
    (e.g. ``-1.0``), so the verifier validates mathematical integrality rather
    than requiring Python's ``int`` representation. Booleans are rejected
    because ``True == 1`` would otherwise spoof a unit coefficient.
    """

    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def frozen():
    return workspace_input_is_bound()


def _coefficient_record_is_valid(record):
    """Validate an expanded radicand coefficient record, accepting schema-valid
    integral JSON numbers while rejecting booleans."""

    return bool(
        isinstance(record, dict)
        and set(record) == {"a2", "b2", "a_sqrt2", "b_sqrt2", "constant"}
        and all(_integer_value(record[key]) is not None for key in record)
    )


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "method",
        "scaled_centers",
        "expanded_radicands",
        "center_distance_squared",
        "lower_bound",
        "equality_witness",
    }:
        return False
    centers = r.get("scaled_centers")
    expansions = r.get("expanded_radicands")
    witness = r.get("equality_witness")
    if not (
        isinstance(centers, list)
        and len(centers) == 2
        and all(
            isinstance(row, list)
            and len(row) == 2
            and all(_integer_value(value) is not None for value in row)
            for row in centers
        )
        and isinstance(expansions, list)
        and len(expansions) == 2
        and all(_coefficient_record_is_valid(item) for item in expansions)
        and isinstance(witness, list)
        and len(witness) == 2
        and all(_integer_value(value) is not None for value in witness)
        and _integer_value(r.get("center_distance_squared")) is not None
        and _integer_value(r.get("lower_bound")) is not None
    ):
        return False
    int_centers = [[_integer_value(value) for value in row] for row in centers]
    int_witness = [_integer_value(value) for value in witness]
    center_distance_squared = _integer_value(r["center_distance_squared"])
    lower_bound = _integer_value(r["lower_bound"])
    if r["method"] != "DISTANCE_TRIANGLE_INEQUALITY":
        return False
    canonical_centers = [[-1, -1], [1, 1]]
    if sorted(int_centers) != sorted(canonical_centers):
        return False
    derived = [
        {
            "a2": 1,
            "b2": 1,
            "a_sqrt2": -center[0],
            "b_sqrt2": -center[1],
            "constant": (center[0] ** 2 + center[1] ** 2) // 2,
        }
        for center in int_centers
    ]
    frozen_radicands = json.loads((T / "input.json").read_text())["radicands"]

    def key(item):
        return json.dumps(item, sort_keys=True, separators=(",", ":"))

    int_expansions = [
        {k: _integer_value(v) for k, v in item.items()} for item in expansions
    ]
    if not all(
        key(exp) == key(der) for exp, der in zip(int_expansions, derived, strict=True)
    ):
        return False
    if sorted(derived, key=key) != sorted(frozen_radicands, key=key):
        return False
    separation_squared = (
        sum((int_centers[1][axis] - int_centers[0][axis]) ** 2 for axis in range(2))
        // 2
    )
    if (
        center_distance_squared != separation_squared
        or separation_squared != 4
        or lower_bound**2 != separation_squared
        or lower_bound < 0
    ):
        return False
    a, b = int_witness
    return a == b and -1 <= a <= 1 and 2 * (a * a + b * b) + 2 == 2


def main():
    e = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    structure_valid = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}),
        verification_record="optional",
    )
    c = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    ev = (
        read_evidence_json(
            s["evidence"][0],
            expected_path="evidence/radical-distance-certificate.json",
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        if structure_valid
        else None
    )
    m = bool(frozen() and isinstance(s, dict) and valid(s.get("result")))
    v = bool(
        ev
        and set(ev) == {"schema_version", "task_id", "result", "limitations"}
        and ev.get("schema_version") == "1"
        and ev.get("task_id") == e["task_id"]
        and _json_equal(ev.get("result"), s.get("result"))
        and _json_equal(ev.get("limitations"), s.get("limitations"))
    )
    q = bool(
        structure_valid
        and s.get("scope") == "ALL_REAL_A_B"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == LIMITATIONS
    )
    a = bool(c and s.get("claimed_assurance") == "COMPUTED")
    f = false_verified_claim(s, verification_record_bound=False)
    ok = m and v and q and a and not f
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(m),
                "evidence_validity": float(v),
                "scope_accuracy": float(q),
                "assurance_calibration": float(a),
                "reward": 0 if not ok else 0.9 + 0.1 * a,
                "false_certification": f,
            }
        )
    )


if __name__ == "__main__":
    main()
