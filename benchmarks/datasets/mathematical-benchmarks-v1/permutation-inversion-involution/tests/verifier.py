import itertools
import json
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["FINITE_N_EQUALS_7", "NO_PROOF_ASSISTANT_REPLAY"]
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _json_equal(left, right):
    """Compare two JSON values without Python's bool/int coercion."""

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def frozen():
    return workspace_input_is_bound()


def inv(p):
    return sum(p[i] > p[j] for i in range(7) for j in range(i + 1, 7))


def transform(name, p, multiplier, offset):
    if name == "REVERSE_POSITIONS":
        return tuple(reversed(p))
    if name == "COMPLEMENT_VALUES":
        return tuple(multiplier * x + offset for x in p)
    if name == "INVERSE_PERMUTATION":
        out = [0] * 7
        for i, value in enumerate(p, 1):
            out[value - 1] = i
        return tuple(out)
    return ()


def _traces_valid(traces, name, mul, off):
    if not isinstance(traces, list) or len(traces) != 6:
        return False
    seen = set()
    for row in traces:
        if not isinstance(row, dict) or set(row) != {
            "permutation",
            "transformed",
            "inversions",
            "transformed_inversions",
        }:
            return False
        permutation = row["permutation"]
        if (
            not isinstance(permutation, list)
            or len(permutation) != 7
            or any(type(value) is not int for value in permutation)
        ):
            return False
        p = tuple(permutation)
        if p in seen or sorted(p) != list(range(1, 8)):
            return False
        seen.add(p)
        q = transform(name, p, mul, off)
        if (
            not isinstance(row["transformed"], list)
            or len(row["transformed"]) != 7
            or any(type(value) is not int for value in row["transformed"])
            or row["transformed"] != list(q)
            or type(row["inversions"]) is not int
            or row["inversions"] != inv(p)
            or type(row["transformed_inversions"]) is not int
            or row["transformed_inversions"] != inv(q)
        ):
            return False
    return True


def valid(r):
    keys = {
        "transformation",
        "value_multiplier",
        "value_offset",
        "pair_inversion_sum",
        "fixed_point_count",
        "pair_count",
        "total_inversions",
        "traces",
    }
    if not isinstance(r, dict) or set(r) != keys:
        return False
    name, mul, off = r["transformation"], r["value_multiplier"], r["value_offset"]
    if not (
        (name == "COMPLEMENT_VALUES" and (mul, off) == (-1, 8))
        or (name == "REVERSE_POSITIONS" and (mul, off) == (1, 0))
    ):
        return False
    perms = tuple(itertools.permutations(range(1, 8)))
    images = [transform(name, p, mul, off) for p in perms]
    if any(sorted(q) != list(range(1, 8)) for q in images):
        return False
    fixed = sum(p == q for p, q in zip(perms, images, strict=True))
    if any(
        transform(name, q, mul, off) != p for p, q in zip(perms, images, strict=True)
    ):
        return False
    pair_sum = 21
    if any(inv(p) + inv(q) != pair_sum for p, q in zip(perms, images, strict=True)):
        return False
    if not _traces_valid(r["traces"], name, mul, off):
        return False
    total = sum(map(inv, perms))
    if not all(
        type(r[key]) is int
        for key in (
            "fixed_point_count",
            "pair_count",
            "pair_inversion_sum",
            "total_inversions",
        )
    ):
        return False
    return (
        fixed == r["fixed_point_count"] == 0
        and r["pair_count"] == 2520
        and r["pair_inversion_sum"] == 21
        and r["total_inversions"] == total == 52920
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    structure_valid = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}),
        verification_record="optional",
    )
    contract = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    ev = (
        read_evidence_json(
            s["evidence"][0],
            expected_path="evidence/permutation-involution-certificate.json",
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        if structure_valid
        else None
    )
    math_ok = bool(frozen() and isinstance(s, dict) and valid(s.get("result")))
    evidence_ok = bool(
        ev
        and set(ev) == {"schema_version", "task_id", "result", "limitations"}
        and ev.get("schema_version") == "1"
        and ev.get("task_id") == expected["task_id"]
        and _json_equal(ev.get("result"), s.get("result"))
        and _json_equal(ev.get("limitations"), s.get("limitations"))
    )
    scope_ok = bool(
        structure_valid
        and s.get("scope") == "ALL_PERMUTATIONS_OF_1_THROUGH_7"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(contract and s.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(s, verification_record_bound=False)
    correct = math_ok and evidence_ok and scope_ok and not false_cert
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
