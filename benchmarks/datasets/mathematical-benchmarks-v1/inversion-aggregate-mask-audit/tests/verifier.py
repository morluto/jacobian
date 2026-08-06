import itertools
import json
from pathlib import Path

from verifier_support import (
    ASSURANCE_LEVELS,
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["FINITE_N4_REPLAY", "NO_LEAN_ELABORATION"]


def _json_equal(a: object, b: object) -> bool:
    """Structural JSON equality that rejects type-mismatched scalars.

    Python's ``==`` treats ``True == 1`` and ``6.0 == 6`` as equal, so an
    evidence file that substitutes a float for an integer field (or a boolean
    for an integer) would match the submitted result. This helper recursively
    requires exact scalar types and element-wise equality for containers.
    """

    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return type(a) is type(b) and a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is None and b is None
    return False


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def counts(p):
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    intended = sum(p[i] > p[j] for i, j in pairs)
    implemented = sum(p[i] <= p[j] for i, j in pairs)
    return implemented, intended


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "witness_permutation",
        "implemented_count",
        "intended_count",
        "implemented_aggregate",
        "intended_aggregate",
        "pair_count",
        "complement_relation",
    }:
        return False
    p = r["witness_permutation"]
    if (
        not isinstance(p, list)
        or len(p) != 4
        or not all(type(value) is int for value in p)
        or sorted(p) != list(range(4))
    ):
        return False
    if not all(
        type(r[key]) is int
        for key in (
            "implemented_count",
            "intended_count",
            "implemented_aggregate",
            "intended_aggregate",
            "pair_count",
        )
    ):
        return False
    if (
        r["complement_relation"]
        != "IMPLEMENTED_PLUS_INTENDED_EQUALS_PAIR_COUNT_PER_PERMUTATION"
    ):
        return False
    ic, tc = counts(p)
    perms = list(itertools.permutations(range(4)))
    ia = sum(counts(q)[0] for q in perms)
    ta = sum(counts(q)[1] for q in perms)
    return (
        ic != tc
        and r["implemented_count"] == ic
        and r["intended_count"] == tc
        and r["implemented_aggregate"] == ia
        and r["intended_aggregate"] == ta
        and r["pair_count"] == 6
        and ia == ta == 72
        and all(sum(counts(q)) == 6 for q in perms)
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    envelope_valid = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ASSURANCE_LEVELS,
        verification_record="forbidden",
    )
    ev = (
        read_evidence_json(
            s["evidence"][0], expected_path="evidence/inversion-audit.json"
        )
        if s is not None and isinstance(s.get("evidence"), list) and s["evidence"]
        else None
    )
    math_ok = bool(s is not None and frozen() and valid(s.get("result")))
    evidence_ok = bool(
        envelope_valid
        and ev
        and set(ev) == {"schema_version", "task_id", "result", "limitations"}
        and ev.get("schema_version") == "1"
        and ev.get("task_id") == expected["task_id"]
        and _json_equal(ev.get("result"), s.get("result"))
        and _json_equal(ev.get("limitations"), LIMITATIONS)
    )
    scope_ok = bool(
        s is not None
        and isinstance(s.get("claimed_assurance"), str)
        and s.get("scope") == "FROZEN_N4_ALL_PERMUTATIONS"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        contract and s is not None and s.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(s, verification_record_bound=False)
    correct = (
        contract
        and math_ok
        and evidence_ok
        and scope_ok
        and assurance_ok
        and not false_cert
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance_ok,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
