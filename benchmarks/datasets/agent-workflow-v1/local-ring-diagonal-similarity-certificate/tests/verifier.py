import json
import math
from itertools import permutations
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATION = "The verifier certifies only the frozen matrix certificate, not the general local-ring theorem."


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
    if not isinstance(perm, list) or sorted(perm) != list(range(6)):
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
        and r.get("matched_pairs") == pairs
        and all(x["a_value"] == x["b_value"] for x in pairs)
    )


def evidence_ok(e, r):
    if not evidence_list_is_bound(e):
        return False
    p = resolve_evidence(e[0], expected_path="evidence/answer.txt")
    if p is None:
        return False
    try:
        text = p.read_text()
        lines = [x for x in text.splitlines() if x.startswith("RESULT_JSON:")]
        return (
            len(lines) == 1
            and json.loads(lines[0].removeprefix("RESULT_JSON:").strip()) == r
            and all(
                w in text.lower()
                for w in ("determinant", "unit", "permutation", "diagonal")
            )
        )
    except (OSError, UnicodeError, ValueError):
        return False


def main():
    e = json.loads((T / "expected.json").read_text())
    d = json.loads((T / "input.json").read_text())
    s = load_submission(W / "submission.json")
    c = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    r = s.get("result") if isinstance(s, dict) else None
    m = bool(workspace_input_is_bound() and valid(r, d))
    ev = bool(isinstance(s, dict) and evidence_ok(s.get("evidence"), r))
    sc = bool(
        isinstance(s, dict)
        and s.get("scope") == "frozen-six-dimensional-Z-mod-125-certificate"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == [LIMITATION]
    )
    a = bool(isinstance(s, dict) and s.get("claimed_assurance") == "COMPUTED")
    f = false_verified_claim(s, verification_record_bound=False)
    ok = bool(c and m and ev and sc and not f)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(c)),
                "correctness": float(m),
                "evidence_validity": float(ev),
                "scope_accuracy": float(sc),
                "assurance_calibration": float(a),
                "reward": 0 if not ok else 0.9 + 0.1 * a,
                "false_certification": f,
            }
        )
    )


if __name__ == "__main__":
    main()
