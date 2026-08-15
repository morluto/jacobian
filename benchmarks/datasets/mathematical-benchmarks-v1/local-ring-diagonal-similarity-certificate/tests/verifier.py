import json
import math
from itertools import permutations
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

T = Path("/tests")


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


def main():
    raw = load_submission_raw(require_input_binding=False)
    d = json.loads((T / "input.json").read_text())
    input_binding = workspace_input_is_bound()
    protocol_ok = submission_matches_public_schema(raw)
    r = raw.get("result") if isinstance(raw, dict) else None
    m = valid(r, d)
    correct = bool(input_binding and protocol_ok and m)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(m),
                "input_binding": float(input_binding),
                "protocol_compliance": float(protocol_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
