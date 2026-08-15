import json
import math
from itertools import permutations
from pathlib import Path

d = json.loads(Path("/app/input.json").read_text())
P = d["P"]
perm = [2, 4, 0, 5, 3, 1]


def sign(p):
    return -1 if sum(p[i] > p[j] for i in range(6) for j in range(i + 1, 6)) % 2 else 1


def mul(diag, left):
    return [
        [(P[i][j] * (diag[i] if left else diag[j])) % 125 for j in range(6)]
        for i in range(6)
    ]


entries = [P[i][perm[i]] for i in range(6)]
r = {
    "modulus": 125,
    "PA": mul(d["a_diagonal"], False),
    "BP": mul(d["b_diagonal"], True),
    "determinant_modulus": sum(
        sign(p) * math.prod(P[i][p[i]] for i in range(6))
        for p in permutations(range(6))
    )
    % 125,
    "unit_permutation": perm,
    "permutation_sign": sign(perm),
    "signed_term_modulus": sign(perm) * math.prod(entries) % 125,
    "matched_pairs": [
        {
            "row": i,
            "column": perm[i],
            "b_value": d["b_diagonal"][i],
            "a_value": d["a_diagonal"][perm[i]],
            "unit_entry": entries[i],
        }
        for i in range(6)
    ],
}
Path("/app/submission.json").write_text(json.dumps({"result": r}, indent=2) + "\n")
