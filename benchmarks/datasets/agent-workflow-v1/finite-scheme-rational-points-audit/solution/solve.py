import hashlib
import itertools
import json
from pathlib import Path

P = 5


def tensor(dimension, products):
    value = [[[0] * dimension for _ in range(dimension)] for _ in range(dimension)]
    for (left, right), output in products.items():
        value[left][right][output] = value[right][left][output] = 1
    return value


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True)) % P


def maps(table, unit):
    return [
        list(v)
        for v in itertools.product(range(P), repeat=len(unit))
        if dot(v, unit) == 1
        and all(
            dot(v, table[i][j]) == v[i] * v[j] % P
            for i in range(len(unit))
            for j in range(len(unit))
        )
    ]


a_table = tensor(5, {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 3): 3, (0, 4): 4, (3, 3): 4})
b_table = tensor(3, {(0, 0): 0, (1, 1): 1, (2, 2): 2})
a_points, b_points = maps(a_table, [1, 1, 1, 0, 0]), maps(b_table, [1, 1, 1])
columns = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
result = {
    "field_prime": 5,
    "a_unit": [1, 1, 1, 0, 0],
    "b_unit": [1, 1, 1],
    "a_multiplication": a_table,
    "b_multiplication": b_table,
    "morphism_columns": columns,
    "a_points": a_points,
    "b_points": b_points,
    "induced_point_map": [
        b_points.index([dot(point, column) for column in columns]) for point in a_points
    ],
    "nilpotent": {
        "vector": [0, 0, 0, 1, 0],
        "power2": [0, 0, 0, 0, 1],
        "power3": [0, 0, 0, 0, 0],
        "exact_order": 3,
    },
    "b_reduced": True,
}
text = (
    "Both affine schemes are nonempty and have the same three rational points under the induced map. A has a nonzero order-three nilpotent while B is reduced, so they are not isomorphic.\nRESULT_JSON:"
    + json.dumps(result, sort_keys=True, separators=(",", ":"))
    + "\n"
)
Path("/app/evidence").mkdir(parents=True, exist_ok=True)
Path("/app/evidence/answer.txt").write_text(text)
digest = hashlib.sha256(text.encode()).hexdigest()
submission = {
    "task_id": "jacobian/finite-scheme-rational-points-audit",
    "conclusion": "RATIONAL_POINT_BIJECTION_DOES_NOT_FORCE_SCHEME_ISOMORPHISM",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "finite-affine-schemes-over-F5-certificate",
    "completeness": "COMPLETE",
    "evidence": [{"path": "evidence/answer.txt", "sha256": f"sha256:{digest}"}],
    "limitations": [
        "The verifier certifies one finite affine countermodel over F_5, not a general theorem about schemes or functors of points."
    ],
}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
