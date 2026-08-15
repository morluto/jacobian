import hashlib
import json
import os
import shutil
from fractions import Fraction
from pathlib import Path

vectors = [
    [1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, -1, -1, 1, 1],
    [4, 0, 0, 0, 4],
    [2, -2, 4, 0, 2],
]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


out = []
for v in vectors:
    w = list(map(Fraction, v))
    for u in out:
        d = dot(u, u)
        if d:
            q = dot(v, u) / d
            w = [a - q * b for a, b in zip(w, u, strict=True)]
    out.append(w)


def r(x):
    return {"numerator": x.numerator, "denominator": x.denominator}


result = {
    "vectors": vectors,
    "residuals": [[r(x) for x in v] for v in out],
    "rank": 4,
    "zero_residual_indices": [4, 5],
    "formal_selected_indices": [0, 1, 2, 3, 4, 5],
    "intended_selected_indices": [0, 1, 2, 3],
}
limitations = [
    "LEAN_ELABORATION_NOT_ASSESSED",
    "NORMALIZATION_OF_NONZERO_RESIDUALS_NOT_REQUIRED",
]
evidence = {
    "schema_version": "1",
    "task_id": "jacobian/gram-schmidt-nonzero-filter-audit",
    "result": result,
    "limitations": limitations,
}
root = Path(os.environ.get("SOLUTION_ROOT", "/app"))
if root == Path("/app"):
    shutil.copyfile("/solution/input.json", root / "input.json")
p = root / "evidence/gram-schmidt-audit.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(evidence, separators=(",", ":")))
submission = {
    "result": result,
    "witness": [
        {
            "path": "evidence/gram-schmidt-audit.json",
            "sha256": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    ],
}
(root / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
(root / "answer.txt").write_text(
    "The nonnegative-norm filter retains the two exact zero residuals.\n"
)
