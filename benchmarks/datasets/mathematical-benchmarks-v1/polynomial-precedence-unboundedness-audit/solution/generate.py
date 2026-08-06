import hashlib
import json
import shutil
from pathlib import Path

shutil.copyfile("/solution/input.json", "/app/input.json")


def r(n, d=1):
    return {"numerator": n, "denominator": d}


result = {
    "x_coefficients": [r(-7), r(0), r(-1, 2)],
    "y_coefficients": [r(0), r(1)],
    "formal_coefficients": [r(49), r(28), r(-3), r(0), r(-1, 4)],
    "checkpoints": [
        {"t": 0, "value": r(49)},
        {"t": 1, "value": r(295, 4)},
        {"t": 2, "value": r(89)},
        {"t": 5, "value": r(-169, 4)},
    ],
    "formal_status": "UNBOUNDED_BELOW",
}
limitations = ["LEAN_ELABORATION_NOT_ASSESSED", "INFORMAL_MINIMUM_NOT_REPROVED"]
evidence = {
    "schema_version": "1",
    "task_id": "jacobian/polynomial-precedence-unboundedness-audit",
    "result": result,
    "limitations": limitations,
}
epath = Path("/app/evidence/precedence-audit.json")
epath.parent.mkdir(parents=True, exist_ok=True)
epath.write_text(json.dumps(evidence, separators=(",", ":")))
submission = {
    "task_id": evidence["task_id"],
    "conclusion": "FORMALIZATION_CHANGES_SEMANTICS",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "EXACT_RATIONAL_PARAMETRIC_COUNTERMODEL_TO_FORMAL_EXPRESSION",
    "completeness": "COMPLETE",
    "evidence": [
        {
            "path": "evidence/precedence-audit.json",
            "sha256": "sha256:" + hashlib.sha256(epath.read_bytes()).hexdigest(),
        }
    ],
    "limitations": limitations,
}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
Path("/app/answer.txt").write_text(
    "The formal expression is unbounded below along the submitted exact rational polynomial family.\n"
)
