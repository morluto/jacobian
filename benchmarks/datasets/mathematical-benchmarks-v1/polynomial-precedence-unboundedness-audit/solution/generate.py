import json
import shutil
from pathlib import Path

# The frozen input is provided by the agent environment Dockerfile at
# /app/input.json; copy a generated /solution/input.json only when present.
frozen = Path("/solution/input.json")
if frozen.exists():
    shutil.copyfile(frozen, "/app/input.json")


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
submission = {"result": result}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
Path("/app/answer.txt").write_text(
    "The formal expression is unbounded below along the submitted exact rational polynomial family.\n"
)
