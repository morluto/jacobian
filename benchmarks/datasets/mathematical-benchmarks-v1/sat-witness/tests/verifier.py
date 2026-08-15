import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _answer_evidence(submission):
    return witness_list_is_bound(submission.get("witness"))


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())

    result = s.get("result", {}) if isinstance(s, dict) else {}
    a = result.get("assignment", {}) if isinstance(result, dict) else {}
    valid = (
        isinstance(result, dict)
        and set(result) == {"status", "assignment"}
        and isinstance(a, dict)
        and set(a) == set(x["variables"])
        and all(isinstance(v, bool) for v in a.values())
    )
    sat = valid and all(
        any((lit > 0) == a[x["variables"][abs(lit) - 1]] for lit in c)
        for c in x["clauses"]
    )
    math_correct = bool(
        workspace_input_is_bound() and result.get("status") == "SATISFIABLE" and sat
    )
    good = _answer_evidence(s) if isinstance(s, dict) else False
    reward = float(math_correct and good)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(good),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
