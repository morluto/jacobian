import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")

_OPERATIONS = {
    "compute the Fibonacci number at index n": "combinatorics.compute.fibonacci",
    "compute the Lucas number at index n": "combinatorics.compute.lucas",
    "evaluate requested indices of a constant coefficient linear recurrence": "combinatorics.recurrence.linear.evaluate",
    "exact coefficient prefix of a rational generating function at zero": "combinatorics.generating_function.coefficients.compute",
}


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict) or not isinstance(x, dict):
        return False
    query = x.get("query")
    if not isinstance(query, str):
        return False
    selected = _OPERATIONS.get(query)
    return selected is not None and result.get("selected_operation") == selected


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = _math(s, x) if protocol_ok else False
    reward = float(protocol_ok and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
