import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")

_FROZEN = {
    (
        "CORE",
        "(P Q : Prop) → P → Q → P ∧ Q",
        ("intro P Q hP hQ",),
        "constructor",
    ): {"goal_count": 2, "completed": False}
}


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict) or not isinstance(x, dict):
        return False
    if (
        set(result) != {"goal_count", "completed"}
        or type(result.get("goal_count")) is not int
        or type(result.get("completed")) is not bool
    ):
        return False
    prefix = x.get("proof_prefix")
    if not isinstance(prefix, list) or any(
        not isinstance(item, str) for item in prefix
    ):
        return False
    expected = _FROZEN.get(
        (x.get("environment"), x.get("statement"), tuple(prefix), x.get("tactic"))
    )
    if expected is None:
        return False
    return (
        result["goal_count"] == expected["goal_count"]
        and result["completed"] is expected["completed"]
    )


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
