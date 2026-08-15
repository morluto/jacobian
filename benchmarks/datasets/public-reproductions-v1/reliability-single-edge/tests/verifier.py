import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _frac(v):
    if not isinstance(v, dict):
        return None
    num, den = v.get("num"), v.get("den")
    if not isinstance(num, str) or not isinstance(den, str):
        return None
    return (num, den)


def _math(s, x, e):
    r = s.get("result", {})
    states = r.get("states")
    if type(states) is not int:
        return False
    return (
        _frac(r.get("probability")) == _frac(e["expected_probability"])
        and states == e["expected_states"]
    )


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    math_correct = _math(s, x, e) if protocol_ok else False
    correct = bool(protocol_ok and math_correct)
    good = bool(protocol_ok and witness_list_is_bound(s["witness"]))
    reward = aggregate_reward(
        correctness=correct, witness_validity=good, protocol_ok=protocol_ok
    )
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
