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


def _term(t):
    if not isinstance(t, dict):
        return None
    c = t.get("coefficient")
    if not isinstance(c, dict):
        return None
    return (c.get("num"), c.get("den"), tuple(t.get("exponents", [])))


def _poly(terms):
    if not isinstance(terms, list):
        return None
    return sorted(_term(t) for t in terms)


def _math(s, x, e):
    r = s.get("result", {})
    if r.get("keller_condition_verified") is not True:
        return False
    det = r.get("determinant")
    if not isinstance(det, dict):
        return False
    return _poly(det.get("terms")) == _poly(e["expected_determinant"]["terms"])


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
