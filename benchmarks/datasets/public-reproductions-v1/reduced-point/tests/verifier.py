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


def _math(s, x, e):
    r = s.get("result", {})
    fr = r.get("free_ranks")
    tor = r.get("torsion")
    if not isinstance(fr, list) or not isinstance(tor, list):
        return False
    if not all(type(v) is int for v in fr):
        return False
    if fr != e["expected_free_ranks"]:
        return False
    return [[str(v) for v in row] for row in tor] == [
        [str(v) for v in row] for row in e["expected_torsion"]
    ]


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    protocol_ok = s is not None
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
