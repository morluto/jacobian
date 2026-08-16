import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _validate_groups(groups, x):
    members = []
    valid = isinstance(groups, list) and len(groups) == 3
    for g in groups:
        valid = (
            valid
            and isinstance(g, dict)
            and isinstance(g.get("name"), str)
            and isinstance(g.get("members"), list)
        )
        if valid:
            members.extend(g["members"])
    actual = {}
    if valid:
        try:
            for g in groups:
                if g["name"] in actual:
                    valid = False
                    break
                actual[g["name"]] = frozenset(g["members"])
        except TypeError:
            valid = False
    expected = {
        f"residue-{r}": frozenset(int(n) for n in x["universe"] if int(n) % 3 == r)
        for r in range(3)
    }
    return members, actual, expected


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    groups = s.get("result", {}).get("cases", []) if isinstance(s, dict) else []
    members, actual, expected = _validate_groups(groups, x)
    wanted = {int(member) for member in x["universe"]}
    math_correct = bool(
        workspace_input_is_bound()
        and all(type(member) is int for member in members)
        and len(members) == len(set(members))
        and set(members) == wanted
        and actual == expected
    )
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
