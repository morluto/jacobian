import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    sha256_uri,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _digest(path):
    return sha256_uri(path)


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


def _check_evidence_items(evidence):
    good = True
    for i in evidence:
        if (
            not isinstance(i, dict)
            or not isinstance(i.get("path"), str)
            or not isinstance(i.get("sha256"), str)
        ):
            good = False
            continue
        p = Path(i["path"])
        t = (W / p).resolve()
        if (
            p.is_absolute()
            or p != Path("evidence/answer.txt")
            or ".." in p.parts
            or (W / p).is_symlink()
            or not t.is_relative_to(W.resolve())
            or not t.is_file()
        ):
            good = False
            continue
        good &= i.get("sha256") == _digest(t)
    return good


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
    good_evidence = bool(
        isinstance(s, dict)
        and isinstance(s.get("witness"), list)
        and s["witness"]
        and _check_evidence_items(s["witness"])
    )
    reward = float(math_correct and good_evidence)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(good_evidence),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
