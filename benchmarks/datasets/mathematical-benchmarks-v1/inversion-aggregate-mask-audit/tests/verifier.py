import itertools
import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W, T = Path("/app"), Path("/tests")


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def counts(p):
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    intended = sum(p[i] > p[j] for i, j in pairs)
    implemented = sum(p[i] <= p[j] for i, j in pairs)
    return implemented, intended


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "witness_permutation",
        "implemented_count",
        "intended_count",
        "implemented_aggregate",
        "intended_aggregate",
        "pair_count",
        "complement_relation",
    }:
        return False
    p = r["witness_permutation"]
    if (
        not isinstance(p, list)
        or len(p) != 4
        or not all(type(value) is int for value in p)
        or sorted(p) != list(range(4))
    ):
        return False
    if not all(
        type(r[key]) is int
        for key in (
            "implemented_count",
            "intended_count",
            "implemented_aggregate",
            "intended_aggregate",
            "pair_count",
        )
    ):
        return False
    if (
        r["complement_relation"]
        != "IMPLEMENTED_PLUS_INTENDED_EQUALS_PAIR_COUNT_PER_PERMUTATION"
    ):
        return False
    ic, tc = counts(p)
    perms = list(itertools.permutations(range(4)))
    ia = sum(counts(q)[0] for q in perms)
    ta = sum(counts(q)[1] for q in perms)
    return (
        ic != tc
        and r["implemented_count"] == ic
        and r["intended_count"] == tc
        and r["implemented_aggregate"] == ia
        and r["intended_aggregate"] == ta
        and r["pair_count"] == 6
        and ia == ta == 72
        and all(sum(counts(q)) == 6 for q in perms)
    )


def main():
    s = load_submission(W / "submission.json")
    math_ok = bool(s is not None and frozen() and valid(s.get("result")))
    correct = math_ok
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
