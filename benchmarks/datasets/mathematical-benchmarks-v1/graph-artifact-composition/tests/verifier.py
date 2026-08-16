import json
from collections import deque
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def sub():
    return load_submission()


def ev(s):
    return bool(s)


def main():
    s = sub()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    v = {str(a): set() for a in x["vertices"]}
    for a, b in x["edges"]:
        v[a].add(b)
        v[b].add(a)
    d = max(map(len, v.values()))
    m = sorted(k for k in v if len(v[k]) == d)
    ds = dict.fromkeys(m, 0)
    q = deque(m)
    while q:
        a = q.popleft()
        for b in v[a]:
            if b not in ds:
                ds[b] = ds[a] + 1
                q.append(b)
    order = sorted(v)
    md = max(ds.values())
    result = {
        "maximum_degree_vertices": m,
        "distance_to_set": [{"vertex": a, "distance": ds[a]} for a in order],
        "maximum_distance_to_set": md,
        "maximizing_vertices": [a for a in order if ds[a] == md],
    }
    distances_are_integers = (
        isinstance(s, dict)
        and isinstance(s.get("result"), dict)
        and all(
            isinstance(item, dict) and type(item.get("distance")) is int
            for item in s["result"].get("distance_to_set", [])
        )
        and (type(s["result"].get("maximum_distance_to_set")) is int)
    )
    math_correct = bool(
        isinstance(s, dict) and distances_are_integers and (s.get("result") == result)
    )
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
