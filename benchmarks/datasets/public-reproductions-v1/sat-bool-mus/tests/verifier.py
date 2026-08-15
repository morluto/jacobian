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


def _satisfied(clause, assign, variables):
    for lit in clause:
        idx = abs(lit) - 1
        if idx < 0 or idx >= len(variables):
            return False
        val = assign[variables[idx]]
        if not isinstance(val, bool):
            return False
        if (lit > 0) == val:
            return True
    return False


def _brute_unsat(clauses, variables):
    n = len(variables)
    for mask in range(1 << n):
        assign = {variables[i]: bool((mask >> i) & 1) for i in range(n)}
        if all(_satisfied(c, assign, variables) for c in clauses):
            return False
    return True


def _math(s, x, e):
    r = s.get("result", {})
    status = r.get("status")
    variables = x.get("variables", [])
    clauses = x.get("clauses", [])
    if e["expected_status"] == "SATISFIABLE":
        a = r.get("assignment")
        if not isinstance(a, dict) or set(a) != set(variables):
            return False
        if not all(isinstance(v, bool) for v in a.values()):
            return False
        return status == "SATISFIABLE" and all(
            _satisfied(c, a, variables) for c in clauses
        )
    return status == "UNSATISFIABLE" and _brute_unsat(clauses, variables)


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
