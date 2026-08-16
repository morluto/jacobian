import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _satisfied(clause, assign, variables):
    for literal in clause:
        idx = abs(literal) - 1
        if idx < 0 or idx >= len(variables):
            return False
        val = assign[variables[idx]]
        if not isinstance(val, bool):
            return False
        if (literal > 0) == val:
            return True
    return False


def _brute_unsat(clauses, variables):
    n = len(variables)
    for mask in range(1 << n):
        assign = {variables[i]: bool(mask >> i & 1) for i in range(n)}
        if all(_satisfied(c, assign, variables) for c in clauses):
            return False
    return True


def _math(s, x):
    r = s.get("result", {})
    status = r.get("status")
    variables = x.get("variables", [])
    clauses = x.get("clauses", [])
    if status == "SATISFIABLE":
        a = r.get("assignment")
        if not isinstance(a, dict) or set(a) != set(variables):
            return False
        if not all(isinstance(v, bool) for v in a.values()):
            return False
        return all(_satisfied(c, a, variables) for c in clauses)
    if status == "UNSATISFIABLE":
        if "assignment" in r:
            return False
        return _brute_unsat(clauses, variables)
    return False


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
