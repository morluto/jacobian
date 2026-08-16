import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W, E = (Path("/app"), Path("/tests"))


def rat(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    if (
        type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    return Fraction(value["numerator"], value["denominator"])


def add(a, b):
    out = [Fraction(0)] * max(len(a), len(b))
    for i, v in enumerate(a):
        out[i] += v
    for i, v in enumerate(b):
        out[i] += v
    return trim(out)


def mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] += u * v
    return trim(out)


def trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def formal_poly(x, y):
    return add(
        add(mul(x, mul(y, y)), mul(add(x, [Fraction(7)]), add(x, [Fraction(7)]))),
        mul(
            add([2 * v for v in y], [Fraction(7)]),
            add([2 * v for v in y], [Fraction(7)]),
        ),
    )


def evaluate(poly, t):
    total = Fraction(0)
    for coefficient in reversed(poly):
        total = total * t + coefficient
    return total


def result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "x_coefficients",
        "y_coefficients",
        "formal_coefficients",
        "checkpoints",
        "formal_status",
    }:
        return False
    x = [rat(v) for v in result["x_coefficients"]]
    y = [rat(v) for v in result["y_coefficients"]]
    claimed = [rat(v) for v in result["formal_coefficients"]]
    if len(x) != 3 or len(y) != 2 or None in x + y + claimed:
        return False
    actual = formal_poly(x, y)
    if (
        claimed != actual
        or len(actual) < 3
        or actual[-1] >= 0
        or (result["formal_status"] != "UNBOUNDED_BELOW")
    ):
        return False
    checks = result["checkpoints"]
    if not isinstance(checks, list) or len(checks) != 4:
        return False
    ts = []
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != {"t", "value"}
            or type(check["t"]) is not int
        ):
            return False
        value = rat(check["value"])
        if value is None or value != evaluate(actual, check["t"]):
            return False
        ts.append(check["t"])
    return len(set(ts)) == 4


def frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        data = json.loads(raw)
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and (data["formal_expression"] != data["informal_expression"])
            and (data["claimed_least"] == 45)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_ok = bool(protocol_ok and result_ok(result) and frozen_ok())
    reward = float(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_ok), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
