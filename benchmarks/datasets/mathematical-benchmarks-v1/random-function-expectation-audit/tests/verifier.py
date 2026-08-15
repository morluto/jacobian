import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def q(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,40}|/[1-9][0-9]*)?", value)
        is None
    ):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return parsed


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    input_binding = workspace_input_is_bound()
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}

    n = x["domain_size"]
    self_hit = Fraction(2 * n - 1, n * n)
    other_hit = Fraction(n - 1, n * n)
    squared_sum = sum(
        (target - source) ** 2
        for source in range(1, n + 1)
        for target in range(1, n + 1)
    )
    expectation = other_hit * squared_sum
    math_ok = bool(
        isinstance(s, dict)
        and set(r)
        == {
            "self_hit_probability",
            "other_hit_probability",
            "ordered_squared_difference_sum",
            "expected_value",
        }
        and type(r.get("ordered_squared_difference_sum")) is int
        and q(r.get("self_hit_probability")) == self_hit
        and q(r.get("other_hit_probability")) == other_hit
        and r.get("ordered_squared_difference_sum") == squared_sum
        and q(r.get("expected_value")) == expectation
        and expectation != 2025
    )
    ev_ok = bool(isinstance(s, dict) and evidence_matches_result(s.get("witness"), r))
    reward = aggregate_reward(
        correctness=math_ok,
        evidence_validity=ev_ok,
        protocol_ok=bool(input_binding and s is not None),
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(ev_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
