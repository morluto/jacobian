import json
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
)

W = Path("/app")
E = Path("/tests")


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer_value(value):
    return value if type(value) is int else None


def witness_matches_result(witness, result):
    if not isinstance(witness, list) or len(witness) != 1:
        return False
    target = resolve_evidence(witness[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json_value_equal(json.loads(marker), result) and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def _valid_witness(result, source):
    if not isinstance(result, dict) or set(result) != {
        "a",
        "b",
        "product_mod_7",
        "power_difference",
        "quotient_by_7_pow_7",
    }:
        return False
    normalized = {key: _integer_value(result[key]) for key in result}
    if any(value is None for value in normalized.values()):
        return False

    a = normalized["a"]
    b = normalized["b"]
    try:
        minimum = source["search_scope"]["minimum"]
        maximum = source["search_scope"]["maximum"]
    except (KeyError, TypeError):
        return False
    if not (minimum <= a <= maximum and minimum <= b <= maximum):
        return False

    product = a * b * (a + b)
    difference = (a + b) ** 7 - a**7 - b**7
    divisor = 7**7
    return bool(
        product % 7 != 0
        and difference % divisor == 0
        and normalized["product_mod_7"] == product % 7
        and normalized["power_difference"] == difference
        and normalized["quotient_by_7_pow_7"] == difference // divisor
    )


def main():
    submission = load_submission()
    source = _load_frozen_input()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = _valid_witness(result, source)
    witness_valid = bool(
        isinstance(submission, dict)
        and witness_matches_result(submission.get("witness"), result)
    )
    reward = float(math_correct and witness_valid)

    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(witness_valid),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
