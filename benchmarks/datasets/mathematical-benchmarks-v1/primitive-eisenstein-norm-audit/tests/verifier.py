import json
import math
from pathlib import Path

from verifier_support import (
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
T = Path("/tests")
MAX_INPUT_BYTES = 1048576


def frozen_contract() -> dict:
    try:
        app = W / "input.json"
        test = T / "input.json"
        if app.is_symlink() or test.is_symlink():
            return {}
        if not is_regular_bounded_file(app, max_bytes=MAX_INPUT_BYTES):
            return {}
        raw = test.read_bytes()
        if app.read_bytes() != raw or len(raw) > MAX_INPUT_BYTES:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return (
        value
        if isinstance(value, dict)
        and value.get("task_id") == "jacobian/primitive-eisenstein-norm-audit"
        else {}
    )


def valuation(value: int, prime: int) -> int:
    exponent = 0
    while value % prime == 0:
        exponent += 1
        value //= prime
    return exponent


def _is_int(value: object) -> bool:
    return type(value) is int


def certificate_valid(result: object, frozen: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "ramified_witness",
        "inert_obstruction",
        "repaired_criterion",
    }:
        return False
    ramified = result["ramified_witness"]
    inert = result["inert_obstruction"]
    if not isinstance(ramified, dict) or set(ramified) != {
        "x",
        "y",
        "norm",
        "gcd",
        "v3",
    }:
        return False
    if not isinstance(inert, dict) or set(inert) != {
        "prime",
        "zero_pairs",
        "square_primitive_status",
    }:
        return False
    x, y = (ramified.get("x"), ramified.get("y"))
    if not _is_int(x) or not _is_int(y) or (not (-30 <= x <= 30 and -30 <= y <= 30)):
        return False
    norm = x * x + x * y + y * y
    if (
        x == 0
        or y == 0
        or math.gcd(x, y) != 1
        or (norm <= 0)
        or (not _is_int(ramified.get("norm")))
        or (ramified.get("norm") != norm)
        or (not _is_int(ramified.get("gcd")))
        or (ramified.get("gcd") != 1)
        or (not _is_int(ramified.get("v3")))
        or (ramified.get("v3") != valuation(norm, 3))
        or (valuation(norm, 3) != 1)
    ):
        return False
    prime = inert.get("prime")
    if not _is_int(prime) or prime not in frozen.get("allowed_inert_primes", []):
        return False
    zero_pairs = inert.get("zero_pairs")
    if not isinstance(zero_pairs, list):
        return False
    expected_pairs = [
        {"x": a, "y": b}
        for a in range(prime)
        for b in range(prime)
        if (a * a + a * b + b * b) % prime == 0
    ]
    normalized: list[dict[str, int]] = []
    for pair in zero_pairs:
        if (
            not isinstance(pair, dict)
            or set(pair) != {"x", "y"}
            or (not _is_int(pair.get("x")))
            or (not _is_int(pair.get("y")))
        ):
            return False
        normalized.append({"x": pair["x"] % prime, "y": pair["y"] % prime})
    return bool(
        sorted(normalized, key=lambda p: (p["x"], p["y"]))
        == sorted(expected_pairs, key=lambda p: (p["x"], p["y"]))
        and inert.get("square_primitive_status") == "IMPOSSIBLE"
        and (
            result.get("repaired_criterion")
            == {
                "three_exponent_maximum": 1,
                "forbidden_prime_residue": {"modulus": 3, "residue": 2},
            }
        )
    )


def main() -> None:
    frozen = frozen_contract()
    submission = load_submission(W / "submission.json")
    contract = bool(submission)
    math_correct = bool(
        frozen
        and submission is not None
        and certificate_valid(submission.get("result"), frozen)
    )
    correct = bool(math_correct and contract)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": float(correct)})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
