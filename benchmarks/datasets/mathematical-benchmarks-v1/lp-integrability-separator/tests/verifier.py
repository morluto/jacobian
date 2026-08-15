import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W, T = Path("/app"), Path("/tests")
MAX_EVIDENCE_BYTES = 64 * 1024


def fraction(value):
    """Parse a rational from a string, accepting mathematically equivalent forms."""
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    return Fraction(value)


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "beta",
        "origin_power_coefficient",
        "infinity_power_coefficient",
        "p2_log_exponent",
        "p2_integral_each",
        "critical_p",
        "lower_regime",
        "upper_regime",
    }:
        return False
    try:
        beta = fraction(result["beta"])
        log_exponent = fraction(result["p2_log_exponent"])
        integral = fraction(result["p2_integral_each"])
    except (ValueError, ZeroDivisionError):
        return False
    return bool(
        beta > Fraction(1, 2)
        and result["origin_power_coefficient"] == "-1/2"
        and result["infinity_power_coefficient"] == "-1/2"
        and log_exponent == -2 * beta
        and integral == 1 / (2 * beta - 1)
        and result["critical_p"] == "2"
        and result["lower_regime"]
        == {"p_interval": "0<p<2", "obstruction": "INFINITY_POWER_TAIL"}
        and result["upper_regime"]
        == {"p_interval": "p>2", "obstruction": "ORIGIN_POWER_SINGULARITY"}
    )


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def _json_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def main():
    submission = load_submission(W / "submission.json")
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_ok = bool(protocol_ok and frozen() and valid_result(result))
    reward = float(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
