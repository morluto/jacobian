import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    witness_list_is_bound,
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


def valid_evidence(evidence, result):
    """Bind evidence content to the submitted result via a RESULT_JSON line."""
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    if not witness_list_is_bound(
        evidence, expected_path="evidence/answer.txt", max_bytes=MAX_EVIDENCE_BYTES
    ):
        return False
    target = resolve_evidence(
        evidence[0], expected_path="evidence/answer.txt", max_bytes=MAX_EVIDENCE_BYTES
    )
    if target is None:
        return False
    try:
        text = target.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line.removeprefix("RESULT_JSON:").strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound_result = json.loads(markers[0])
    except (ValueError, RecursionError):
        return False
    if not isinstance(result, dict):
        return False
    return _json_equal(bound_result, result)


def main():
    submission = load_submission(W / "submission.json")
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_ok = bool(protocol_ok and frozen() and valid_result(result))
    evidence_ok = bool(
        protocol_ok and frozen() and valid_evidence(submission.get("witness"), result)
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=evidence_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(evidence_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
