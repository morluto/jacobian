import json
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W, T = Path("/app"), Path("/tests")


def fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError
    parsed = Fraction(numerator, denominator)
    if parsed.numerator != numerator or parsed.denominator != denominator:
        raise ValueError
    return parsed


def encoded(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def expected_spike(n, alpha):
    center = Fraction(2 * n + 1, 2)
    width = alpha / n
    return {
        "n": n,
        "center": encoded(center),
        "half_width": encoded(width),
        "left": encoded(center - width),
        "right": encoded(center + width),
        "area": encoded(width),
        "integer_sample": encoded(Fraction(1, n * n)),
    }


def _valid_spikes(spikes, alpha):
    if not isinstance(spikes, list) or len(spikes) != 12:
        return False
    if any(
        not isinstance(spike, dict) or type(spike.get("n")) is not int
        for spike in spikes
    ):
        return False
    expected = {n: expected_spike(n, alpha) for n in range(1, 13)}
    by_n = {}
    for spike in spikes:
        n = spike["n"]
        if n in by_n or spike != expected.get(n):
            return False
        by_n[n] = spike
    if set(by_n) != set(expected):
        return False
    ordered = [by_n[n] for n in sorted(by_n)]
    for left, right in pairwise(ordered):
        if fraction(left["right"]) >= fraction(right["left"]):
            return False
    for spike in spikes:
        left, right = fraction(spike["left"]), fraction(spike["right"])
        n = spike["n"]
        if left <= n or right >= n + 1:
            return False
    return True


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "alpha",
        "baseline_power",
        "spike_height",
        "spikes",
        "integral_classification",
        "sample_series_classification",
    }:
        return False
    try:
        alpha = fraction(result["alpha"])
    except (ValueError, ZeroDivisionError):
        return False
    if not (0 < alpha <= Fraction(1, 4)):
        return False
    if result["baseline_power"] != 2 or result["spike_height"] != "1":
        return False
    if not _valid_spikes(result.get("spikes"), alpha):
        return False
    return bool(
        result["integral_classification"]
        == {"spike_area_series": "alpha*sum(1/n)", "status": "DIVERGENT"}
        and result["sample_series_classification"]
        == {"sample_series": "sum(1/n^2)", "status": "CONVERGENT"}
    )


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def main():
    submission = load_submission(W / "submission.json")
    data = submission if isinstance(submission, dict) else {}
    input_bound = frozen()
    math_ok = bool(
        isinstance(submission, dict)
        and input_bound
        and valid_result(data.get("result"))
    )
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
