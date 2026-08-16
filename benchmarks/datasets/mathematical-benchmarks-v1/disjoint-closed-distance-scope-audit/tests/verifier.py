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


def _epsilon_witnesses_ok(eps, s, c):
    if not isinstance(eps, list) or len(eps) != 8:
        return False
    for k, witness in enumerate(eps, 2):
        if not isinstance(witness, dict) or set(witness) != {
            "epsilon",
            "index",
            "distance_squared",
        }:
            return False
        e, n, d = (
            rat(witness["epsilon"]),
            witness["index"],
            rat(witness["distance_squared"]),
        )
        if (
            e != Fraction(1, k)
            or type(n) is not int
            or n < 1
            or (d != Fraction(s * s, (n + c) ** 2))
            or (not d < e * e)
        ):
            return False
    return True


def _separation_ok(sep, h):
    if not isinstance(sep, dict) or set(sep) != {
        "same_family_lower_bound_squared",
        "cross_family_vertical_nonzero",
        "closedness_reason",
    }:
        return False
    return (
        rat(sep["same_family_lower_bound_squared"]) == h * h
        and sep["cross_family_vertical_nonzero"] is True
        and (sep["closedness_reason"] == "DISTINCT_INDICES_HAVE_HORIZONTAL_GAP")
    )


def result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "horizontal_step",
        "vertical_scale",
        "offset",
        "sample_indices",
        "distance_squared",
        "epsilon_witnesses",
        "separation_certificate",
        "formal_conclusion",
        "corrected_conclusion",
    }:
        return False
    h, s, c = (result["horizontal_step"], result["vertical_scale"], result["offset"])
    if any(type(x) is not int for x in (h, s, c)) or not (
        2 <= h <= 20 and 1 <= s <= 20 and (2 <= c <= 20)
    ):
        return False
    ns = result["sample_indices"]
    if (
        ns != sorted(ns)
        or len(ns) != 10
        or len(set(ns)) != 10
        or any(type(n) is not int or n < 1 for n in ns)
    ):
        return False
    distances = result["distance_squared"]
    if not isinstance(distances, list) or len(distances) != len(ns):
        return False
    expected = [Fraction(s * s, (n + c) ** 2) for n in ns]
    if [rat(x) for x in distances] != expected:
        return False
    if not _epsilon_witnesses_ok(result["epsilon_witnesses"], s, c):
        return False
    if not _separation_ok(result["separation_certificate"], h):
        return False
    return (
        result["formal_conclusion"] == "POSITIVE_DISTANCE"
        and result["corrected_conclusion"] == "SEPARATED_BUT_DISTANCE_INFIMUM_ZERO"
    )


def frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        data = json.loads(raw)
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and (data["source_row"] == 18)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_ok = bool(result_ok(result) and frozen_ok())
    reward = float(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_ok), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
