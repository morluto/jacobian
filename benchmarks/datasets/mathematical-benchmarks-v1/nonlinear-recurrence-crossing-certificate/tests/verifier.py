import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W, E = (Path("/app"), Path("/tests"))
NEG_INF, POS_INF = ("NEGATIVE_INFINITY", "POSITIVE_INFINITY")


def _frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (W / "input.json").is_symlink() or (W / "input.json").read_bytes() != raw:
            return {}
        return json.loads(raw)
    except (OSError, ValueError):
        return {}


def _is_int(value):
    """Reject booleans and floats; accept only exact Python integers."""
    return type(value) is int


def _rational(value):
    """Parse a reduced rational object, rejecting booleans and floats."""
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator, denominator = (value["numerator"], value["denominator"])
    if not _is_int(numerator) or not _is_int(denominator) or denominator < 1:
        return None
    if math.gcd(numerator, denominator) != 1:
        return None
    return Fraction(numerator, denominator)


def _extended_bound(value):
    """Parse an extended rational bound (rational or infinity sentinel)."""
    if value == NEG_INF:
        return NEG_INF
    if value == POS_INF:
        return POS_INF
    return _rational(value)


def _image(bound):
    """Compute f(bound) = bound - 1/bound for bound >= 0.

    f is strictly increasing on (0, inf), so the image of an open interval
    (lower, upper) is (f(lower), f(upper)).  f(0+) = -inf, so a zero lower
    bound maps to NEGATIVE_INFINITY.
    """
    if bound == 0:
        return NEG_INF
    return bound - Fraction(1) / bound


def _valid_chain(bounds, threshold):
    """Validate three bounds form a chain from (0, sqrt(threshold)) to negative.

    The chain has three roles:
    - entry:   (0, U) -> (-inf, V) with U^2 >= threshold, V > 1
    - terminal: (0, 1) -> (-inf, 0)
    - bridge:  (1, V) -> (0, W) with W < 1

    Starting from |a| < sqrt(threshold): if a < 0, done; if a > 0 then a in
    (0, U) so f(a) in (-inf, V).  If f(a) < 0, done (1 step).  If f(a) in
    (0, 1), terminal gives f(f(a)) < 0 (2 steps).  If f(a) in (1, V), bridge
    gives f(f(a)) in (0, W) subset (0, 1), then terminal gives f(f(f(a))) < 0
    (3 steps).  The chain therefore takes at most 3 steps.
    """
    terminal = None
    bridge = None
    entry = None
    for bound in bounds:
        lower, upper, out_lower, out_upper = bound
        if lower == 0 and upper == 1 and (out_lower == NEG_INF) and (out_upper == 0):
            terminal = bound
        elif lower == 1 and out_lower == 0:
            bridge = bound
        elif lower == 0 and out_lower == NEG_INF and (upper != 1):
            entry = bound
    if terminal is None or bridge is None or entry is None:
        return False
    entry_upper = entry[1]
    if not isinstance(entry_upper, Fraction) or entry_upper * entry_upper < threshold:
        return False
    entry_output_upper = entry[3]
    if (
        not isinstance(entry_output_upper, Fraction)
        or entry_output_upper <= 1
        or bridge[1] != entry_output_upper
    ):
        return False
    bridge_output_upper = bridge[3]
    return isinstance(bridge_output_upper, Fraction) and bridge_output_upper < 1


def _potential_coefficients_valid(result):
    coefficients = result["potential_identity_coefficients"]
    return not (
        not isinstance(coefficients, list)
        or len(coefficients) != 3
        or (not all(_is_int(c) for c in coefficients))
        or (coefficients != [1, -2, 1])
    )


def _initial_potential_valid(result, frozen):
    initial_value = frozen.get("initial_value")
    initial_index = frozen.get("initial_index")
    if not _is_int(initial_value) or not _is_int(initial_index):
        return None
    initial_potential = result["initial_potential"]
    if not _is_int(initial_potential) or initial_potential != initial_value**2:
        return None
    return (initial_potential, initial_index)


def _threshold_and_decrement_valid(result):
    threshold = _rational(result["threshold"])
    if threshold is None or threshold <= 0:
        return None
    expected_decrement = 2 - Fraction(1) / threshold
    decrement = _rational(result["decrement_lower_bound"])
    if decrement is None or decrement != expected_decrement:
        return None
    return (threshold, expected_decrement)


def _phase_transitions_valid(
    result, initial_potential, initial_index, expected_decrement, threshold
):
    transitions = result["phase_transitions"]
    if not _is_int(transitions) or transitions < 1:
        return False
    potential = Fraction(initial_potential)
    if not (
        potential - transitions * expected_decrement < threshold
        and potential - (transitions - 1) * expected_decrement >= threshold
    ):
        return False
    return not (
        not _is_int(result["threshold_index_upper"])
        or result["threshold_index_upper"] != initial_index + transitions
    )


def _terminal_bounds_valid(result):
    bounds = result["terminal_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 3:
        return None
    parsed_bounds = []
    for item in bounds:
        if not isinstance(item, dict) or set(item) != {
            "input_lower",
            "input_upper",
            "output_lower",
            "output_upper",
        }:
            return None
        input_lower = _extended_bound(item["input_lower"])
        input_upper = _extended_bound(item["input_upper"])
        output_lower = _extended_bound(item["output_lower"])
        output_upper = _extended_bound(item["output_upper"])
        if any(
            b is None for b in (input_lower, input_upper, output_lower, output_upper)
        ):
            return None
        if not isinstance(input_lower, Fraction) or input_lower < 0:
            return None
        if not isinstance(input_upper, Fraction) or input_upper <= 0:
            return None
        if input_lower >= input_upper:
            return None
        if output_lower != _image(input_lower) or output_upper != _image(input_upper):
            return None
        parsed_bounds.append((input_lower, input_upper, output_lower, output_upper))
    return parsed_bounds


def _result_valid(result, frozen):
    required = {
        "potential_identity_coefficients",
        "initial_potential",
        "threshold",
        "decrement_lower_bound",
        "phase_transitions",
        "threshold_index_upper",
        "terminal_bounds",
        "negative_index_upper",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    if not _potential_coefficients_valid(result):
        return False
    initial = _initial_potential_valid(result, frozen)
    if initial is None:
        return False
    initial_potential, initial_index = initial
    td = _threshold_and_decrement_valid(result)
    if td is None:
        return False
    threshold, expected_decrement = td
    if not _phase_transitions_valid(
        result, initial_potential, initial_index, expected_decrement, threshold
    ):
        return False
    parsed_bounds = _terminal_bounds_valid(result)
    if parsed_bounds is None:
        return False
    if not _valid_chain(parsed_bounds, threshold):
        return False
    if (
        not _is_int(result["negative_index_upper"])
        or result["negative_index_upper"] != result["threshold_index_upper"] + 3
    ):
        return False
    target = frozen.get("target", {})
    return bool(
        isinstance(target, dict)
        and _is_int(target.get("index_upper_exclusive"))
        and (result["negative_index_upper"] < target["index_upper_exclusive"])
    )


def main():
    submission, frozen = (load_submission(), _frozen())
    math_correct = bool(submission and _result_valid(submission.get("result"), frozen))
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
