import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

FROZEN_INPUT = Path(__file__).with_name("input.json")
REQUIRED_RESULT_FIELDS = frozenset({"sequence", "flow", "primal_value", "dual_value"})
MAX_FRACTION_BITS = 1024


def load_instance(path=FROZEN_INPUT):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "task_id",
        "cycle_size",
        "marked_indices",
        "sum_constraint",
        "edge_constraint",
        "objective",
    }:
        return None
    cycle_size = value["cycle_size"]
    marked_indices = value["marked_indices"]
    if (
        type(cycle_size) is not int
        or cycle_size <= 0
        or (not isinstance(marked_indices, list))
        or (not marked_indices)
        or any(type(index) is not int for index in marked_indices)
        or (len(set(marked_indices)) != len(marked_indices))
        or any(index < 1 or index > cycle_size for index in marked_indices)
    ):
        return None
    return {
        "cycle_size": cycle_size,
        "marked_indices": tuple(index - 1 for index in marked_indices),
    }


def fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator < 1
        or (numerator.bit_length() > MAX_FRACTION_BITS)
        or (denominator.bit_length() > MAX_FRACTION_BITS)
    ):
        return None
    result = Fraction(numerator, denominator)
    return result


def minimum_cost(instance=None):
    instance = instance or load_instance()
    if instance is None:
        return None
    cycle_size = instance["cycle_size"]
    marks = set(instance["marked_indices"])
    marked_fraction = Fraction(len(marks), cycle_size)
    weights = [
        1 - marked_fraction if i in marks else -marked_fraction
        for i in range(cycle_size)
    ]
    cumulative = []
    total = Fraction()
    for value in weights:
        total += value
        cumulative.append(total)
    median = sorted(cumulative)[len(cumulative) // 2]
    return sum(abs(value - median) for value in cumulative)


def valid(result, instance=None):
    instance = instance or load_instance()
    if instance is None or not isinstance(result, dict):
        return False
    try:
        if not set(result) >= REQUIRED_RESULT_FIELDS:
            return False
        cycle_size = instance["cycle_size"]
        marks = set(instance["marked_indices"])
        if (
            not isinstance(result["sequence"], list)
            or not isinstance(result["flow"], list)
            or len(result["sequence"]) != cycle_size
            or (len(result["flow"]) != cycle_size)
        ):
            return False
        sequence = [fraction(value) for value in result["sequence"]]
        flow = [fraction(value) for value in result["flow"]]
        if any(value is None for value in sequence + flow):
            return False
        marked_fraction = Fraction(len(marks), cycle_size)
        weights = [
            1 - marked_fraction if i in marks else -marked_fraction
            for i in range(cycle_size)
        ]
        optimum = minimum_cost(instance)
        return bool(
            optimum is not None
            and sum(sequence) == 0
            and all(
                abs(sequence[i] - sequence[(i + 1) % cycle_size]) <= 1
                for i in range(cycle_size)
            )
            and (sum(sequence[i] for i in marks) == optimum)
            and all(flow[i] - flow[i - 1] == weights[i] for i in range(cycle_size))
            and (sum(abs(value) for value in flow) == optimum)
            and (
                fraction(result["primal_value"])
                == fraction(result["dual_value"])
                == optimum
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def result_contract(result, instance):
    if not isinstance(result, dict) or set(result) != REQUIRED_RESULT_FIELDS:
        return False
    cycle_size = instance["cycle_size"]
    return bool(
        isinstance(result["sequence"], list)
        and isinstance(result["flow"], list)
        and (len(result["sequence"]) == cycle_size)
        and (len(result["flow"]) == cycle_size)
        and all(fraction(value) is not None for value in result["sequence"])
        and all(fraction(value) is not None for value in result["flow"])
        and (fraction(result["primal_value"]) is not None)
        and (fraction(result["dual_value"]) is not None)
    )


def main():
    submission = load_submission()
    instance = load_instance()
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result", {})
    math_correct = bool(
        instance is not None
        and isinstance(submission, dict)
        and result_contract(result, instance)
        and valid(result, instance)
    )
    correct = bool(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": float(correct)})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
