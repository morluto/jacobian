import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    load_submission_raw,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")


def _frozen() -> dict[str, Any]:
    """Load the frozen verifier input from the hidden tests copy.

    Mathematical validation always uses the frozen verifier input, independent
    of whether the agent-visible input is bound.  Input binding is reported as
    a separate diagnostic via ``workspace_input_is_bound``.
    """
    try:
        hidden = TESTS / "input.json"
        if hidden.is_symlink():
            return {}
        value = json.loads(hidden.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _result_shape(value: object) -> bool:
    fields = {
        "start_index",
        "point_pairs",
        "epsilon_witnesses",
        "natural_conclusion",
        "predicted_conclusion",
        "semantic_relation",
        "missing_assumption",
        "local_finiteness_rule",
        "closedness_conclusion",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    if (
        type(value["start_index"]) is not int
        or not 4 <= value["start_index"] <= 20
        or not isinstance(value["point_pairs"], list)
        or len(value["point_pairs"]) != 8
        or not isinstance(value["epsilon_witnesses"], list)
        or not 4 <= len(value["epsilon_witnesses"]) <= 8
    ):
        return False
    for row in value["point_pairs"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "a", "b", "distance"}
            or type(row["index"]) is not int
            or not isinstance(row["a"], list)
            or len(row["a"]) != 2
            or not all(_rational(item) is not None for item in row["a"])
            or not isinstance(row["b"], list)
            or len(row["b"]) != 2
            or not all(_rational(item) is not None for item in row["b"])
            or _rational(row["distance"]) is None
        ):
            return False
    for witness in value["epsilon_witnesses"]:
        if (
            not isinstance(witness, dict)
            or set(witness) != {"epsilon", "index", "distance"}
            or _rational(witness["epsilon"]) is None
            or type(witness["index"]) is not int
            or not 4 <= witness["index"] <= 100000
            or _rational(witness["distance"]) is None
        ):
            return False
    return bool(
        value["natural_conclusion"] == "SEPARATED_SETS"
        and value["predicted_conclusion"] == "UNIFORM_POSITIVE_DISTANCE"
        and value["semantic_relation"] == "PREDICTION_STRICTLY_STRONGER"
        and value["missing_assumption"] == "COMPACTNESS_OF_ONE_SET"
        and value["local_finiteness_rule"] == "FIRST_COORDINATE_EQUALS_UNBOUNDED_INDEX"
        and value["closedness_conclusion"] == "BOTH_SETS_CLOSED_BY_LOCAL_FINITENESS"
    )


def _point(value: object) -> tuple[Fraction, Fraction] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    coordinates = tuple(_rational(item) for item in value)
    return None if any(item is None for item in coordinates) else coordinates


def _pair_row(row: object, index: int) -> bool:
    if (
        not isinstance(row, dict)
        or set(row) != {"index", "a", "b", "distance"}
        or _integer(row["index"]) is None
        or row["index"] != index
    ):
        return False
    a, b, distance = _point(row["a"]), _point(row["b"]), _rational(row["distance"])
    return (
        a == (Fraction(index), Fraction(0))
        and b == (Fraction(index), Fraction(1, index))
        and distance == Fraction(1, index)
    )


def _witness_fields(
    witness: object,
) -> tuple[Fraction, int, Fraction] | None:
    if not isinstance(witness, dict) or set(witness) != {
        "epsilon",
        "index",
        "distance",
    }:
        return None
    epsilon = _rational(witness["epsilon"])
    index = _integer(witness["index"])
    distance = _rational(witness["distance"])
    if epsilon is None or index is None or distance is None or not (epsilon > 0):
        return None
    return epsilon, index, distance


def _result_schema(value: object) -> bool:
    if not _result_shape(value):
        return False
    start = value["start_index"]
    pairs = value["point_pairs"]
    if {row["index"] for row in pairs} != set(range(start, start + len(pairs))):
        return False
    prior_epsilon: Fraction | None = None
    prior_index = -1
    for witness in value["epsilon_witnesses"]:
        parsed = _witness_fields(witness)
        if parsed is None:
            return False
        epsilon, index, _ = parsed
        if index < start or (
            prior_epsilon is not None
            and (epsilon >= prior_epsilon or index <= prior_index)
        ):
            return False
        prior_epsilon, prior_index = epsilon, index
    return True


def _result(value: object, frozen: dict[str, Any]) -> bool:
    if not _result_shape(value) or frozen.get("prediction_label") is not False:
        return False
    start = _integer(value["start_index"])
    pairs = value["point_pairs"]
    if (
        start is None
        or not 4 <= start <= 20
        or not isinstance(pairs, list)
        or len(pairs) != frozen.get("sample_count")
    ):
        return False
    for row in pairs:
        if not _pair_row(row, row["index"]):
            return False
    witnesses = value["epsilon_witnesses"]
    if not isinstance(witnesses, list) or not 4 <= len(witnesses) <= 8:
        return False
    for witness in witnesses:
        parsed = _witness_fields(witness)
        if parsed is None:
            return False
        epsilon, index, distance = parsed
        if index > 100000 or distance != Fraction(1, index) or not distance < epsilon:
            return False
    return bool(
        value["natural_conclusion"] == "SEPARATED_SETS"
        and value["predicted_conclusion"] == "UNIFORM_POSITIVE_DISTANCE"
        and value["semantic_relation"] == "PREDICTION_STRICTLY_STRONGER"
        and value["missing_assumption"] == "COMPACTNESS_OF_ONE_SET"
        and value["local_finiteness_rule"] == "FIRST_COORDINATE_EQUALS_UNBOUNDED_INDEX"
        and value["closedness_conclusion"] == "BOTH_SETS_CLOSED_BY_LOCAL_FINITENESS"
    )


def main() -> None:
    submission = load_submission_raw(require_input_binding=False)
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    math_correct = bool(_result(data.get("result"), _frozen()))
    protocol_ok = bool(
        isinstance(submission, dict)
        and set(data) == {"result"}
        and _result_schema(data.get("result"))
    )
    correct = bool(protocol_ok and input_bound and math_correct)
    reward = aggregate_reward(
        correctness=correct,
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "protocol_compliance": float(protocol_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
