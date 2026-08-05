import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
EVIDENCE_PATH = "evidence/distance-audit.json"
LIMITATION = "The verifier replays exact rational instances and trusts the standard theorem that locally finite Euclidean subsets are closed; it does not machine-prove the universal topological argument."


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
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
        normalized = str(parsed)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if normalized == value else None


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


def _result(value: object, frozen: dict[str, Any]) -> bool:
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
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or frozen.get("prediction_label") is not False
    ):
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
    for offset, row in enumerate(pairs):
        if not _pair_row(row, start + offset):
            return False
    witnesses = value["epsilon_witnesses"]
    if not isinstance(witnesses, list) or not 4 <= len(witnesses) <= 8:
        return False
    prior_epsilon: Fraction | None = None
    prior_index = start - 1
    for witness in witnesses:
        parsed = _witness_fields(witness)
        if parsed is None:
            return False
        epsilon, index, distance = parsed
        if (
            index < start
            or index > 100000
            or index <= prior_index
            or distance != Fraction(1, index)
            or not distance < epsilon
        ):
            return False
        if prior_epsilon is not None and epsilon >= prior_epsilon:
            return False
        prior_epsilon, prior_index = epsilon, index
    return bool(
        value["natural_conclusion"] == "SEPARATED_SETS"
        and value["predicted_conclusion"] == "UNIFORM_POSITIVE_DISTANCE"
        and value["semantic_relation"] == "PREDICTION_STRICTLY_STRONGER"
        and value["missing_assumption"] == "COMPACTNESS_OF_ONE_SET"
        and value["local_finiteness_rule"] == "FIRST_COORDINATE_EQUALS_UNBOUNDED_INDEX"
        and value["closedness_conclusion"] == "BOTH_SETS_CLOSED_BY_LOCAL_FINITENESS"
    )


def _evidence(submission: dict[str, Any]) -> bool:
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    payload = read_evidence_json(evidence[0], expected_path=EVIDENCE_PATH)
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload["schema_version"] == "1"
        and payload["task_id"] == submission.get("task_id")
        and payload["result"] == submission.get("result")
        and payload["limitations"] == submission.get("limitations")
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    input_bound = workspace_input_is_bound()
    math_correct = bool(contract and _result(data.get("result"), _frozen()))
    evidence_valid = bool(math_correct and _evidence(data))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and input_bound
        and LIMITATION in data.get("limitations", [])
        and not false_certification
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
