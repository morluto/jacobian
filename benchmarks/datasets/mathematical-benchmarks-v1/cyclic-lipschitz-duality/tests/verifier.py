import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

FROZEN_INPUT = Path(__file__).with_name("input.json")
RATIONAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")
REQUIRED_RESULT_FIELDS = frozenset({"sequence", "flow", "primal_value", "dual_value"})


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
        or not isinstance(marked_indices, list)
        or not marked_indices
        or any(type(index) is not int for index in marked_indices)
        or len(set(marked_indices)) != len(marked_indices)
        or any(index < 1 or index > cycle_size for index in marked_indices)
    ):
        return None
    return {
        "cycle_size": cycle_size,
        "marked_indices": tuple(index - 1 for index in marked_indices),
    }


def fraction(text):
    if type(text) is not str or len(text) > 128 or RATIONAL_RE.fullmatch(text) is None:
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None
    return value if str(value) == text else None


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
            or len(result["flow"]) != cycle_size
            or any(type(value) is not str for value in result["sequence"])
            or any(type(value) is not str for value in result["flow"])
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
            and sum(sequence[i] for i in marks) == optimum
            and all(flow[i] - flow[i - 1] == weights[i] for i in range(cycle_size))
            and sum(abs(value) for value in flow) == optimum
            and result["primal_value"] == result["dual_value"] == str(optimum)
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
        and len(result["sequence"]) == cycle_size
        and len(result["flow"]) == cycle_size
        and all(type(value) is str for value in result["sequence"])
        and all(type(value) is str for value in result["flow"])
        and type(result["primal_value"]) is str
        and type(result["dual_value"]) is str
    )


def scope_is_correct(scope, instance):
    if type(scope) is not str or len(scope) > 4096:
        return False
    folded = " ".join(scope.casefold().split())
    if re.search(r"\b(?:not|without|except|fewer|more)\b", folded):
        return False
    if not re.search(r"\b(?:cycle|cyclic)\b", folded):
        return False
    if "marked" not in folded or not re.search(
        r"\b(?:index|indices|position|positions)\b", folded
    ):
        return False
    numbers = {int(value) for value in re.findall(r"(?<![0-9])[0-9]+(?![0-9])", folded)}
    expected = {
        instance["cycle_size"],
        *(index + 1 for index in instance["marked_indices"]),
    }
    return numbers == expected


def evidence_is_valid(evidence, result, instance):
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    payload = read_evidence_json(evidence[0], expected_path="evidence/answer.txt")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "task_id",
        "primal",
        "dual",
        "optimality",
        "limitations",
    }:
        return False
    optimum = minimum_cost(instance)
    primal = payload["primal"]
    dual = payload["dual"]
    return bool(
        payload["schema_version"] == "1"
        and payload["task_id"] == "jacobian/cyclic-lipschitz-duality"
        and isinstance(primal, dict)
        and set(primal)
        == {"sequence_length", "zero_sum", "adjacent_bound", "objective"}
        and type(primal["sequence_length"]) is int
        and primal["sequence_length"] == instance["cycle_size"]
        and primal["zero_sum"] == "0"
        and primal["adjacent_bound"] == "1"
        and primal["objective"] == str(optimum)
        and isinstance(dual, dict)
        and set(dual) == {"divergence", "l1_cost", "minimum_cost"}
        and isinstance(dual["divergence"], str)
        and "".join(dual["divergence"].split()).casefold() == "q_i-q_(i-1)=w_i"
        and dual["l1_cost"] == str(optimum)
        and dual["minimum_cost"] == str(optimum)
        and payload["optimality"] == "weak_duality_after_median_minimum"
        and payload["limitations"] == result.get("limitations")
    )


def main():
    submission = load_submission()
    instance = load_instance()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/cyclic-lipschitz-duality",
        conclusion="OPTIMUM_CERTIFIED",
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result", {})
    contract = bool(
        instance is not None
        and contract
        and result_contract(result, instance)
        and isinstance(data.get("limitations"), list)
        and bool(data["limitations"])
        and all(type(item) is str for item in data["limitations"])
    )
    math_correct = bool(instance is not None and valid(result, instance))
    evidence_valid = bool(
        instance is not None and evidence_is_valid(data.get("evidence"), data, instance)
    )
    scope_correct = bool(contract and scope_is_correct(data.get("scope"), instance))
    assurance_correct = bool(data.get("claimed_assurance") == "COMPUTED")
    false_certification = bool(
        false_verified_claim(submission, verification_record_bound=False)
    )
    reward = (
        1.0
        if contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
        else 0.0
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
