import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W = Path("/app")
E = Path("/tests")

WEIGHTS = {
    1: [1, -1, 0, 0],
    2: [0, 1, -1, 0],
    3: [0, 0, 1, -1],
    4: [0, 0, 0, 1],
}
PRODUCT = [
    [1, 0, 0, 0],
    [-1, 1, 0, 0],
    [0, -1, 1, 0],
    [0, 0, -1, 1],
    [0, 0, 0, -1],
]


def _load_frozen_input():
    try:
        workspace, frozen = W / "input.json", E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _vector(value):
    return (
        value
        if isinstance(value, list)
        and len(value) == 4
        and all(type(x) is int for x in value)
        else None
    )


def _int_list(value, length):
    """Validate a list of exact integers with a fixed length, rejecting booleans."""
    return (
        value
        if isinstance(value, list)
        and len(value) == length
        and all(type(x) is int for x in value)
        else None
    )


def _json_equal(left, right):
    """Compare two JSON values without Python's bool/int coercion.

    Python treats ``True == 1`` as equal, so a certificate that replaces an
    integer ``1`` with boolean ``true`` would pass ``==`` despite not being an
    exact copy.  Serializing both values to canonical JSON distinguishes them.
    """
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _weights(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    parsed = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"power", "affine_coefficients"}:
            return None
        power, vector = item["power"], _vector(item["affine_coefficients"])
        if type(power) is not int or power in parsed or vector is None:
            return None
        parsed[power] = vector
    return parsed


def _result_is_valid(result, frozen):
    required = {
        "parameter_basis",
        "nonnegative_weights",
        "weight_sum",
        "one_minus_z_times_q",
        "root_identity_rhs",
        "controlled_powers",
        "reciprocal_reduction",
        "modulus_conclusion",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    if (
        frozen.get("parameter_basis") != ["1", "a", "b", "c"]
        or result["parameter_basis"] != frozen["parameter_basis"]
    ):
        return False
    weights = _weights(result["nonnegative_weights"])
    rhs = _weights(result["root_identity_rhs"])
    product = result["one_minus_z_times_q"]
    if (
        not isinstance(product, list)
        or len(product) != 5
        or any(_vector(v) is None for v in product)
    ):
        return False
    total = [sum(WEIGHTS[p][i] for p in WEIGHTS) for i in range(4)]
    weight_sum = _int_list(result["weight_sum"], 4)
    controlled_powers = _int_list(result["controlled_powers"], 4)
    return bool(
        weights == WEIGHTS
        and rhs == WEIGHTS
        and weight_sum is not None
        and weight_sum == total == [1, 0, 0, 0]
        and product == PRODUCT
        and controlled_powers is not None
        and controlled_powers == [1, 2, 3, 4]
        and result["reciprocal_reduction"]
        == "NONZERO_ROOT_LAMBDA_MAPS_TO_Q_ROOT_Z_EQUALS_1_OVER_LAMBDA"
        and result["modulus_conclusion"] == "ABS_LAMBDA_LE_1"
        and frozen.get("assumptions") == ["1-a>=0", "a-b>=0", "b-c>=0", "c>=0"]
    )


def main():
    submission, frozen = load_submission(), _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and _result_is_valid(submission.get("result"), frozen)
    )
    evidence = None
    if (
        protocol_ok
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
    ):
        evidence = read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/root-bound-certificate.json",
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and json_value_equal(evidence["result"], submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=evidence_valid,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(evidence_valid),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
