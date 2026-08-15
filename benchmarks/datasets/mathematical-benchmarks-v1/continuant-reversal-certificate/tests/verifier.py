import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
EXPECTED_PROOF = {
    "base_cases": [{"length": 0, "value": "1"}, {"length": 1, "value": "1"}],
    "partition": "each_tiling_ends_in_a_square_or_domino",
    "recurrence": "F_(k+1)=F_k+a_k*F_(k-1)",
    "reflection": "i_maps_to_n_minus_i_and_is_an_involution",
}


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _supports(n):
    values = [()]
    for edge in range(1, n):
        values += [
            (*support, edge)
            for support in values
            if not support or support[-1] < edge - 1
        ]
    return sorted(values, key=lambda item: (len(item), item))


def _parse_supports(value):
    if not isinstance(value, list):
        return None
    parsed = []
    for support in value:
        if not isinstance(support, list):
            return None
        normalized = []
        for item in support:
            if type(item) is int:
                normalized.append(item)
            elif type(item) is float and item.is_integer():
                normalized.append(int(item))
            else:
                return None
        if normalized != sorted(set(normalized)):
            return None
        parsed.append(tuple(normalized))
    return parsed


def _pairs_are_valid(value, supports, n):
    if not isinstance(value, list) or len(value) != len(supports):
        return False
    expected = {
        tuple(support): tuple(sorted(n - item for item in support))
        for support in supports
    }
    actual = {}
    for pair in value:
        if not isinstance(pair, dict) or set(pair) != {"forward", "reflected"}:
            return False
        forward = _parse_supports([pair["forward"]])
        reflected = _parse_supports([pair["reflected"]])
        if forward is None or reflected is None:
            return False
        key = forward[0]
        if key in actual or key not in expected or reflected[0] != expected[key]:
            return False
        actual[key] = reflected[0]
    return actual == expected


def _result_is_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "board_length",
        "forward_monomials",
        "reverse_monomials",
        "reflection_pairs",
        "recurrence_contract",
        "proof_obligations",
        "conclusion",
    }:
        return False
    n = frozen.get("board_length")
    if n != 10 or result["board_length"] != n:
        return False
    supports = _supports(n)
    forward = _parse_supports(result["forward_monomials"])
    reverse = _parse_supports(result["reverse_monomials"])
    try:
        forward_set = set(forward)
        reverse_set = set(reverse)
    except TypeError:
        return False
    proof_obligations = result["proof_obligations"]
    proof_lengths_ok = (
        isinstance(proof_obligations, dict)
        and isinstance(proof_obligations.get("base_cases"), list)
        and len(proof_obligations["base_cases"]) == 2
        and all(
            isinstance(case, dict)
            and type(case.get("length")) is int
            and isinstance(case.get("value"), str)
            for case in proof_obligations["base_cases"]
        )
    )
    return bool(
        len(forward) == len(forward_set) == len(supports)
        and len(reverse) == len(reverse_set) == len(supports)
        and forward_set == set(supports)
        and reverse_set == set(supports)
        and _pairs_are_valid(result["reflection_pairs"], supports, n)
        and isinstance(result["recurrence_contract"], dict)
        and isinstance(result["recurrence_contract"].get("initial_values"), list)
        and all(
            type(value) is int
            for value in result["recurrence_contract"]["initial_values"]
        )
        and result["recurrence_contract"]
        == {
            "initial_values": [1, 1],
            "forward_coefficient": "a_k",
            "reverse_coefficient": "a_(n-k)",
            "reflection_rule": "i_maps_to_n_minus_i",
        }
        and proof_lengths_ok
        and proof_obligations == EXPECTED_PROOF
        and result["conclusion"] == "FINAL_POLYNOMIALS_EQUAL"
    )


def main():
    submission = load_submission()
    frozen = _load_frozen_input()
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and _result_is_valid(submission.get("result"), frozen)
    )
    reward = float(protocol_ok and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
