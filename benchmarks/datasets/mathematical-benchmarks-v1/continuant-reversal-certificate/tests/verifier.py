import json
import re
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
)

W = Path("/app")
E = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
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


def _evidence_matches(evidence):
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt")
    ):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        raw_text = target.read_text()
        text = raw_text.casefold()
    except (OSError, UnicodeError):
        return False
    if len(text) < 120:
        return False
    marker = next(
        (
            line[len("RESULT_JSON:") :].strip()
            for line in raw_text.splitlines()
            if line.startswith("RESULT_JSON:")
        ),
        None,
    )
    if marker is None:
        return False
    try:
        bound_proof = json.loads(marker)
    except (TypeError, ValueError):
        return False
    return (
        bound_proof == EXPECTED_PROOF
        and all(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in (
                r"tiling.{0,160}(?:recurrence|monomial)",
                r"recurrence.{0,160}(?:tiling|reflection)",
                r"reflection.{0,160}(?:support|monomial)",
            )
        )
        and all(
            phrase in text
            for phrase in (
                "tiling",
                "square",
                "domino",
                "a_i",
                "n-i",
                "coefficient",
            )
        )
    )


def main():
    submission = load_submission()
    frozen = _load_frozen_input()
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and _result_is_valid(submission.get("result"), frozen)
    )
    evidence_valid = bool(
        protocol_ok and math_correct and _evidence_matches(submission.get("witness"))
    )
    reward = aggregate_reward(
        correctness=math_correct,
        evidence_validity=evidence_valid,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
