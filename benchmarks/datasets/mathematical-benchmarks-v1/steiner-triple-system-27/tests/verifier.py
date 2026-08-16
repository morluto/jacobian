import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

E = Path("/tests")


def _valid_design(result, source):
    if not isinstance(result, dict) or not {"order", "blocks"} <= set(result):
        return False
    order = result.get("order")
    blocks = result.get("blocks")
    if type(order) is not int or order != source.get("point_set", {}).get(
        "cardinality"
    ):
        return False
    if not isinstance(blocks, list) or len(blocks) != source.get(
        "required_block_count"
    ):
        return False
    canonical = []
    for block in blocks:
        if (
            not isinstance(block, list)
            or len(block) != 3
            or any(type(point) is not int or not 0 <= point < order for point in block)
            or (len(set(block)) != 3)
        ):
            return False
        canonical.append(tuple(sorted(block)))
    if len(set(canonical)) != len(canonical):
        return False
    pairs = Counter(pair for block in canonical for pair in combinations(block, 2))
    expected_pairs = set(combinations(range(order), 2))
    return bool(set(pairs) == expected_pairs and set(pairs.values()) == {1})


def main():
    submission = load_submission_raw(require_input_binding=False)
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    source = json.loads((E / "input.json").read_text())
    contract = submission_matches_public_schema(submission)
    math_correct = _valid_design(data.get("result"), source)
    correct = bool(input_bound and contract and math_correct)
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "protocol_compliance": float(contract),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(output)


if __name__ == "__main__":
    main()
