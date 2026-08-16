import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
_CLASSIFICATIONS = frozenset(
    {"BIJECTIVE", "INJECTIVE_NOT_SURJECTIVE", "SURJECTIVE_NOT_INJECTIVE"}
)
_ROW_KEYS_COMMUTES = frozenset({"id", "classification", "commutes", "checked_subsets"})
_ROW_KEYS_NOT_COMMUTES = frozenset(
    {
        "id",
        "classification",
        "commutes",
        "checked_subsets",
        "first_failure",
        "left_image",
        "right_complement",
    }
)


def expected_case(case):
    n, m, mapping = case["domain_size"], case["codomain_size"], case["mapping"]
    injective = len(set(mapping)) == n
    surjective = set(mapping) == set(range(m))
    classification = (
        "BIJECTIVE"
        if injective and surjective
        else "INJECTIVE_NOT_SURJECTIVE"
        if injective
        else "SURJECTIVE_NOT_INJECTIVE"
    )
    failure = None
    for mask in range(1 << n):
        subset = {i for i in range(n) if mask >> i & 1}
        left = sorted({mapping[i] for i in range(n) if i not in subset})
        right = sorted(set(range(m)) - {mapping[i] for i in subset})
        if left != right and failure is None:
            failure = (sorted(subset), left, right)
    return {
        "id": case["id"],
        "classification": classification,
        "commutes": failure is None,
        "checked_subsets": 1 << n,
        "first_failure": None if failure is None else failure[0],
        "left_image": None if failure is None else failure[1],
        "right_complement": None if failure is None else failure[2],
    }


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_set_ok(value):
    """A null or unique in-range integer list per the published schema."""
    if value is None:
        return True
    if not isinstance(value, list):
        return False
    if not all(_is_int(item) and 0 <= item <= 4 for item in value):
        return False
    return len(set(value)) == len(value)


def _row_schema_ok(row):
    if not isinstance(row.get("id"), str):
        return False
    if not isinstance(row.get("classification"), str):
        return False
    if row["classification"] not in _CLASSIFICATIONS:
        return False
    if type(row.get("commutes")) is not bool:
        return False
    if not (_is_int(row.get("checked_subsets")) and 1 <= row["checked_subsets"] <= 32):
        return False
    if row["commutes"]:
        return set(row) == _ROW_KEYS_COMMUTES
    if set(row) != _ROW_KEYS_NOT_COMMUTES:
        return False
    return all(
        _int_set_ok(row[key])
        for key in ("first_failure", "left_image", "right_complement")
    )


def _normalize_row(row):
    out = dict(row)
    if row["commutes"]:
        out["first_failure"] = None
        out["left_image"] = None
        out["right_complement"] = None
    else:
        for key in ("first_failure", "left_image", "right_complement"):
            out[key] = sorted(out[key]) if out[key] is not None else None
    return out


def valid(result):
    if (
        not isinstance(result, dict)
        or set(result) != {"cases"}
        or not isinstance(result["cases"], list)
        or len(result["cases"]) != 3
    ):
        return False
    rows = result["cases"]
    if any(not isinstance(row, dict) for row in rows):
        return False
    if any(not _row_schema_ok(row) for row in rows):
        return False
    frozen_cases = json.loads((T / "input.json").read_text())["cases"]
    expected = {
        case["id"]: _normalize_row(expected_case(case)) for case in frozen_cases
    }
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            return False
        by_id[row["id"]] = _normalize_row(row)
    if set(by_id) != set(expected):
        return False
    return all(by_id[cid] == expected[cid] for cid in expected)


def main():
    _input_binding = workspace_input_is_bound()
    submission = load_submission(W / "submission.json", require_input_binding=False)
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_ok = bool(protocol_ok and valid(result))
    reward = aggregate_reward(
        correctness=math_ok,
        protocol_ok=protocol_ok,
    )
    if not _input_binding:
        reward = 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(_input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
