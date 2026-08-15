import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    valid_sha256_uri,
)

W, T = Path("/app"), Path("/tests")
_CLASSIFICATIONS = frozenset(
    {"BIJECTIVE", "INJECTIVE_NOT_SURJECTIVE", "SURJECTIVE_NOT_INJECTIVE"}
)
_ROW_KEYS = frozenset(
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
    if set(row) != _ROW_KEYS:
        return False
    if not isinstance(row["id"], str):
        return False
    if not isinstance(row["classification"], str):
        return False
    if row["classification"] not in _CLASSIFICATIONS:
        return False
    if type(row["commutes"]) is not bool:
        return False
    if not (_is_int(row["checked_subsets"]) and 1 <= row["checked_subsets"] <= 32):
        return False
    return all(
        _int_set_ok(row[key])
        for key in ("first_failure", "left_image", "right_complement")
    )


def _result_schema_ok(result):
    """Validate the nested result structure against the published schema."""
    if (
        not isinstance(result, dict)
        or set(result) != {"cases"}
        or not isinstance(result["cases"], list)
        or len(result["cases"]) != 3
        or any(not isinstance(row, dict) for row in result["cases"])
    ):
        return False
    return all(_row_schema_ok(row) for row in result["cases"])


def _evidence_descriptor_ok(descriptor):
    """Validate the evidence descriptor shape and digest pattern."""
    return bool(
        isinstance(descriptor, dict)
        and set(descriptor) == {"path", "sha256"}
        and descriptor.get("path") == "evidence/image-complement-certificate.json"
        and valid_sha256_uri(descriptor.get("sha256"))
    )


def _normalize_row(row):
    out = dict(row)
    for key in ("first_failure", "left_image", "right_complement"):
        value = out[key]
        out[key] = None if value is None else sorted(value)
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
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_ok = bool(protocol_ok and valid(result))
    evidence_descriptor = (
        data["witness"][0]
        if isinstance(data.get("witness"), list) and len(data["witness"]) == 1
        else None
    )
    ev = (
        read_evidence_json(
            evidence_descriptor,
            expected_path="evidence/image-complement-certificate.json",
            max_bytes=16 * 1024 * 1024,
        )
        if evidence_descriptor is not None
        else None
    )
    ev_ok = bool(
        protocol_ok
        and ev
        and set(ev) == {"schema_version", "task_id", "result"}
        and type(ev.get("schema_version")) is str
        and ev.get("schema_version") == "1"
        and type(ev.get("task_id")) is str
        and ev.get("task_id") == expected["task_id"]
        and json_value_equal(ev.get("result"), data.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
