import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["S4_CLASSIFICATION_NOT_REPROVED", "CHEBOTAREV_THEOREM_TRUSTED"]


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def determinant(a):
    a = [[Fraction(x) for x in row] for row in a]
    out = Fraction(1)
    for col in range(len(a)):
        pivot = next((r for r in range(col, len(a)) if a[r][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            out = -out
        out *= a[col][col]
        p = a[col][col]
        for j in range(col, len(a)):
            a[col][j] /= p
        for r in range(col + 1, len(a)):
            q = a[r][col]
            for j in range(col, len(a)):
                a[r][j] -= q * a[col][j]
    return int(out)


def discriminant():
    f, g = [1, 0, 0, -4, 1], [4, 0, 0, -4]
    syl = []
    for shift in range(3):
        syl.append([0] * shift + f + [0] * (2 - shift))
    for shift in range(4):
        syl.append([0] * shift + g + [0] * (3 - shift))
    return determinant(syl)


def partition(p):
    seen, cycles = set(), []
    for i in range(4):
        if i not in seen:
            j, n = i, 0
            while j not in seen:
                seen.add(j)
                n += 1
                j = p[j]
            cycles.append(n)
    return "+".join(map(str, sorted(cycles, reverse=True)))


def table():
    counts = Counter(partition(p) for p in itertools.permutations(range(4)))
    fixed = {}
    for p in itertools.permutations(range(4)):
        key = partition(p)
        fixed[key] = max(fixed.get(key, 0), sum(1 for i in range(4) if p[i] == i))
    order = ["1+1+1+1", "2+1+1", "2+2", "3+1", "4"]
    return [
        {
            "partition": k,
            "class_size": counts[k],
            "fixed_points": fixed[k],
            "contributes": fixed[k] > 0,
        }
        for k in order
    ]


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(exact_value(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                exact_value(value, target)
                for value, target in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def cycle_table_matches(actual, expected_table):
    expected_by_partition = {row["partition"]: row for row in expected_table}
    if not isinstance(actual, list) or len(actual) != len(expected_table):
        return False
    seen = set()
    for row in actual:
        if not isinstance(row, dict) or set(row) != {
            "partition",
            "class_size",
            "fixed_points",
            "contributes",
        }:
            return False
        key = row["partition"]
        if not isinstance(key, str) or key in seen:
            return False
        seen.add(key)
        target = expected_by_partition.get(key)
        if target is None or not exact_value(row, target):
            return False
    return True


def valid(r):
    expected_table = table()
    defects = {
        "MOD2_IRREDUCIBILITY_FALSE",
        "DISCRIMINANT_WRONG",
        "DOUBLE_TRANSPOSITIONS_HAVE_NO_FIXED_POINTS",
        "PUBLISHED_DENSITY_WRONG",
    }
    return (
        isinstance(r, dict)
        and set(r)
        == {
            "mod2_factor",
            "actual_discriminant",
            "cycle_types",
            "fixed_point_total",
            "density",
            "encoded_answer",
            "defects",
        }
        and exact_value(r["mod2_factor"], "(x+1)^4")
        and [1, 0, 0, 0, 1] == [1, 4 % 2, 6 % 2, 4 % 2, 1]
        and exact_value(r["actual_discriminant"], discriminant())
        and r["actual_discriminant"] == -6656
        and cycle_table_matches(r["cycle_types"], expected_table)
        and type(r["fixed_point_total"]) is int
        and r["fixed_point_total"]
        == sum(x["class_size"] for x in expected_table if x["contributes"])
        == 15
        and exact_value(r["density"], {"numerator": 5, "denominator": 8})
        and type(r["encoded_answer"]) is int
        and r["encoded_answer"] == 508
        and isinstance(r["defects"], list)
        and len(r["defects"]) == len(defects)
        and all(type(value) is str for value in r["defects"])
        and len(set(r["defects"])) == len(r["defects"])
        and set(r["defects"]) == defects
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    evidence_descriptor = (
        s["evidence"][0]
        if isinstance(s, dict)
        and isinstance(s.get("evidence"), list)
        and len(s.get("evidence", [])) == 1
        else None
    )
    ev = (
        read_evidence_json(
            evidence_descriptor, expected_path="evidence/chebotarev-audit.json"
        )
        if evidence_descriptor is not None
        else None
    )
    math_ok = bool(frozen() and isinstance(s, dict) and valid(s.get("result")))
    evidence_ok = bool(
        ev
        and set(ev) == {"schema_version", "task_id", "result", "limitations"}
        and ev.get("schema_version") == "1"
        and ev.get("task_id") == expected["task_id"]
        and exact_value(
            ev.get("result"), s.get("result") if isinstance(s, dict) else None
        )
        and ev.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        contract
        and s.get("scope") == "CONDITIONAL_ON_FROZEN_S4_PREMISE"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(contract and s.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(s, verification_record_bound=False)
    correct = math_ok and evidence_ok and scope_ok and not false_cert
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
