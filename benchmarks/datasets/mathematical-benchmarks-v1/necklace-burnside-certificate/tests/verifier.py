import itertools
import json
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
N = 16
LIMITATIONS = ["FINITE_LENGTH_16_INSTANCE", "NO_GENERAL_ENUMERATION_THEOREM"]


def valid(word):
    return all(
        not (word[i] == word[(i + 1) % N] == word[(i + 2) % N]) for i in range(N)
    )


def rotation(word, k):
    return word[k:] + word[:k]


def reflection(word, k):
    return tuple(word[(k - i) % N] for i in range(N))


def derive():
    words = [word for word in itertools.product((0, 1), repeat=N) if valid(word)]
    rotations = [sum(rotation(word, k) == word for word in words) for k in range(N)]
    reflections = [sum(reflection(word, k) == word for word in words) for k in range(N)]
    representatives = sorted(
        {
            "".join(
                map(
                    str,
                    min(
                        [rotation(word, k) for k in range(N)]
                        + [reflection(word, k) for k in range(N)]
                    ),
                )
            )
            for word in words
        }
    )
    return {
        "valid_labelled_words": len(words),
        "rotation_fixed_counts": rotations,
        "reflection_fixed_counts": reflections,
        "burnside_numerator": sum(rotations + reflections),
        "orbit_count": len(representatives),
        "canonical_representatives": representatives,
    }


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


def matches(result):
    return exact_value(result, derive())


def result_shape_valid(result):
    """Check the result has the correct keys and scalar types without
    semantic equality, so schema violations are reported as protocol
    failures rather than only as mathematical incorrectness."""
    if not isinstance(result, dict):
        return False
    if set(result) != {
        "valid_labelled_words",
        "rotation_fixed_counts",
        "reflection_fixed_counts",
        "burnside_numerator",
        "orbit_count",
        "canonical_representatives",
    }:
        return False
    if (
        type(result["valid_labelled_words"]) is not int
        or result["valid_labelled_words"] < 1
    ):
        return False
    if (
        type(result["burnside_numerator"]) is not int
        or result["burnside_numerator"] < 1
    ):
        return False
    if type(result["orbit_count"]) is not int or result["orbit_count"] < 1:
        return False
    rfc = result["rotation_fixed_counts"]
    refc = result["reflection_fixed_counts"]
    if (
        not isinstance(rfc, list)
        or len(rfc) != 16
        or not all(type(x) is int and x >= 0 for x in rfc)
    ):
        return False
    if (
        not isinstance(refc, list)
        or len(refc) != 16
        or not all(type(x) is int and x >= 0 for x in refc)
    ):
        return False
    reps = result["canonical_representatives"]
    # Validate that every entry is a hashable string before constructing the
    # set so an unhashable JSON value (e.g. {}) fails closed instead of
    # raising an uncaught TypeError before reward.json is written.
    return (
        isinstance(reps, list)
        and len(reps) >= 1
        and all(type(r) is str and len(r) == 16 and set(r) <= {"0", "1"} for r in reps)
        and len(set(reps)) == len(reps)
        and reps == sorted(reps)
    )


def frozen():
    return workspace_input_is_bound(W / "input.json", tests=T)


def main():
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    # Protocol compliance includes the envelope contract and the result
    # shape so schema violations are reported as protocol failures, not
    # only as mathematical incorrectness.
    protocol_ok = bool(contract and result_shape_valid(result))
    evidence = None
    if (
        isinstance(submission, dict)
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    ):
        evidence = read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/answer.txt",
        )
    derived = derive()
    input_bound = frozen()
    # Mathematical correctness is evaluated independently of the envelope and
    # input binding so an assurance, protocol, or input-validity failure is not
    # misreported as wrong mathematics.  Input validity is reported as its own
    # diagnostic and only aggregate reward is gated on both.
    math_ok = bool(isinstance(result, dict) and matches(result))
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == expected["task_id"]
        and exact_value(evidence.get("result"), derived)
        and exact_value(evidence.get("result"), result)
        and evidence.get("limitations") == LIMITATIONS
        and evidence.get("limitations") == submission.get("limitations")
    )
    scope_ok = bool(
        isinstance(submission, dict)
        and submission.get("scope")
        == "ALL_LENGTH_16_BINARY_WORDS_AND_ALL_32_DIHEDRAL_ACTIONS"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("claimed_assurance"), str)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(
        protocol_ok
        and math_ok
        and input_bound
        and evidence_ok
        and scope_ok
        and assurance_ok
        and not false_cert
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol_ok),
                "correctness": float(math_ok),
                "input_binding": float(input_bound),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                # correct already requires assurance_ok; keep an honest binary reward
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
