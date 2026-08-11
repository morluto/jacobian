import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.public_reproductions_v1.support import _run_verifier

TASK = "closed-set-distance-strengthening-audit"
LIMITATION = (
    "The verifier replays exact rational instances and trusts the standard "
    "theorem that locally finite Euclidean subsets are closed; it does not "
    "machine-prove the universal topological argument."
)


def _oracle() -> dict[str, object]:
    return json.loads(
        (
            Path("benchmarks/datasets/public-reproductions-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(
    tmp_path: Path,
    submission: dict[str, object],
    *,
    evidence_payload: dict[str, object] | None = None,
):
    task = Path("benchmarks/datasets/public-reproductions-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    if evidence_payload is None:
        evidence_payload = {
            "schema_version": "1",
            "task_id": f"jacobian/{TASK}",
            "result": submission["result"],
            "limitations": submission["limitations"],
        }
    evidence_path = app / "evidence/distance-audit.json"
    evidence_path.write_text(json.dumps(evidence_payload, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return task, app, logs


def _verify(tmp_path: Path, submission: dict[str, object]):
    return _run_verifier(*_prepare(tmp_path, submission))


def _pairs(start: int) -> list[dict[str, object]]:
    return [
        {"index": n, "a": [str(n), "0"], "b": [str(n), f"1/{n}"], "distance": f"1/{n}"}
        for n in range(start, start + 8)
    ]


def test_oracle_and_alternative_family_are_accepted(tmp_path: Path) -> None:
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"]["start_index"] = 7
    alternative["result"]["point_pairs"] = _pairs(7)
    alternative["result"]["epsilon_witnesses"] = [
        {"epsilon": "1/4", "index": 7, "distance": "1/7"},
        {"epsilon": "1/8", "index": 9, "distance": "1/9"},
        {"epsilon": "1/16", "index": 17, "distance": "1/17"},
        {"epsilon": "1/32", "index": 33, "distance": "1/33"},
    ]
    assert _verify(tmp_path / "alternative", alternative).reward == 1.0


def test_accepts_consecutive_point_pairs_independent_of_array_order(
    tmp_path: Path,
) -> None:
    reordered = copy.deepcopy(_oracle())
    pairs = reordered["result"]["point_pairs"]
    pairs[1], pairs[6] = pairs[6], pairs[1]

    result = _verify(tmp_path / "reordered-pairs", reordered)
    assert result.details["protocol_compliance"] == 1.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 1.0


def test_pair_consecutiveness_is_protocol_not_mathematical_failure(
    tmp_path: Path,
) -> None:
    submission = copy.deepcopy(_oracle())
    pairs = _pairs(4)
    pairs.pop(3)
    pairs.append(_pairs(12)[0])
    submission["result"]["point_pairs"] = pairs

    result = _verify(tmp_path / "nonconsecutive-pairs", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_witness_start_bound_is_protocol_not_mathematical_failure(
    tmp_path: Path,
) -> None:
    submission = copy.deepcopy(_oracle())
    submission["result"]["start_index"] = 7
    submission["result"]["point_pairs"] = _pairs(7)
    submission["result"]["epsilon_witnesses"] = [
        {"epsilon": "1/3", "index": 4, "distance": "1/4"},
        {"epsilon": "1/5", "index": 8, "distance": "1/8"},
        {"epsilon": "1/10", "index": 11, "distance": "1/11"},
        {"epsilon": "1/20", "index": 21, "distance": "1/21"},
    ]

    result = _verify(tmp_path / "witness-before-start", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_rejects_corrupt_geometry_and_nonvanishing_gap(tmp_path: Path) -> None:
    for name, mutation in [
        (
            "coordinate",
            lambda result: result["point_pairs"][3]["b"].__setitem__(1, "1/8"),
        ),
        (
            "distance",
            lambda result: result["epsilon_witnesses"][2].update(distance="1/10"),
        ),
        (
            "ordering",
            lambda result: result["epsilon_witnesses"][2].update(epsilon="1/4"),
        ),
    ]:
        submission = copy.deepcopy(_oracle())
        mutation(submission["result"])
        assert _verify(tmp_path / name, submission).reward == 0.0


def test_noncanonical_rational_is_protocol_not_mathematical_failure(
    tmp_path: Path,
) -> None:
    noncanonical = copy.deepcopy(_oracle())
    noncanonical["result"]["point_pairs"][0]["distance"] = "2/8"
    result = _verify(tmp_path / "noncanonical", noncanonical)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_false_certification_is_rejected(tmp_path: Path) -> None:
    verified = copy.deepcopy(_oracle())
    verified["claimed_assurance"] = "VERIFIED"
    assert (
        _verify(tmp_path / "verified", verified).details["false_certification"] is True
    )


def test_schema_valid_unverified_assurance_is_not_a_protocol_failure(
    tmp_path: Path,
) -> None:
    submission = copy.deepcopy(_oracle())
    submission["claimed_assurance"] = "UNVERIFIED"

    result = _verify(tmp_path / "unverified", submission)
    assert result.details["protocol_compliance"] == 1.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 0.0
    assert result.reward == 0.0
    assert result.details["false_certification"] is False


# ---------------------------------------------------------------------------
# Adversarial regression tests for PR #493 review threads.
# -----------------------------------------------------------------------


def test_accepts_epsilon_witnesses_above_one(tmp_path: Path) -> None:
    """T2: the public contract allows any positive epsilon; no hidden < 1 bound."""
    submission = copy.deepcopy(_oracle())
    submission["result"]["epsilon_witnesses"] = [
        {"epsilon": "2", "index": 4, "distance": "1/4"},
        {"epsilon": "3/2", "index": 6, "distance": "1/6"},
        {"epsilon": "1", "index": 11, "distance": "1/11"},
        {"epsilon": "1/2", "index": 21, "distance": "1/21"},
    ]
    result = _verify(tmp_path / "epsilon-above-one", submission)
    assert result.reward == 1.0
    assert result.details["correctness"] == 1.0


def test_rejects_paraphrased_limitation_but_preserves_correctness(
    tmp_path: Path,
) -> None:
    """T1: a paraphrased limitation is rejected, but math stays correct."""
    submission = copy.deepcopy(_oracle())
    submission["limitations"] = [
        "The verifier checks exact rational instances and relies on the known "
        "theorem that locally finite Euclidean sets are closed without a "
        "machine proof of the general argument."
    ]
    result = _verify(tmp_path / "paraphrased", submission)
    assert result.reward == 0.0
    assert result.details["correctness"] == 1.0


def test_rejects_exponent_form_rational_without_crash(tmp_path: Path) -> None:
    """T3: exponent-form rationals must fail closed, not crash the verifier."""
    for field in ("distance", "a", "b"):
        submission = copy.deepcopy(_oracle())
        if field in ("a", "b"):
            submission["result"]["point_pairs"][0][field][0] = "1e4301"
        else:
            submission["result"]["point_pairs"][0][field] = "1e4301"
        result = _verify(tmp_path / f"exponent-{field}", submission)
        assert result.reward == 0.0
        assert result.details["correctness"] == 0.0
        assert result.details["protocol_compliance"] == 0.0

    huge_exponent = copy.deepcopy(_oracle())
    huge_exponent["result"]["point_pairs"][0]["distance"] = "1e" + "9" * 100_000
    result = _verify(tmp_path / "huge-exponent", huge_exponent)
    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0
    assert result.details["protocol_compliance"] == 0.0


def test_rejects_float_point_pair_indices(tmp_path: Path) -> None:
    """T4: float indices like 4.0 must not bypass integer validation."""
    submission = copy.deepcopy(_oracle())
    for _i, row in enumerate(submission["result"]["point_pairs"]):
        row["index"] = float(row["index"])
    result = _verify(tmp_path / "float-indices", submission)
    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0


def test_rejects_evidence_without_schema_version(tmp_path: Path) -> None:
    """T5: evidence missing the published schema_version field is rejected."""
    submission = copy.deepcopy(_oracle())
    payload = {
        "task_id": f"jacobian/{TASK}",
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    result = _run_verifier(
        *_prepare(tmp_path / "no-schema-version", submission, evidence_payload=payload)
    )
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0
    assert result.details["correctness"] == 1.0


def test_rejects_unhashable_assurance_without_crash(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["claimed_assurance"] = []
    result = _verify(tmp_path / "unhashable-assurance", submission)
    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 0.0
    assert result.reward == 0.0
    assert result.details["false_certification"] is False


def test_witness_ordering_is_protocol_not_mathematical_failure(
    tmp_path: Path,
) -> None:
    submission = copy.deepcopy(_oracle())
    witnesses = submission["result"]["epsilon_witnesses"]
    witnesses[0], witnesses[1] = witnesses[1], witnesses[0]

    result = _verify(tmp_path / "witness-order", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_tampered_input_preserves_correctness_and_gates_reward(
    tmp_path: Path,
) -> None:
    """T6: input binding is a separate diagnostic; math stays correct."""
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path / "tampered-input", submission)
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    input_path.write_text(json.dumps(input_data))
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_envelope_error_preserves_math_diagnostic(tmp_path: Path) -> None:
    """A duplicate evidence descriptor is protocol-invalid, not mathematically wrong."""
    submission = copy.deepcopy(_oracle())
    submission["evidence"].append(copy.deepcopy(submission["evidence"][0]))
    result = _verify(tmp_path / "duplicate-evidence", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_protocol_contract_gates_an_envelope_only_error(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["completeness"] = "INCOMPLETE"
    result = _verify(tmp_path / "invalid-envelope", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_evidence_validity_is_independent_of_math(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["result"]["point_pairs"][0]["distance"] = "1/999"
    result = _verify(tmp_path / "wrong-math", submission)
    assert result.details["protocol_compliance"] == 1.0
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_envelope_error_gates_aggregate_reward(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["conclusion"] = "UNSUPPORTED"
    result = _verify(tmp_path / "invalid-conclusion", submission)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_schema_failures_report_protocol_without_crashing(tmp_path: Path) -> None:
    float_index = copy.deepcopy(_oracle())
    float_index["result"]["point_pairs"][0]["index"] = 4.0
    result = _verify(tmp_path / "float-index", float_index)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0

    for name, limitations in (
        ("empty-limitations", []),
        ("null-limitations", None),
        ("numeric-limitations", 1),
    ):
        submission = copy.deepcopy(_oracle())
        submission["limitations"] = limitations
        result = _verify(tmp_path / name, submission)
        assert result.details["protocol_compliance"] == 0.0
        assert result.details["correctness"] == 1.0
        assert result.reward == 0.0
