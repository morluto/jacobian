import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1.support import _run_verifier

TASK = "divisor-sum-square-sequence-repair"
TASK_ID = f"jacobian/{TASK}"


def _oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _verify(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    path = app / "evidence/sequence-construction.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def _probe(p):
    a = 2 ** (p + 1) + p * 2 ** (2 * p)
    return {"prime": p, "a_p": a, "b_p": 1 + p * a, "square_root": 1 + p * 2**p}


def _alternate_probe(p):
    """Probe for the alternate construction a_n=2^n, a_p=2^(p+2)+4*p*2^(2p)."""

    a = 2 ** (p + 2) + 4 * p * 2 ** (2 * p)
    return {"prime": p, "a_p": a, "b_p": 1 + p * a, "square_root": 1 + 2 * p * 2**p}


def test_oracle_and_alternative_primes(tmp_path):
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alt = _oracle()
    alt["result"]["probes"] = [_probe(p) for p in (13, 17, 19, 23)]
    assert _verify(tmp_path / "alt", alt).reward == 1.0


def test_non_ascending_probe_order_accepted(tmp_path):
    """Distinct valid probes in any order earn full reward (T9)."""
    sub = _oracle()
    sub["result"]["probes"] = [_probe(p) for p in (5, 3, 7, 11)]
    assert _verify(tmp_path / "perm", sub).reward == 1.0


def test_corruption_and_false_assurance_fail(tmp_path):
    for name, mutate in [
        ("square", lambda s: s["result"]["probes"][0].update(b_p=624)),
        ("assurance", lambda s: s.update(claimed_assurance="VERIFIED")),
    ]:
        submission = copy.deepcopy(_oracle())
        mutate(submission)
        assert _verify(tmp_path / name, submission).reward == 0


def test_false_verified_preserves_correctness(tmp_path):
    """Unsupported VERIFIED must force reward 0 but not erase correctness (T10)."""
    sub = _oracle()
    sub["claimed_assurance"] = "VERIFIED"
    result = _verify(tmp_path / "false_verified", sub)
    assert result.reward == 0
    assert result.details["correctness"] == 1.0
    assert result.details["false_certification"] is True


def test_oversized_probe_rejected(tmp_path):
    """A huge probe prime must not crash the verifier; reward is 0 (T3)."""
    sub = _oracle()
    sub["result"]["probes"][0]["prime"] = 10**400
    result = _verify(tmp_path / "huge_probe", sub)
    assert result.reward == 0


def test_boolean_construction_fields_rejected(tmp_path):
    """JSON true must not satisfy integer fields (T8)."""
    for field in ("a_1", "default_exponent_offset"):
        sub = _oracle()
        sub["result"][field] = True
        assert _verify(tmp_path / f"bool_{field}", sub).reward == 0


def test_float_probe_values_rejected(tmp_path):
    """Integral floats must not satisfy integer probe fields (T11)."""
    sub = _oracle()
    sub["result"]["probes"][0]["a_p"] = float(sub["result"]["probes"][0]["a_p"])
    assert _verify(tmp_path / "float_a_p", sub).reward == 0


def test_old_threshold_rule_string_accepted(tmp_path):
    """The threshold_rule is a descriptive label; the verifier checks math."""

    sub = _oracle()
    sub["result"]["threshold_rule"] = "n>=k_implies_a_n_divisible_by_2^k"
    assert _verify(tmp_path / "old_threshold", sub).reward == 1.0


def test_alternate_piecewise_construction_accepted(tmp_path):
    """A different valid power-of-two construction must earn full reward."""

    sub = _oracle()
    sub["result"]["default_exponent_offset"] = 0
    sub["result"]["prime_formula"] = "2^n"
    sub["result"]["threshold_rule"] = "n>=max(2,k)_implies_2^k_divides_a_n"
    sub["result"]["probes"] = [_alternate_probe(p) for p in (3, 5, 7, 11)]
    assert _verify(tmp_path / "alt_construction", sub).reward == 1.0


def test_probe_not_divisible_by_2p_rejected(tmp_path):
    """A probe whose a_p is not divisible by 2^p fails the threshold property."""

    sub = _oracle()
    sub["result"]["probes"][0]["a_p"] = 2**3  # not divisible by 2^3=8? 8%8==0, try odd
    sub["result"]["probes"][0]["a_p"] = 3
    sub["result"]["probes"][0]["b_p"] = 1 + 3 * 3
    sub["result"]["probes"][0]["square_root"] = 0
    assert _verify(tmp_path / "bad_threshold", sub).reward == 0


def test_evidence_type_coercion_rejected(tmp_path):
    """Evidence with bool/float values must not match integer submission result."""

    sub = copy.deepcopy(_oracle())
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": copy.deepcopy(sub["result"]),
        "limitations": sub["limitations"],
    }
    evidence["result"]["a_1"] = True
    path = app / "evidence/sequence-construction.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    sub["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(sub))
    result = _run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0


def test_completeness_partial_rejected(tmp_path):
    """A PARTIAL completeness value must not earn reward."""

    sub = _oracle()
    sub["completeness"] = "PARTIAL"
    assert _verify(tmp_path / "partial", sub).reward == 0


def test_garbage_prime_formula_rejected(tmp_path):
    """A prime_formula that is not a power-of-two expression must fail."""

    sub = _oracle()
    sub["result"]["prime_formula"] = "not a formula"
    result = _verify(tmp_path / "garbage_formula", sub)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0


def test_garbage_threshold_rule_rejected(tmp_path):
    """A threshold_rule that does not describe the divisibility property must fail."""

    sub = _oracle()
    sub["result"]["threshold_rule"] = "false"
    result = _verify(tmp_path / "garbage_threshold", sub)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0


def test_recursive_evidence_comparison_does_not_crash(tmp_path):
    """Deeply nested evidence must fail closed without raising RecursionError."""

    sub = copy.deepcopy(_oracle())
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    nested: list = []
    for _ in range(500):
        nested = [nested]
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": nested,
        "limitations": sub["limitations"],
    }
    path = app / "evidence/sequence-construction.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    sub["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(sub))
    result = _run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0
