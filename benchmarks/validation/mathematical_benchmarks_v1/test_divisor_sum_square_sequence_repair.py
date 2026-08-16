import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "divisor-sum-square-sequence-repair"


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
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
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


def test_alternate_piecewise_construction_accepted(tmp_path):
    """A different valid power-of-two construction must earn full reward."""

    sub = _oracle()
    sub["result"]["default_exponent_offset"] = 0
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


def test_completeness_partial_rejected(tmp_path):
    """A PARTIAL completeness value must not earn reward."""

    sub = _oracle()
    sub["completeness"] = "PARTIAL"
    assert _verify(tmp_path / "partial", sub).reward == 0


def test_legacy_prime_formula_field_rejected(tmp_path):
    """The removed decorative formula field must not bypass the typed contract."""

    sub = _oracle()
    sub["result"]["prime_formula"] = "2^(p+1)+p*2^(2p)"
    result = _verify(tmp_path / "garbage_formula", sub)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0


def test_legacy_threshold_rule_field_rejected(tmp_path):
    """The removed prose rule field must not bypass the typed contract."""

    sub = _oracle()
    sub["result"]["threshold_rule"] = "n>=max(2,k)_implies_2^k_divides_a_n"
    result = _verify(tmp_path / "garbage_threshold", sub)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0
