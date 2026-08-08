from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/tutte-flow-domain-audit"
FLAWED = [3, 2, 1, 4, 0, 1, 2, 4, 2, 1, 2, 1, 1, 2, 4]
REPAIR = [2, 1, 4, 3, 1, 1, 3, 3, 3, 1, 2, 1, 2, 1, 4]


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "tutte_flow_verifier", TASK / "tests/verifier.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_submission_is_bounded_before_read(monkeypatch):
    module = _module()

    class UnreadablePath:
        def __init__(self, _value):
            pass

        def read_text(self):
            raise AssertionError("oversized submission must not be read")

    def reject_oversized(_path, *, max_bytes):
        assert max_bytes == module.MAX_SUBMISSION_BYTES
        return False

    monkeypatch.setattr(module, "Path", UnreadablePath)
    monkeypatch.setattr(module, "is_regular_bounded_file", reject_oversized)
    assert module._raw() is None


def _result(module):
    return {
        "flawed_flow": FLAWED,
        "flawed_balances": module._balances(FLAWED),
        "zero_edge_index": 4,
        "repair_flow": REPAIR,
        "repair_balances": module._balances(REPAIR),
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_zero_flow_shortcut():
    module = _module()
    result = _result(module)
    result["flawed_flow"] = [0] * 15
    result["flawed_balances"] = [0] * 10
    assert not module.mathematics(result)


def test_rejects_zero_in_repair():
    module = _module()
    result = _result(module)
    result["repair_flow"] = FLAWED
    result["repair_balances"] = [0] * 10
    assert not module.mathematics(result)


def test_accepts_scalar_multiple():
    module = _module()
    result = _result(module)
    scaled = [(2 * x) % 5 for x in REPAIR]
    result["repair_flow"] = scaled
    result["repair_balances"] = module._balances(scaled)
    assert module.mathematics(result)


def test_rejects_non_integer_reported_balances():
    module = _module()
    result = _result(module)
    result["flawed_balances"][0] = False
    assert not module.mathematics(result)


def test_rejects_non_integer_zero_edge_index():
    module = _module()
    result = _result(module)
    result["zero_edge_index"] = 4.0
    assert not module.mathematics(result)


def test_evidence_comparison_preserves_json_types():
    module = _module()
    assert not module._json_equal({"balance": 0}, {"balance": False})
    assert not module._json_equal({"index": 4}, {"index": 4.0})


def test_evidence_comparison_rejects_excessive_nesting():
    module = _module()
    left = right = 0
    for _ in range(129):
        left = [left]
        right = [right]
    assert not module._json_equal(left, right)


def test_raw_parser_rejects_duplicate_keys(tmp_path, monkeypatch):
    module = _module()
    submission = tmp_path / "submission.json"
    submission.write_text('{"result": {}, "result": {}}')
    monkeypatch.setattr(module, "Path", lambda _value: submission)
    assert module._raw() is None


def test_assurance_value_is_independent_of_protocol():
    module = _module()
    assert module._assurance_is_calibrated(
        {"claimed_assurance": "CHECKED", "conclusion": "wrong"}
    )
    assert not module._assurance_is_calibrated({"claimed_assurance": True})


def test_evidence_metadata_is_bound_to_raw_envelope():
    module = _module()
    raw = {
        "task_id": module.TASK_ID,
        "result": {"value": 1},
        "limitations": module.LIMITATIONS,
    }
    payload = {"schema_version": "1", **raw}
    assert module._evidence_payload_is_bound(payload, raw)
    assert not module._evidence_payload_is_bound(
        payload, {**raw, "task_id": "different-task"}
    )
    assert not module._evidence_payload_is_bound(
        payload, {**raw, "limitations": ["different-limit"]}
    )


def test_verifier_contract_declares_diagnostic_splits():
    contract = json.loads((TASK / "tests/verifier_contract.json").read_text())
    assert contract == {
        "schema_version": "1",
        "input_binding_decoupled": True,
        "scope_independent_assurance": True,
    }


def test_reward_output_uses_host_contract(tmp_path, monkeypatch):
    module = _module()
    logs = tmp_path / "verifier"
    monkeypatch.setattr(module, "Path", lambda _value: logs)
    module._write({"reward": 1.0, "correctness": 1.0})
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert json.loads((logs / "reward-details.json").read_text()) == {
        "correctness": 1.0
    }
