from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/cycle-double-cover-multiplicity-audit"
)
CYCLES = [
    [0, 1, 2, 3, 4],
    [0, 1, 6, 8, 5],
    [1, 2, 7, 9, 6],
    [2, 3, 8, 5, 7],
    [3, 4, 9, 6, 8],
    [0, 4, 9, 7, 5],
]


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "cycle_double_cover_verifier", TASK / "tests/verifier.py"
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
    repair = [cycle.copy() for cycle in CYCLES]
    flawed = [cycle.copy() for cycle in repair[:-1]]
    flawed_counts = module._multiplicities(flawed)
    repair_counts = module._multiplicities(repair)
    assert flawed_counts and repair_counts
    return {
        "flawed_cycles": flawed,
        "flawed_multiplicities": flawed_counts,
        "non_double_edge_indices": [i for i, c in enumerate(flawed_counts) if c != 2],
        "repair_cycles": repair,
        "repair_multiplicities": repair_counts,
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_union_only_as_repair():
    module = _module()
    result = _result(module)
    result["repair_cycles"] = result["flawed_cycles"]
    result["repair_multiplicities"] = result["flawed_multiplicities"]
    assert not module.mathematics(result)


def test_rejects_omitted_bad_edge():
    module = _module()
    result = _result(module)
    result["non_double_edge_indices"] = result["non_double_edge_indices"][:-1]
    assert not module.mathematics(result)


def test_accepts_rotations_and_reversal():
    module = _module()
    result = _result(module)
    result["repair_cycles"] = [cycle[2:] + cycle[:2] for cycle in reversed(CYCLES)]
    assert module.mathematics(result)


def test_rejects_unhashable_cycle_vertex_without_crashing():
    module = _module()
    result = _result(module)
    result["flawed_cycles"][0][0] = []
    assert not module.mathematics(result)


def test_rejects_non_integer_reported_multiplicity():
    module = _module()
    result = _result(module)
    result["flawed_multiplicities"][0] = float(result["flawed_multiplicities"][0])
    assert not module.mathematics(result)


def test_raw_parser_rejects_duplicate_keys(tmp_path, monkeypatch):
    module = _module()
    submission = tmp_path / "submission.json"
    submission.write_text('{"result": {}, "result": {}}')
    monkeypatch.setattr(module, "Path", lambda _value: submission)
    assert module._raw() is None


def test_raw_parser_rejects_nonstandard_numbers(tmp_path, monkeypatch):
    module = _module()
    submission = tmp_path / "submission.json"
    monkeypatch.setattr(module, "Path", lambda _value: submission)
    for value in ("Infinity", "NaN", "1e10000"):
        submission.write_text('{"task_id": ' + value + "}")
        assert module._raw() is None


def test_evidence_comparison_preserves_json_types():
    module = _module()
    assert not module._json_equal({"count": 1}, {"count": True})
    assert not module._json_equal({"count": 2}, {"count": 2.0})


def test_evidence_comparison_rejects_nonfinite_and_handles_deep_values():
    module = _module()
    assert not module._json_equal({"value": float("inf")}, {"value": float("inf")})
    left: object = 0
    right: object = 0
    for _ in range(600):
        left = [left]
        right = [right]
    assert module._json_equal(left, right)


def test_evidence_copy_uses_raw_envelope_fields():
    module = _module()
    raw = {
        "task_id": "wrong-task",
        "result": _result(module),
        "limitations": ["different"],
    }
    matching = {
        "task_id": "wrong-task",
        "result": _result(module),
        "limitations": ["different"],
    }
    matching["schema_version"] = "1"
    assert module._evidence_payload_is_bound(matching, raw)
    canonical = dict(matching, task_id=module.TASK_ID, limitations=module.LIMITATIONS)
    assert not module._evidence_payload_is_bound(canonical, raw)


def test_assurance_value_is_independent_of_protocol():
    module = _module()
    assert module._assurance_is_calibrated(
        {"claimed_assurance": "CHECKED", "conclusion": "wrong"}
    )
    assert not module._assurance_is_calibrated({"claimed_assurance": True})


def test_reward_output_uses_host_contract(tmp_path, monkeypatch):
    module = _module()
    logs = tmp_path / "verifier"
    monkeypatch.setattr(module, "Path", lambda _value: logs)
    module._write({"reward": 1.0, "correctness": 1.0})
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert json.loads((logs / "reward-details.json").read_text()) == {
        "correctness": 1.0
    }


def test_declares_decoupled_input_binding():
    contract = json.loads((TASK / "tests/verifier_contract.json").read_text())
    assert contract == {"schema_version": "1", "input_binding_decoupled": True}
