from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import sys
from itertools import product
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/illumination-strictness-audit"
VERTICES = list(product((-1, 1), repeat=3))


def _module():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "illumination_verifier", TASK / "tests/verifier.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _support_module():
    spec = importlib.util.spec_from_file_location(
        "illumination_verifier_support", TASK / "tests/verifier_support.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child_case(tmp_path: Path) -> tuple[Path, Path]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "tests/input.json", app / "input.json")
    module = _module()
    result = _result(module)
    payload = {
        "schema_version": "1",
        "task_id": module.TASK_ID,
        "result": result,
        "limitations": module.LIMITATIONS,
    }
    evidence = app / "evidence/answer.json"
    evidence.parent.mkdir()
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "task_id": module.TASK_ID,
        "conclusion": "WEAK_ILLUMINATION_IS_UNSOUND_AND_REPAIRED",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "cube-illumination-strictness-audit-v1",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/answer.json",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "limitations": module.LIMITATIONS,
    }
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    return app, logs


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
    flawed = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    repair = [[-x for x in v] for v in VERTICES]
    pairs = [
        {"vertex_index": i, "direction_index": j}
        for i, v in enumerate(VERTICES)
        for j, d in enumerate(flawed)
        if module._weak(v, d) and not module._strict(v, d)
    ]
    return {
        "flawed_directions": flawed,
        "weak_false_positive_pairs": pairs,
        "repair_directions": repair,
        "vertex_to_direction": list(range(8)),
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_zero_vector_shortcut():
    module = _module()
    result = _result(module)
    result["flawed_directions"][0] = [0, 0, 0]
    assert not module.mathematics(result)


def test_rejects_missing_false_positive():
    module = _module()
    result = _result(module)
    result["weak_false_positive_pairs"] = result["weak_false_positive_pairs"][:-1]
    assert not module.mathematics(result)


def test_rejects_unhashable_direction_without_crashing():
    module = _module()
    result = _result(module)
    result["flawed_directions"][0] = [[-1], -1, 0]
    assert not module.mathematics(result)


def test_rejects_non_integer_false_positive_indices():
    module = _module()
    for replacement in (False, 0.0):
        result = _result(module)
        result["weak_false_positive_pairs"][0]["vertex_index"] = replacement
        assert not module.mathematics(result)


def test_evidence_result_comparison_preserves_json_types():
    module = _module()
    result = _result(module)
    evidence_result = _result(module)
    evidence_result["vertex_to_direction"][0] = False
    assert not module._json_equal(evidence_result, result)

    evidence_result = _result(module)
    evidence_result["weak_false_positive_pairs"][0]["direction_index"] = 0.0
    assert not module._json_equal(evidence_result, result)


def test_streaming_evidence_accepts_padding_but_preserves_token_boundaries() -> None:
    module = _support_module()
    padding = b" " * (2 * 1024 * 1024)
    payload = b'{"schema_version":' + padding + b'"1","task_id":"x"}'

    assert module._read_streaming_json_value(io.BytesIO(payload)) == {
        "schema_version": "1",
        "task_id": "x",
    }
    with pytest.raises(json.JSONDecodeError):
        module._read_streaming_json_value(io.BytesIO(b'{"value":- 1}'))
    with pytest.raises(ValueError, match="invalid JSON constant"):
        module._read_streaming_json_value(io.BytesIO(b'{"value":NaN}'))
    with pytest.raises(ValueError, match="out-of-range JSON number"):
        module._read_streaming_json_value(io.BytesIO(b'{"value":1e400}'))


def test_evidence_copied_fields_bind_to_raw_submission():
    module = _module()
    raw = {
        "task_id": module.TASK_ID,
        "result": _result(module),
        "limitations": module.LIMITATIONS,
    }
    payload = {
        "schema_version": "1",
        "task_id": module.TASK_ID,
        "result": _result(module),
        "limitations": module.LIMITATIONS,
    }
    assert module._evidence_payload_matches_submission(payload, raw)

    for field, replacement in (
        ("task_id", "jacobian/a-different-task"),
        ("limitations", ["A_DIFFERENT_LIMITATION"]),
    ):
        changed_raw = dict(raw)
        changed_raw[field] = replacement
        assert not module._evidence_payload_matches_submission(payload, changed_raw)

    omitted_raw = dict(raw)
    omitted_raw.pop("result")
    null_payload = dict(payload)
    null_payload["result"] = None
    assert not module._evidence_payload_matches_submission(null_payload, omitted_raw)


def test_accepts_reordered_repair():
    module = _module()
    result = _result(module)
    result["repair_directions"].reverse()
    result["vertex_to_direction"].reverse()
    assert module.mathematics(result)


def test_raw_submission_rejects_duplicate_keys(tmp_path):
    module = _module()
    submission = tmp_path / "submission.json"
    submission.write_text('{"result":{"bad":true},"result":{"bad":false}}')

    assert module._raw(submission) is None


def test_raw_submission_rejects_overflowing_float(tmp_path):
    module = _module()
    submission = tmp_path / "submission.json"
    submission.write_text('{"result":1e400}')

    assert module._raw(submission) is None


def test_assurance_diagnostic_is_independent_of_protocol(monkeypatch, tmp_path):
    module = _module()
    raw = {
        "claimed_assurance": "CHECKED",
        "scope": "wrong-scope",
        "result": _result(module),
    }
    monkeypatch.setattr(module, "_raw", lambda: raw)
    monkeypatch.setattr(module, "load_submission", lambda **_kwargs: object())
    monkeypatch.setattr(
        module, "strict_submission_contract", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(module, "workspace_input_is_bound", lambda: True)
    monkeypatch.setattr(
        module, "evidence_list_is_bound", lambda *_args, **_kwargs: False
    )
    output = tmp_path / "reward.json"
    monkeypatch.setattr(
        module, "_write", lambda values: output.write_text(json.dumps(values))
    )

    module.main()

    values = json.loads(output.read_text())
    assert values["protocol"] == 0.0
    assert values["assurance"] == 1.0
    assert values["reward"] == 0.0


def test_unhashable_assurance_only_fails_assurance(monkeypatch, tmp_path):
    module = _module()
    raw = {"claimed_assurance": [], "result": _result(module)}
    monkeypatch.setattr(module, "_raw", lambda: raw)
    monkeypatch.setattr(module, "load_submission", lambda **_kwargs: object())
    monkeypatch.setattr(
        module, "strict_submission_contract", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(module, "workspace_input_is_bound", lambda: True)
    monkeypatch.setattr(
        module, "evidence_list_is_bound", lambda *_args, **_kwargs: False
    )
    output = tmp_path / "reward.json"
    monkeypatch.setattr(
        module, "_write", lambda values: output.write_text(json.dumps(values))
    )

    module.main()

    values = json.loads(output.read_text())
    assert values["mathematics"] == 1.0
    assert values["assurance"] == values["protocol"] == values["reward"] == 0.0


def test_task_exports_visible_input_to_separate_verifier():
    task_toml = (TASK / "task.toml").read_text()
    assert (
        'artifacts=["/app/submission.json","/app/evidence","/app/input.json"]'
        in task_toml
    )


def test_tampered_declared_input_preserves_correctness_but_gates_reward(tmp_path):
    app, logs = _child_case(tmp_path)
    (app / "input.json").write_text("{}\n")

    reward = run_verifier_in_child(task=TASK, app=app, logs=logs)

    assert reward.details["input_binding"] == 0.0
    assert reward.details["correctness"] == 1.0
    assert reward.details["aggregate_reward"] == 0.0
