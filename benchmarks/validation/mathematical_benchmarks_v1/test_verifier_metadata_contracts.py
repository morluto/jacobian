"""Fail-closed parsing tests for task-local verifier metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _metadata


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": "1", "input_binding_decoupled": "yes"}, "boolean"),
        ({"schema_version": "2"}, "schema_version"),
        ({"schema_version": "1", "unknown": True}, "unknown"),
    ],
)
def test_task_contract_metadata_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    tests = tmp_path / "sample" / "tests"
    tests.mkdir(parents=True)
    _fixtures._write_json(tests / "verifier_contract.json", payload)
    monkeypatch.setattr(_metadata, "TASKS", tmp_path)

    with pytest.raises(ValueError, match=message):
        _metadata.load_task_contract_metadata("sample")
