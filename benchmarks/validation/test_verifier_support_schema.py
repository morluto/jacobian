"""Contract-level tests for the maintained JSON Schema verifier dependency."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "templates"
    / "task"
    / "tests"
    / "verifier_support.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_vs_under_test", _TEMPLATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_vs_under_test"] = module
    spec.loader.exec_module(module)
    return module


_VS = _load_module()


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "public_contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "submission_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "minimum": 0},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        },
                    },
                    "required": ["count", "labels"],
                    "additionalProperties": False,
                },
            }
        )
    )
    return path


def test_load_public_contract_accepts_valid_draft_2020_12_schema(
    contract_path: Path,
) -> None:
    assert _VS._load_public_contract(contract_path) is not None


@pytest.mark.parametrize(
    "submission",
    [
        {"count": -1, "labels": []},
        {"count": True, "labels": []},
        {"count": 1, "labels": ["duplicate", "duplicate"]},
        {"count": 1, "labels": [], "unknown": None},
    ],
)
def test_public_submission_validation_rejects_schema_violations(
    contract_path: Path,
    submission: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _VS, "_load_public_contract", lambda: json.loads(contract_path.read_text())
    )
    assert _VS._public_submission_is_valid(submission) is False


def test_public_submission_validation_accepts_complete_object(
    contract_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _VS, "_load_public_contract", lambda: json.loads(contract_path.read_text())
    )
    assert _VS._public_submission_is_valid({"count": 1, "labels": ["one"]}) is True


def test_load_public_contract_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "public_contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "submission_schema": {"type": "not-a-json-schema-type"},
            }
        )
    )
    assert _VS._load_public_contract(path) is None


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_load_submission_rejects_nonfinite_json(
    tmp_path: Path,
    contract_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    number: str,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text(f'{{"count": {number}, "labels": []}}')
    monkeypatch.setattr(
        _VS, "_load_public_contract", lambda: json.loads(contract_path.read_text())
    )
    assert _VS.load_submission(submission, require_input_binding=False) is None


def test_load_submission_enforces_the_complete_public_schema(
    tmp_path: Path,
    contract_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text('{"count": -1, "labels": [], "unknown": true}')
    contract = json.loads(contract_path.read_text())
    monkeypatch.setattr(_VS, "_load_public_contract", lambda: contract)

    assert _VS.load_submission(submission, require_input_binding=False) is None
