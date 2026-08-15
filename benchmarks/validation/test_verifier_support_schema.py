"""Contract-level tests for the maintained JSON Schema verifier dependency."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from benchmarks.validation._source_module import load_source_module

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "templates"
    / "task"
    / "tests"
    / "verifier_support.py"
)


_VS = load_source_module("_vs_under_test", _TEMPLATE)


@pytest.mark.parametrize("preserve_existing", [False, True])
def test_load_module_scopes_sys_modules_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preserve_existing: bool,
) -> None:
    module_name = "_verifier_support_schema_module_probe"
    source = tmp_path / "module_probe.py"
    source.write_text(
        "import sys\nregistered_while_loading = sys.modules[__name__]\n",
        encoding="utf-8",
    )
    sentinel = ModuleType("sentinel")
    if preserve_existing:
        monkeypatch.setitem(sys.modules, module_name, sentinel)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    loaded = load_source_module(module_name, source)

    assert vars(loaded)["registered_while_loading"] is loaded
    if preserve_existing:
        assert sys.modules[module_name] is sentinel
    else:
        assert module_name not in sys.modules


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


def test_aggregate_reward_full_credit_when_hard_gates_pass() -> None:
    assert _VS.aggregate_reward(correctness=1.0, witness_validity=True) == 1.0


def test_aggregate_reward_zeros_on_invalid_evidence_even_when_math_passes() -> None:
    assert _VS.aggregate_reward(correctness=1.0, witness_validity=0.0) == 0.0


def test_aggregate_reward_zeros_on_protocol_failure() -> None:
    assert (
        _VS.aggregate_reward(correctness=1.0, witness_validity=1.0, protocol_ok=False)
        == 0.0
    )


def test_aggregate_reward_is_binary_with_no_partial_credit() -> None:
    # A partial correctness diagnostic is not a full mathematical outcome.
    assert _VS.aggregate_reward(correctness=0.5, witness_validity=1.0) == 0.0
    # Booleans are accepted and treated as full/empty unit scores.
    assert _VS.aggregate_reward(correctness=True, witness_validity=True) == 1.0
    assert _VS.aggregate_reward(correctness=True, witness_validity=False) == 0.0


def test_aggregate_reward_has_no_soft_assurance_parameter() -> None:
    signature = inspect.signature(_VS.aggregate_reward)
    assert set(signature.parameters) == {
        "correctness",
        "witness_validity",
        "protocol_ok",
    }


def test_template_exports_new_protocol_surface() -> None:
    for name in (
        "aggregate_reward",
        "submission_matches_public_schema",
        "normalize_reward_file",
    ):
        assert name in _VS.__all__


def test_template_does_not_export_retired_envelope_helpers() -> None:
    retired = (
        "strict_submission_contract",
        "false_verified_claim",
        "authorized_record_is_bound",
        "ASSURANCE_LEVELS",
        "SUBMISSION_FIELDS",
    )
    for name in retired:
        assert name not in _VS.__all__
        assert not hasattr(_VS, name)


def test_submission_matches_public_schema_validates_against_contract(
    contract_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _VS, "_load_public_contract", lambda: json.loads(contract_path.read_text())
    )
    assert _VS.submission_matches_public_schema({"count": 1, "labels": ["one"]}) is True
    assert _VS.submission_matches_public_schema({"count": -1, "labels": []}) is False


def test_reject_duplicate_keys_raises_on_duplicate_object_name() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _VS._reject_duplicate_keys([("a", 1), ("a", 2)])


def test_reject_duplicate_keys_accepts_unique_object_names() -> None:
    result = _VS._reject_duplicate_keys([("a", 1), ("b", 2)])
    assert result == {"a": 1, "b": 2}


def test_reject_duplicate_keys_accepts_empty_pairs() -> None:
    assert _VS._reject_duplicate_keys([]) == {}


def test_load_submission_rejects_duplicate_keys(
    tmp_path: Path,
    contract_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text('{"count": 1, "count": 2, "labels": []}')
    monkeypatch.setattr(
        _VS, "_load_public_contract", lambda: json.loads(contract_path.read_text())
    )
    assert _VS.load_submission(submission, require_input_binding=False) is None


def test_load_submission_raw_rejects_duplicate_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text('{"a": 1, "a": 2}')
    monkeypatch.setattr(_VS, "workspace_input_is_bound", lambda: True)
    assert _VS.load_submission_raw(submission, require_input_binding=False) is None


def test_read_evidence_json_rejects_duplicate_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"a": 1, "a": 2}')
    digest = "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    descriptor = {"path": "evidence/answer.txt", "sha256": digest}
    monkeypatch.setattr(
        _VS,
        "resolve_evidence",
        lambda descriptor, **kw: evidence,
    )
    assert (
        _VS.read_evidence_json(descriptor, expected_path="evidence/answer.txt") is None
    )
