"""Focused tests for the canonical public task contract model, renderer, and
non-mutating check CLI.

Tests use temporary fixture tasks only; no committed task bundle is read,
written, or fanned out.  The tests exercise:

* strict Pydantic validation (extra=forbid, ordering, ceiling-in-allowed,
  verification-record-when-VERIFIED, forbidden field rejection);
* deterministic, idempotent rendering of ``submission_schema.json`` and the
  ``instruction.md`` Submission block;
* the non-mutating ``check`` CLI detecting drift and the ``sync`` CLI writing
  idempotently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.public_contract import (
    SCHEMA_VERSION,
    ContractError,
    PublicContract,
    check,
    load_contract,
    render_instruction,
    render_submission_block,
    render_submission_schema,
    sync,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _base_contract_dict() -> dict:
    return {
        "schema_version": "1",
        "task_id": "jacobian/test-fixture-task",
        "submission_path": "/app/submission.json",
        "assurance_ceiling": "COMPUTED",
        "allowed_assurance": ["UNVERIFIED", "COMPUTED"],
        "allowed_completeness": ["COMPLETE"],
        "conclusion": {"const": "FALSE"},
        "scope": {
            "type": "string",
            "const": "jacobian/test-fixture-task complete finite input",
        },
        "evidence": {
            "min_items": 1,
            "max_items": 1,
            "allowed_paths": ["evidence/answer.txt"],
            "media_types": ["text/plain"],
        },
        "required_artifact_filenames": ["evidence/answer.txt"],
        "public_notes": "Find a counterexample and document it.",
        "submission_result": {
            "type": "object",
            "additionalProperties": False,
            "required": ["witness"],
            "properties": {
                "witness": {"type": "integer", "minimum": 0},
            },
        },
    }


def _verified_contract_dict() -> dict:
    return {
        "schema_version": "1",
        "task_id": "jacobian/test-verified-fixture",
        "submission_path": "/app/submission.json",
        "assurance_ceiling": "VERIFIED",
        "allowed_assurance": ["UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"],
        "allowed_completeness": ["COMPLETE"],
        "conclusion": {"const": "TRUE"},
        "scope": {
            "type": "string",
            "const": "jacobian/test-verified-fixture complete finite input",
        },
        "evidence": {
            "min_items": 1,
            "max_items": 1,
            "allowed_paths": ["evidence/answer.txt"],
            "media_types": ["text/plain"],
        },
        "verification_record": {
            "path": "evidence/verification-record.json",
            "schema_ref": "environment/verification_record_schema.json",
        },
        "required_artifact_filenames": [
            "evidence/answer.txt",
            "evidence/verification-record.json",
        ],
        "public_notes": "Prove the claim and bind the verification record.",
        "submission_result": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {
                "value": {"type": "integer"},
            },
        },
    }


def _write_contract(tmp_path: Path, data: dict) -> Path:
    declared = PublicContract.model_validate(data)
    data = data | {"submission_schema": json.loads(render_submission_schema(declared))}
    path = tmp_path / "public_contract.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_task_dir(tmp_path: Path, name: str = "task") -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True, exist_ok=True)
    return task


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_base_contract_loads(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        assert contract.schema_version == SCHEMA_VERSION
        assert contract.task_id == "jacobian/test-fixture-task"
        assert contract.verification_record is None

    def test_extra_field_rejected(self) -> None:
        data = _base_contract_dict() | {"unexpected": True}
        with pytest.raises(ValueError, match="extra"):
            PublicContract.model_validate(data)

    def test_schema_version_pinned(self) -> None:
        data = _base_contract_dict() | {"schema_version": "2"}
        with pytest.raises(ValueError, match="schema_version"):
            PublicContract.model_validate(data)

    def test_assurance_ordering_enforced(self) -> None:
        data = _base_contract_dict() | {
            "allowed_assurance": ["COMPUTED", "UNVERIFIED"],
        }
        with pytest.raises(ValueError, match="ordered"):
            PublicContract.model_validate(data)

    def test_ceiling_must_be_in_allowed(self) -> None:
        data = _base_contract_dict() | {
            "assurance_ceiling": "VERIFIED",
            "allowed_assurance": ["UNVERIFIED", "COMPUTED"],
        }
        with pytest.raises(ValueError, match="assurance_ceiling must appear"):
            PublicContract.model_validate(data)

    def test_verified_requires_verification_record(self) -> None:
        data = _verified_contract_dict()
        del data["verification_record"]
        with pytest.raises(ValueError, match="verification_record is required"):
            PublicContract.model_validate(data)

    def test_conclusion_must_be_const_or_enum(self) -> None:
        data = _base_contract_dict() | {"conclusion": {"type": "string"}}
        with pytest.raises(ValueError, match="conclusion"):
            PublicContract.model_validate(data)

    def test_conclusion_const_and_enum_mutually_exclusive(self) -> None:
        data = _base_contract_dict() | {
            "conclusion": {"const": "TRUE", "enum": ["TRUE", "FALSE"]},
        }
        with pytest.raises(ValueError, match="conclusion"):
            PublicContract.model_validate(data)

    def test_scope_at_most_one_constraint(self) -> None:
        data = _base_contract_dict() | {
            "scope": {"type": "string", "const": "x", "pattern": "y"},
        }
        with pytest.raises(ValueError, match="at most one"):
            PublicContract.model_validate(data)

    def test_evidence_max_ge_min(self) -> None:
        data = _base_contract_dict() | {
            "evidence": {
                "min_items": 3,
                "max_items": 1,
                "allowed_paths": ["evidence/answer.txt"],
                "media_types": ["text/plain"],
            },
        }
        with pytest.raises(ValueError, match="max_items"):
            PublicContract.model_validate(data)

    def test_evidence_path_must_be_in_artifacts(self) -> None:
        data = _base_contract_dict() | {
            "evidence": {
                "min_items": 1,
                "max_items": 1,
                "allowed_paths": ["evidence/other.txt"],
                "media_types": ["text/plain"],
            },
        }
        with pytest.raises(ValueError, match="not in required_artifact_filenames"):
            PublicContract.model_validate(data)

    def test_artifact_filename_rejects_absolute(self) -> None:
        data = _base_contract_dict() | {
            "required_artifact_filenames": ["/app/evidence/answer.txt"],
        }
        with pytest.raises(ValueError, match="relative"):
            PublicContract.model_validate(data)

    def test_artifact_filename_rejects_traversal(self) -> None:
        data = _base_contract_dict() | {
            "required_artifact_filenames": ["../evidence/answer.txt"],
        }
        with pytest.raises(ValueError, match=r"\.\."):
            PublicContract.model_validate(data)

    def test_domain_owned_solution_field_is_allowed(self) -> None:
        data = _base_contract_dict() | {
            "submission_result": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        }
        PublicContract.model_validate(data)

    def test_domain_owned_expected_field_is_allowed(self) -> None:
        data = _base_contract_dict() | {
            "submission_result": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cases"],
                "properties": {
                    "cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["expected"],
                            "properties": {
                                "expected": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        }
        PublicContract.model_validate(data)

    def test_domain_owned_evidence_payload_field_is_allowed(self) -> None:
        data = _base_contract_dict() | {
            "evidence": {
                "min_items": 1,
                "max_items": 1,
                "allowed_paths": ["evidence/answer.txt"],
                "media_types": ["application/json"],
                "payload_shape": {
                    "solution": {"type": "string"},
                },
            },
        }
        PublicContract.model_validate(data)

    def test_envelope_field_not_forbidden(self) -> None:
        """``verification_record_uri`` is an envelope descriptor, not hidden."""
        data = _verified_contract_dict()
        contract = PublicContract.model_validate(data)
        assert contract.verification_record is not None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_submission_schema_is_valid_json(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        text = render_submission_schema(contract)
        schema = json.loads(text)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["properties"]["task_id"]["const"] == "jacobian/test-fixture-task"
        assert schema["properties"]["claimed_assurance"]["enum"] == [
            "UNVERIFIED",
            "COMPUTED",
            "CHECKED",
            "VERIFIED",
        ]
        assert schema["properties"]["completeness"]["const"] == "COMPLETE"
        assert "verification_record_uri" not in schema["properties"]
        assert "if" not in schema

    def test_submission_schema_includes_verification_record_when_verified(self) -> None:
        contract = PublicContract.model_validate(_verified_contract_dict())
        schema = json.loads(render_submission_schema(contract))
        assert "verification_record_uri" in schema["properties"]
        assert (
            schema["properties"]["verification_record_uri"]["properties"]["path"][
                "const"
            ]
            == "evidence/verification-record.json"
        )
        assert schema["if"]["properties"]["claimed_assurance"]["const"] == "VERIFIED"
        assert schema["then"]["required"] == ["verification_record_uri"]

    def test_submission_schema_evidence_single_path_renders_const(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        schema = json.loads(render_submission_schema(contract))
        item = schema["properties"]["evidence"]["items"]
        assert item["properties"]["path"]["const"] == "evidence/answer.txt"
        assert item["properties"]["sha256"]["pattern"] == r"^sha256:[0-9a-f]{64}$"

    def test_submission_schema_evidence_multi_path_renders_enum(self) -> None:
        data = _base_contract_dict() | {
            "evidence": {
                "min_items": 1,
                "max_items": 2,
                "allowed_paths": ["evidence/answer.txt", "evidence/extra.txt"],
                "media_types": ["text/plain"],
            },
            "required_artifact_filenames": [
                "evidence/answer.txt",
                "evidence/extra.txt",
            ],
        }
        contract = PublicContract.model_validate(data)
        schema = json.loads(render_submission_schema(contract))
        item = schema["properties"]["evidence"]["items"]
        assert item["properties"]["path"]["enum"] == [
            "evidence/answer.txt",
            "evidence/extra.txt",
        ]

    def test_submission_schema_multi_completeness_renders_enum(self) -> None:
        data = _base_contract_dict() | {
            "allowed_completeness": ["COMPLETE", "PARTIAL", "UNKNOWN"],
        }
        contract = PublicContract.model_validate(data)
        schema = json.loads(render_submission_schema(contract))
        assert schema["properties"]["completeness"]["enum"] == [
            "COMPLETE",
            "PARTIAL",
            "UNKNOWN",
        ]

    def test_submission_schema_deterministic(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        assert render_submission_schema(contract) == render_submission_schema(contract)

    def test_submission_block_has_markers(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        block = render_submission_block(contract)
        assert "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->" in block
        assert "<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->" in block
        assert "## Submission" in block
        assert "**Conclusion:**" in block

    def test_render_instruction_replaces_existing_block(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        existing = (
            "# Title\n\nSome prose.\n\n"
            "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->\n"
            "## Submission\n\nOLD CONTENT\n"
            "<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->\n\n"
            "More prose.\n"
        )
        result = render_instruction(contract, existing)
        assert "OLD CONTENT" not in result
        assert "# Title" in result
        assert "More prose." in result
        assert "**Conclusion:**" in result

    def test_render_instruction_appends_when_no_block(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        existing = "# Title\n\nSome prose without a block.\n"
        result = render_instruction(contract, existing)
        assert "# Title" in result
        assert "Some prose without a block." in result
        assert "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->" in result

    def test_render_instruction_creates_file_when_none(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        result = render_instruction(contract, None)
        assert result.startswith("<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->")
        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# Sync / check CLI
# ---------------------------------------------------------------------------


class TestSyncCheck:
    def test_sync_writes_both_files(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        written = sync(contract_path, task_dir)
        assert (task_dir / "environment" / "submission_schema.json").exists()
        assert (task_dir / "instruction.md").exists()
        assert len(written) == 2

    def test_sync_is_idempotent(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        written = sync(contract_path, task_dir)
        assert written == []

    def test_check_passes_after_sync(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        assert check(contract_path, task_dir) == []

    def test_check_detects_schema_drift(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        schema_path = task_dir / "environment" / "submission_schema.json"
        schema_path.write_text('{"different": true}\n', encoding="utf-8")
        drifts = check(contract_path, task_dir)
        assert any("submission_schema.json" in d for d in drifts)

    def test_check_detects_instruction_drift(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        instruction_path = task_dir / "instruction.md"
        instruction_path.write_text("# Tampered\n", encoding="utf-8")
        drifts = check(contract_path, task_dir)
        assert any("instruction.md" in d for d in drifts)

    def test_check_detects_missing_files(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        drifts = check(contract_path, task_dir)
        assert len(drifts) == 2
        assert any("missing" in d for d in drifts)

    def test_check_does_not_mutate(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        schema_path = task_dir / "environment" / "submission_schema.json"
        schema_path.write_text('{"tampered": true}\n', encoding="utf-8")
        check(contract_path, task_dir)
        assert schema_path.read_text(encoding="utf-8") == '{"tampered": true}\n'

    def test_load_contract_rejects_invalid(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ContractError, match="cannot read"):
            load_contract(bad)

    def test_verified_contract_sync_roundtrip(self, tmp_path: Path) -> None:
        contract_path = _write_contract(tmp_path, _verified_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        schema = json.loads(
            (task_dir / "environment" / "submission_schema.json").read_text(
                encoding="utf-8"
            )
        )
        assert "verification_record_uri" in schema["properties"]
        assert check(contract_path, task_dir) == []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_check_exits_zero_when_clean(self, tmp_path: Path) -> None:
        from benchmarks.tooling.public_contract import _main

        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        sync(contract_path, task_dir)
        assert (
            _main(["check", "--contract", str(contract_path), "--task", str(task_dir)])
            == 0
        )

    def test_cli_check_exits_one_on_drift(self, tmp_path: Path) -> None:
        from benchmarks.tooling.public_contract import _main

        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        assert (
            _main(["check", "--contract", str(contract_path), "--task", str(task_dir)])
            == 1
        )

    def test_cli_sync_exits_zero(self, tmp_path: Path) -> None:
        from benchmarks.tooling.public_contract import _main

        contract_path = _write_contract(tmp_path, _base_contract_dict())
        task_dir = _make_task_dir(tmp_path)
        assert (
            _main(["sync", "--contract", str(contract_path), "--task", str(task_dir)])
            == 0
        )
        assert (task_dir / "environment" / "submission_schema.json").exists()
