"""Focused tests for the canonical public task contract model, renderer, and
non-mutating check CLI.

Tests use temporary fixture tasks only; no committed task bundle is read,
written, or fanned out.  The tests exercise:

* strict Pydantic validation (extra=forbid and legacy-envelope field
  rejection);
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
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _base_contract_dict() -> dict:
    return {
        "schema_version": "1",
        "task_id": "jacobian/test-fixture-task",
        "submission_path": "/app/submission.json",
        "witness": {
            "min_items": 1,
            "max_items": 1,
            "allowed_paths": ["evidence/answer.txt"],
            "media_types": ["text/plain"],
        },
        "required_witness_filenames": ["evidence/answer.txt"],
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


def _write_contract(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "public_contract.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_task_dir(tmp_path: Path, name: str = "task") -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True, exist_ok=True)
    return task


def test_task_template_contract_is_current() -> None:
    task = Path("benchmarks/templates/task")
    assert check(task / "tests" / "public_contract.json", task) == []
    instruction = (task / "instruction.md").read_text()
    readme = (task / "README.md").read_text()
    assert "lowest terms" not in instruction.lower()
    assert "expected.json" not in instruction
    assert "hidden `tests/expected.json`" in readme
    assert "universal certificate union" in readme
    assert "harbor-prepare-task" in readme
    assert "harbor-validate-task" in readme


def test_checked_in_schema_accepts_current_result_and_witness_contracts() -> None:
    schema = json.loads(
        Path("benchmarks/schemas/public-contract.schema.json").read_text()
    )
    result_only = json.loads(
        Path("benchmarks/templates/task/tests/public_contract.json").read_text()
    )
    witness_contract = _base_contract_dict()
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(result_only)) == []
    assert list(validator.iter_errors(witness_contract)) == []


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_base_contract_loads(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        assert contract.schema_version == SCHEMA_VERSION
        assert contract.task_id == "jacobian/test-fixture-task"

    def test_result_only_contract_uses_no_witness_default(self) -> None:
        data = _base_contract_dict()
        data.pop("witness")
        data.pop("required_witness_filenames")
        contract = PublicContract.model_validate(data)
        assert contract.witness.min_items == 0
        assert contract.witness.max_items == 0

    def test_extra_field_rejected(self) -> None:
        data = _base_contract_dict() | {"unexpected": True}
        with pytest.raises(ValueError, match="extra"):
            PublicContract.model_validate(data)

    def test_schema_version_pinned(self) -> None:
        data = _base_contract_dict() | {"schema_version": "2"}
        with pytest.raises(ValueError, match="schema_version"):
            PublicContract.model_validate(data)

    @pytest.mark.parametrize(
        "field",
        (
            "allowed_assurance",
            "allowed_completeness",
            "assurance_ceiling",
            "conclusion",
            "scope",
            "verification_record",
            "limitations",
        ),
    )
    def test_legacy_envelope_declarations_are_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match="extra"):
            PublicContract.model_validate(_base_contract_dict() | {field: {}})

    @pytest.mark.parametrize(
        "note",
        (
            "Claim COMPUTED assurance.",
            "Include the limitations array.",
            "Write one RESULT_JSON line in digest-bound evidence.",
            "Represent every rational in lowest terms.",
            "Score the conclusion with keyword-scored prose.",
            "Compare the result to tests/expected.json.",
        ),
    )
    def test_legacy_protocol_language_is_rejected_from_public_notes(
        self, note: str
    ) -> None:
        with pytest.raises(ValueError, match="public_notes"):
            PublicContract.model_validate(
                _base_contract_dict() | {"public_notes": note}
            )

    def test_evidence_max_ge_min(self) -> None:
        data = _base_contract_dict() | {
            "witness": {
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
            "witness": {
                "min_items": 1,
                "max_items": 1,
                "allowed_paths": ["evidence/other.txt"],
                "media_types": ["text/plain"],
            },
        }
        with pytest.raises(ValueError, match="not in required_witness_filenames"):
            PublicContract.model_validate(data)

    def test_artifact_filename_rejects_absolute(self) -> None:
        data = _base_contract_dict() | {
            "required_witness_filenames": ["/app/evidence/answer.txt"],
        }
        with pytest.raises(ValueError, match="relative"):
            PublicContract.model_validate(data)

    def test_artifact_filename_rejects_traversal(self) -> None:
        data = _base_contract_dict() | {
            "required_witness_filenames": ["../evidence/answer.txt"],
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
            "witness": {
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

    def test_witness_file_body_schema_is_rejected_as_payload_shape(self) -> None:
        data = _base_contract_dict() | {
            "witness": {
                "min_items": 1,
                "max_items": 1,
                "allowed_paths": ["evidence/answer.txt"],
                "media_types": ["application/json"],
                "payload_shape": {
                    "type": "object",
                    "properties": {"schema_version": {"const": "1"}},
                    "required": ["schema_version"],
                },
            },
        }
        with pytest.raises(ValueError, match="witness-item field names"):
            PublicContract.model_validate(data)


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
        # The public protocol is a typed result plus an optional task-specific
        # witness only; generic envelope fields are not emitted.
        assert set(schema["properties"]) == {"result", "witness"}
        assert schema["required"] == ["result", "witness"]
        for forbidden in (
            "task_id",
            "conclusion",
            "claimed_assurance",
            "scope",
            "completeness",
            "evidence",
            "limitations",
            "verification_record_uri",
        ):
            assert forbidden not in schema["properties"]
        assert "if" not in schema

    def test_submission_schema_witness_single_path_renders_const(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        schema = json.loads(render_submission_schema(contract))
        item = schema["properties"]["witness"]["items"]
        assert item["properties"]["path"]["const"] == "evidence/answer.txt"
        assert item["properties"]["sha256"]["pattern"] == r"^sha256:[0-9a-f]{64}$"

    def test_submission_schema_witness_multi_path_renders_enum(self) -> None:
        data = _base_contract_dict() | {
            "witness": {
                "min_items": 1,
                "max_items": 2,
                "allowed_paths": ["evidence/answer.txt", "evidence/extra.txt"],
                "media_types": ["text/plain"],
            },
            "required_witness_filenames": [
                "evidence/answer.txt",
                "evidence/extra.txt",
            ],
        }
        contract = PublicContract.model_validate(data)
        schema = json.loads(render_submission_schema(contract))
        item = schema["properties"]["witness"]["items"]
        assert item["properties"]["path"]["enum"] == [
            "evidence/answer.txt",
            "evidence/extra.txt",
        ]

    def test_submission_schema_result_only_when_no_witness(self) -> None:
        data = _base_contract_dict() | {
            "witness": {
                "min_items": 0,
                "max_items": 0,
                "allowed_paths": [],
                "media_types": [],
            },
            "required_witness_filenames": [],
        }
        contract = PublicContract.model_validate(data)
        schema = json.loads(render_submission_schema(contract))
        assert set(schema["properties"]) == {"result"}
        assert schema["required"] == ["result"]
        assert "witness" not in schema["properties"]

    def test_submission_schema_deterministic(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        assert render_submission_schema(contract) == render_submission_schema(contract)

    def test_submission_block_has_markers(self) -> None:
        contract = PublicContract.model_validate(_base_contract_dict())
        block = render_submission_block(contract)
        assert "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->" in block
        assert "<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->" in block
        assert "## Submission" in block
        assert "`result`" in block
        assert "**Witness:**" in block
        # Generic envelope fields are not mentioned in the public protocol.
        for forbidden in (
            "Conclusion",
            "claimed_assurance",
            "completeness",
            "verification_record",
        ):
            assert forbidden not in block

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
        assert "**Witness:**" in result

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

    @pytest.mark.parametrize("payload", ("[]", "null", "1"))
    def test_load_contract_rejects_non_object_json(
        self, tmp_path: Path, payload: str
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(payload, encoding="utf-8")
        with pytest.raises(ContractError, match="top-level JSON must be an object"):
            load_contract(bad)


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
