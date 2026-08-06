"""Internal repository tooling for canonical public Harbor task contracts.

This is not a Harbor source adapter and does not define an alternate task
format. It validates one verifier-owned declaration and deterministically
projects the standard Harbor files used by ``mathematical-benchmarks-v1`` tasks.

The ``public_contract.json`` is the single repository-time source of truth for
the agent-visible public protocol of one task.  From it, this module
deterministically regenerates two files inside the task bundle:

* ``environment/submission_schema.json`` -- the complete JSON Schema that
  ``/app/submission.json`` must satisfy.
* the ``## Submission`` block inside ``instruction.md`` -- a marked, idempotent
  prose section that states the public protocol in agent-readable form.

The model is strict (``extra="forbid"`` and no scalar coercion) at the
repository tooling boundary. No compatibility versions are accepted:
``schema_version`` is pinned to ``"1"``.

This module does **not** generate evaluation-runtime material (Oracle
solutions, verifier logic, authorized records, hidden fixtures).  It owns only
the public protocol surface.

CLI
---

``python -m benchmarks.tooling.public_contract sync --contract C --task DIR``
    Write the two generated files into ``DIR`` idempotently (only when content
    differs).

``python -m benchmarks.tooling.public_contract check --contract C --task DIR``
    Non-mutating: load the contract, render the two files, and report any drift
    against the files currently on disk.  Exits non-zero on drift or violation.

The ``sync-dataset`` and ``check-dataset`` variants apply the same operation to
an explicit task selection. They intentionally refuse an implicit whole-suite
selection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"
ASSURANCE_VALUES = ("UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED")
ASSURANCE_ORDER = {value: i for i, value in enumerate(ASSURANCE_VALUES)}
COMPLETENESS_VALUES = (
    "COMPLETE",
    "PARTIAL",
    "UNKNOWN",
    "COMPLETE_FOR_DECLARED_FAMILY",
)
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
TASK_ID_PATTERN = r"^jacobian/[a-z0-9-]+$"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

SUBMISSION_BLOCK_START = "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->"
SUBMISSION_BLOCK_END = "<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->"


# ---------------------------------------------------------------------------
# Pydantic models (strict, extra=forbid)
# ---------------------------------------------------------------------------


class ScopeRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str = Field(default="string")
    const: str | None = None
    enum: list[str] | None = None
    pattern: str | None = None

    @field_validator("type")
    @classmethod
    def _type_ok(cls, v: str) -> str:
        if v not in ("string", "object"):
            raise ValueError("scope.type must be 'string' or 'object'")
        return v

    @model_validator(mode="after")
    def _exactly_one_constraint(self) -> ScopeRule:
        constraints = sum(
            1 for value in (self.const, self.enum, self.pattern) if value is not None
        )
        if constraints > 1:
            raise ValueError("scope may set at most one of const, enum, pattern")
        return self

    def to_schema(self) -> dict[str, Any]:
        """Render the scope as a JSON Schema fragment."""
        fragment: dict[str, Any] = {"type": self.type}
        if self.const is not None:
            fragment["const"] = self.const
        elif self.enum is not None:
            fragment["enum"] = list(self.enum)
        elif self.pattern is not None:
            fragment["pattern"] = self.pattern
        return fragment


class EvidenceRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_items: int = Field(ge=1)
    max_items: int = Field(ge=1)
    allowed_paths: list[str] = Field(min_length=1)
    digest_pattern: str = Field(default=DIGEST_PATTERN)
    media_types: list[str] = Field(min_length=1)
    payload_shape: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _max_ge_min(self) -> EvidenceRule:
        if self.max_items < self.min_items:
            raise ValueError("evidence.max_items must be >= min_items")
        return self

    def item_schema(self) -> dict[str, Any]:
        """Render the per-evidence-item JSON Schema."""
        properties: dict[str, Any] = {
            "path": (
                {"const": self.allowed_paths[0]}
                if len(self.allowed_paths) == 1
                else {"enum": list(self.allowed_paths)}
            ),
            "sha256": {"type": "string", "pattern": self.digest_pattern},
        }
        required = ["path", "sha256"]
        if self.payload_shape:
            for key, fragment in self.payload_shape.items():
                if key in ("path", "sha256"):
                    raise ValueError(
                        f"evidence.payload_shape must not override '{key}'"
                    )
                properties[key] = fragment
                required.append(key)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }


class VerificationRecordRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1)
    required_for_assurance: str = Field(default="VERIFIED")
    schema_ref: str | None = None

    @field_validator("required_for_assurance")
    @classmethod
    def _rfa(cls, v: str) -> str:
        if v != "VERIFIED":
            raise ValueError("required_for_assurance must be 'VERIFIED'")
        return v


class PublicContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    task_id: str = Field(min_length=1, pattern=TASK_ID_PATTERN)
    submission_path: str = Field(pattern=r"^/app/[a-z0-9._/-]+$")
    assurance_ceiling: str
    allowed_assurance: list[str] = Field(min_length=1)
    allowed_completeness: list[str] = Field(min_length=1)
    conclusion: dict[str, Any]
    scope: ScopeRule
    evidence: EvidenceRule
    verification_record: VerificationRecordRule | None = None
    required_artifact_filenames: list[str] = Field(min_length=1)
    public_notes: str = Field(min_length=1)
    submission_result: dict[str, Any]
    limitations: dict[str, Any] | None = None
    schema_definitions: dict[str, Any] = Field(default_factory=dict)
    submission_schema: dict[str, Any] = Field(default_factory=dict)

    # -- field validators --------------------------------------------------

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
        return v

    @field_validator("assurance_ceiling")
    @classmethod
    def _ceiling(cls, v: str) -> str:
        if v not in ASSURANCE_VALUES:
            raise ValueError(f"assurance_ceiling must be one of {ASSURANCE_VALUES}")
        return v

    @field_validator("allowed_assurance")
    @classmethod
    def _allowed_assurance(cls, v: list[str]) -> list[str]:
        for value in v:
            if value not in ASSURANCE_VALUES:
                raise ValueError(f"allowed_assurance contains unknown value: {value!r}")
        ordered = sorted(v, key=lambda a: ASSURANCE_ORDER[a])
        if ordered != v:
            raise ValueError(
                "allowed_assurance must be ordered UNVERIFIED<COMPUTED<CHECKED<VERIFIED"
            )
        return v

    @field_validator("allowed_completeness")
    @classmethod
    def _allowed_completeness(cls, v: list[str]) -> list[str]:
        for value in v:
            if value not in COMPLETENESS_VALUES:
                raise ValueError(
                    f"allowed_completeness contains unknown value: {value!r}"
                )
        return v

    @field_validator("conclusion")
    @classmethod
    def _conclusion(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict) or not v:
            raise ValueError("conclusion must be a non-empty object")
        keys = set(v.keys())
        if keys == {"const"}:
            return v
        if keys == {"enum"}:
            enum = v["enum"]
            if not isinstance(enum, list) or not enum:
                raise ValueError("conclusion.enum must be a non-empty list")
            return v
        raise ValueError("conclusion must have exactly one of 'const' or 'enum'")

    @field_validator("required_artifact_filenames")
    @classmethod
    def _artifacts(cls, v: list[str]) -> list[str]:
        for name in v:
            if name.startswith("/"):
                raise ValueError(
                    "required_artifact_filenames must be relative "
                    "(e.g. 'evidence/answer.txt'), not absolute"
                )
            if ".." in name:
                raise ValueError("required_artifact_filenames must not contain '..'")
        return v

    # -- model validators --------------------------------------------------

    @model_validator(mode="after")
    def _ceiling_in_allowed(self) -> PublicContract:
        if self.assurance_ceiling not in self.allowed_assurance:
            raise ValueError("assurance_ceiling must appear in allowed_assurance")
        return self

    @model_validator(mode="after")
    def _verify_record_when_verified(self) -> PublicContract:
        if "VERIFIED" in self.allowed_assurance and self.verification_record is None:
            raise ValueError("verification_record is required when VERIFIED is allowed")
        return self

    @model_validator(mode="after")
    def _evidence_paths_in_artifacts(self) -> PublicContract:
        artifacts = set(self.required_artifact_filenames)
        for ev_path in self.evidence.allowed_paths:
            if ev_path not in artifacts:
                raise ValueError(
                    f"evidence path {ev_path!r} is not in required_artifact_filenames"
                )
        if (
            self.verification_record is not None
            and self.verification_record.path not in artifacts
        ):
            raise ValueError(
                f"verification_record path "
                f"{self.verification_record.path!r} is not in "
                f"required_artifact_filenames"
            )
        return self

    @model_validator(mode="after")
    def _complete_schema_matches_declarations(self) -> PublicContract:
        if self.submission_schema and self.submission_schema != _declared_schema(self):
            raise ValueError(
                "submission_schema disagrees with declared protocol fields"
            )
        return self


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _dump_json(value: Any) -> str:
    """Deterministic JSON: 2-space indent, sorted keys, trailing newline."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _declared_schema(contract: PublicContract) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "task_id": {"const": contract.task_id},
        "conclusion": dict(contract.conclusion),
        "result": dict(contract.submission_result),
        "claimed_assurance": {"enum": list(ASSURANCE_VALUES)},
        "scope": contract.scope.to_schema(),
        "evidence": {
            "type": "array",
            "minItems": contract.evidence.min_items,
            "maxItems": contract.evidence.max_items,
            "items": contract.evidence.item_schema(),
        },
        "limitations": dict(
            contract.limitations
            if contract.limitations is not None
            else {"type": "array", "items": {"type": "string"}}
        ),
    }
    if len(contract.allowed_completeness) == 1:
        properties["completeness"] = {"const": contract.allowed_completeness[0]}
    else:
        properties["completeness"] = {"enum": list(contract.allowed_completeness)}
    if contract.verification_record is not None:
        properties["verification_record_uri"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "sha256"],
            "properties": {
                "path": {"const": contract.verification_record.path},
                "sha256": {"type": "string", "pattern": DIGEST_PATTERN},
            },
        }
    required = [
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    ]
    schema: dict[str, Any] = {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    if contract.schema_definitions:
        schema["$defs"] = contract.schema_definitions
    if contract.verification_record is not None:
        schema["if"] = {
            "properties": {
                "claimed_assurance": {"const": "VERIFIED"},
            }
        }
        schema["then"] = {"required": ["verification_record_uri"]}
    return schema


def render_submission_schema(contract: PublicContract) -> str:
    """Render the complete ``environment/submission_schema.json`` text."""

    return _dump_json(_declared_schema(contract))


def _conclusion_label(conclusion: dict[str, Any]) -> str:
    if "const" in conclusion:
        return f"exactly `{conclusion['const']}`"
    return "one of " + ", ".join(f"`{v}`" for v in conclusion["enum"])


def _scope_label(scope: ScopeRule) -> str:
    if scope.const is not None:
        return "the exact value declared in `submission_schema.json`"
    if scope.enum is not None:
        return "one of the values declared in `submission_schema.json`"
    if scope.pattern is not None:
        return f"a string matching `{scope.pattern}`"
    return f"a {scope.type} value"


def render_submission_block(contract: PublicContract) -> str:
    """Render the marked ``## Submission`` block for ``instruction.md``."""
    lines: list[str] = [
        "## Submission",
        "",
        contract.public_notes.strip(),
        "",
        f"Write `{contract.submission_path}` to the exact schema in "
        f"`environment/submission_schema.json`. The submission envelope requires "
        f"`task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, "
        f"`completeness`, `evidence`, and `limitations`.",
        "",
        f"- **Conclusion:** {_conclusion_label(contract.conclusion)}",
        "- **Assurance:** scoreable values are "
        + ", ".join(f"`{a}`" for a in contract.allowed_assurance)
        + f" (ceiling `{contract.assurance_ceiling}`); the submission schema "
        "accepts any of "
        + ", ".join(f"`{a}`" for a in ASSURANCE_VALUES)
        + " but only scoreable assurances receive credit.",
        f"- **Scope:** {_scope_label(contract.scope)}",
        "- **Completeness:** "
        + (
            f"`{contract.allowed_completeness[0]}`"
            if len(contract.allowed_completeness) == 1
            else "one of " + ", ".join(f"`{c}`" for c in contract.allowed_completeness)
        )
        + ".",
        f"- **Evidence:** {contract.evidence.min_items}"
        f"-{contract.evidence.max_items} item(s); allowed path(s): "
        + ", ".join(f"`{p}`" for p in contract.evidence.allowed_paths)
        + f"; digest must match `{contract.evidence.digest_pattern}`.",
    ]
    if contract.evidence.media_types:
        lines.append(
            "- **Evidence media types:** "
            + ", ".join(f"`{m}`" for m in contract.evidence.media_types)
            + "."
        )
    lines.append(
        "- **Required artifact filenames:** "
        + ", ".join(f"`{f}`" for f in contract.required_artifact_filenames)
        + "."
    )
    if contract.verification_record is not None:
        lines.append(
            f"- **Verification record:** write "
            f"`{contract.verification_record.path}` and bind it through "
            f"`verification_record_uri` when claiming `VERIFIED`."
        )
    block = "\n".join(lines)
    return f"{SUBMISSION_BLOCK_START}\n{block}\n{SUBMISSION_BLOCK_END}"


def render_instruction(contract: PublicContract, existing: str | None) -> str:
    """Insert or replace the Submission block in ``instruction.md``.

    If ``existing`` is None, the block becomes the whole file.  Otherwise the
    region between the markers is replaced (or the block appended if no
    markers exist yet).
    """
    block = render_submission_block(contract)
    if existing is None:
        return block + "\n"
    pattern = re.compile(
        re.escape(SUBMISSION_BLOCK_START) + r".*?" + re.escape(SUBMISSION_BLOCK_END),
        re.DOTALL,
    )
    if pattern.search(existing):
        return pattern.sub(block, existing)
    separator = "\n\n" if existing and not existing.endswith("\n") else "\n"
    return existing + separator + block + "\n"


# ---------------------------------------------------------------------------
# Load / write / check
# ---------------------------------------------------------------------------


class ContractError(ValueError):
    """A public-contract validation, rendering, or drift error."""


def load_contract(path: Path) -> PublicContract:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract {path}: {exc}") from exc
    try:
        contract = PublicContract.model_validate(raw)
        if not contract.submission_schema:
            raise ValueError("submission_schema is required")
        return contract
    except ValueError as exc:
        raise ContractError(f"contract {path} is invalid: {exc}") from exc


def _task_paths(task_dir: Path) -> tuple[Path, Path]:
    env = task_dir / "environment"
    return env / "submission_schema.json", task_dir / "instruction.md"


def sync(contract_path: Path, task_dir: Path) -> list[str]:
    """Write the two generated files idempotently. Returns written paths."""
    contract = load_contract(contract_path)
    schema_path, instruction_path = _task_paths(task_dir)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    schema_text = render_submission_schema(contract)
    if (
        not schema_path.exists()
        or schema_path.read_text(encoding="utf-8") != schema_text
    ):
        schema_path.write_text(schema_text, encoding="utf-8")
        written.append(schema_path.as_posix())
    instruction_text = render_instruction(
        contract,
        instruction_path.read_text(encoding="utf-8")
        if instruction_path.exists()
        else None,
    )
    if (
        not instruction_path.exists()
        or instruction_path.read_text(encoding="utf-8") != instruction_text
    ):
        instruction_path.write_text(instruction_text, encoding="utf-8")
        written.append(instruction_path.as_posix())
    return written


def check(contract_path: Path, task_dir: Path) -> list[str]:
    """Non-mutating drift check. Returns a list of drift descriptions."""
    contract = load_contract(contract_path)
    schema_path, instruction_path = _task_paths(task_dir)
    drifts: list[str] = []
    expected_schema = render_submission_schema(contract)
    if not schema_path.exists():
        drifts.append(f"{schema_path}: missing")
    elif schema_path.read_text(encoding="utf-8") != expected_schema:
        drifts.append(f"{schema_path}: content drift")
    expected_instruction = render_instruction(
        contract,
        instruction_path.read_text(encoding="utf-8")
        if instruction_path.exists()
        else None,
    )
    if not instruction_path.exists():
        drifts.append(f"{instruction_path}: missing")
    elif instruction_path.read_text(encoding="utf-8") != expected_instruction:
        drifts.append(f"{instruction_path}: content drift")
    return drifts


def _selected_tasks(dataset_root: Path, names: list[str]) -> list[Path]:
    if not names:
        raise ContractError("at least one task name is required")
    root = dataset_root.resolve(strict=True)
    tasks: list[Path] = []
    for name in names:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise ContractError(f"invalid task name: {name!r}")
        task = (root / name).resolve(strict=True)
        if task.parent != root:
            raise ContractError(f"task escapes dataset root: {name!r}")
        tasks.append(task)
    return tasks


def sync_dataset(dataset_root: Path, names: list[str]) -> list[str]:
    written: list[str] = []
    for task in _selected_tasks(dataset_root, names):
        written.extend(sync(task / "tests" / "public_contract.json", task))
    return written


def check_dataset(dataset_root: Path, names: list[str]) -> list[str]:
    drifts: list[str] = []
    for task in _selected_tasks(dataset_root, names):
        drifts.extend(check(task / "tests" / "public_contract.json", task))
    return drifts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_sync(args: argparse.Namespace) -> int:
    written = sync(args.contract, args.task)
    for path in written:
        print(f"wrote {path}")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    drifts = check(args.contract, args.task)
    if drifts:
        for drift in drifts:
            print(f"drift: {drift}", file=sys.stderr)
        return 1
    print("public contract: OK")
    return 0


def _run_sync_dataset(args: argparse.Namespace) -> int:
    for path in sync_dataset(args.dataset_root, args.tasks):
        print(f"wrote {path}")
    return 0


def _run_check_dataset(args: argparse.Namespace) -> int:
    drifts = check_dataset(args.dataset_root, args.tasks)
    if drifts:
        for drift in drifts:
            print(f"drift: {drift}", file=sys.stderr)
        return 1
    print("public contracts: OK")
    return 0


_COMMAND_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "sync": _run_sync,
    "check": _run_check,
    "sync-dataset": _run_sync_dataset,
    "check-dataset": _run_check_dataset,
}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Canonical public task contract sync and check.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sync_p = sub.add_parser("sync", help="Write generated files idempotently.")
    sync_p.add_argument("--contract", type=Path, required=True)
    sync_p.add_argument("--task", type=Path, required=True)
    check_p = sub.add_parser("check", help="Non-mutating drift check.")
    check_p.add_argument("--contract", type=Path, required=True)
    check_p.add_argument("--task", type=Path, required=True)
    sync_dataset_p = sub.add_parser(
        "sync-dataset", help="Write projections for explicitly selected tasks."
    )
    sync_dataset_p.add_argument("--dataset-root", type=Path, required=True)
    sync_dataset_p.add_argument("--tasks", nargs="+", required=True)
    check_dataset_p = sub.add_parser(
        "check-dataset", help="Check projections for explicitly selected tasks."
    )
    check_dataset_p.add_argument("--dataset-root", type=Path, required=True)
    check_dataset_p.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args(argv)
    handler = _COMMAND_DISPATCH.get(args.command)
    if handler is None:
        return 0
    try:
        return handler(args)
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ASSURANCE_VALUES",
    "COMPLETENESS_VALUES",
    "SCHEMA_VERSION",
    "ContractError",
    "PublicContract",
    "check",
    "check_dataset",
    "load_contract",
    "render_instruction",
    "render_submission_block",
    "render_submission_schema",
    "sync",
    "sync_dataset",
]
