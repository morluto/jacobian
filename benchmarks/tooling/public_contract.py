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
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
TASK_ID_PATTERN = r"^jacobian/[a-z0-9-]+$"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

SUBMISSION_BLOCK_START = "<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->"
SUBMISSION_BLOCK_END = "<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->"
_RETIRED_PUBLIC_NOTE_TERMS = re.compile(
    r"\b(?:claimed_assurance|assurance|completeness|limitations|"
    r"verification_record|RESULT_JSON|digest-bound\s+evidence|"
    r"lowest\s+terms|expected\.json|keyword[- ]scored)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pydantic models (strict, extra=forbid)
# ---------------------------------------------------------------------------


# JSON Schema document keywords. ``payload_shape`` is a map of witness-item
# field names to schema fragments, not a schema for the witness file body.
_PAYLOAD_SHAPE_SCHEMA_DOCUMENT_KEYS = frozenset(
    {
        "$defs",
        "$schema",
        "additionalProperties",
        "allOf",
        "anyOf",
        "definitions",
        "items",
        "oneOf",
        "properties",
        "required",
        "type",
    }
)


class WitnessRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_items: int = Field(ge=0)
    max_items: int = Field(ge=0)
    allowed_paths: list[str] = Field(default_factory=list)
    digest_pattern: str = Field(default=DIGEST_PATTERN)
    media_types: list[str] = Field(default_factory=list)
    payload_shape: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _max_ge_min(self) -> WitnessRule:
        if self.max_items < self.min_items:
            raise ValueError("witness.max_items must be >= min_items")
        if self.max_items == 0 and (self.allowed_paths or self.media_types):
            raise ValueError(
                "witness with max_items=0 must not declare paths or media types"
            )
        if self.max_items > 0 and (not self.allowed_paths or not self.media_types):
            raise ValueError(
                "witness with max_items>0 requires allowed_paths and media_types"
            )
        return self

    @model_validator(mode="after")
    def _payload_shape_is_field_map(self) -> WitnessRule:
        if not self.payload_shape:
            return self
        keys = set(self.payload_shape)
        if keys and keys <= _PAYLOAD_SHAPE_SCHEMA_DOCUMENT_KEYS:
            raise ValueError(
                "witness.payload_shape must map submission witness-item "
                "field names to JSON Schema fragments; it is not a schema "
                "for the witness file body. Leave payload_shape null and "
                "document file-body shape in instruction.md / public_notes, "
                "or declare extra envelope fields such as "
                "{'solution': {'type': 'string'}}."
            )
        return self

    def item_schema(self) -> dict[str, Any]:
        """Render the per-witness-item JSON Schema."""
        if not self.allowed_paths:
            raise ValueError("witness without allowed paths has no item schema")
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
                    raise ValueError(f"witness.payload_shape must not override '{key}'")
                properties[key] = fragment
                required.append(key)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }


def _no_witness_rule() -> WitnessRule:
    return WitnessRule(min_items=0, max_items=0)


class PublicContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    task_id: str = Field(min_length=1, pattern=TASK_ID_PATTERN)
    submission_path: str = Field(pattern=r"^/app/[a-z0-9._/-]+$")
    witness: WitnessRule = Field(default_factory=_no_witness_rule)
    required_witness_filenames: list[str] = Field(default_factory=list)
    public_notes: str = Field(min_length=1)
    submission_result: dict[str, Any]
    schema_definitions: dict[str, Any] = Field(default_factory=dict)
    submission_schema: dict[str, Any] = Field(default_factory=dict)

    # -- field validators --------------------------------------------------

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be '{SCHEMA_VERSION}'")
        return v

    @field_validator("required_witness_filenames")
    @classmethod
    def _artifacts(cls, v: list[str]) -> list[str]:
        for name in v:
            if name.startswith("/"):
                raise ValueError(
                    "required_witness_filenames must be relative "
                    "(e.g. 'witness/answer.txt'), not absolute"
                )
            if ".." in name:
                raise ValueError("required_witness_filenames must not contain '..'")
        return v

    @field_validator("public_notes")
    @classmethod
    def _public_notes_describe_current_protocol(cls, v: str) -> str:
        if _RETIRED_PUBLIC_NOTE_TERMS.search(v):
            raise ValueError(
                "public_notes must not describe retired generic submission "
                "envelope fields or evidence prose"
            )
        return v

    # -- model validators --------------------------------------------------

    @model_validator(mode="after")
    def _witness_paths_in_artifacts(self) -> PublicContract:
        artifacts = set(self.required_witness_filenames)
        for ev_path in self.witness.allowed_paths:
            if ev_path not in artifacts:
                raise ValueError(
                    f"witness path {ev_path!r} is not in required_witness_filenames"
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
    properties: dict[str, Any] = {"result": dict(contract.submission_result)}
    if contract.witness.max_items:
        properties["witness"] = {
            "type": "array",
            "minItems": contract.witness.min_items,
            "maxItems": contract.witness.max_items,
            "items": contract.witness.item_schema(),
        }
    required = ["result"]
    if contract.witness.min_items:
        required.append("witness")
    schema: dict[str, Any] = {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    if contract.schema_definitions:
        schema["$defs"] = contract.schema_definitions
    return schema


def render_submission_schema(contract: PublicContract) -> str:
    """Render the complete ``environment/submission_schema.json`` text."""

    return _dump_json(_declared_schema(contract))


def render_submission_block(contract: PublicContract) -> str:
    """Render the marked ``## Submission`` block for ``instruction.md``."""
    lines: list[str] = [
        "## Submission",
        "",
        contract.public_notes.strip(),
        "",
        f"Write `{contract.submission_path}` to the exact schema in "
        f"`environment/submission_schema.json`. The submission requires a typed "
        f"`result`"
        + (" and the declared `witness`." if contract.witness.min_items else "."),
        "",
    ]
    if contract.witness.max_items:
        lines.append(
            "- **Witness:** "
            f"{contract.witness.min_items}-{contract.witness.max_items} item(s); "
            "allowed path(s): "
            + ", ".join(f"`{p}`" for p in contract.witness.allowed_paths)
            + f"; digest must match `{contract.witness.digest_pattern}`; media "
            "type(s): "
            + ", ".join(f"`{m}`" for m in contract.witness.media_types)
            + "."
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
