"""Digest-bound Lean proof-state and transition encoding helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateArtifact


def _proof_state_command(*, statement: str, proof_prefix: tuple[str, ...]) -> str:
    proof = "\n".join(f"  {line}" for line in (*proof_prefix, "sorry"))
    return f"example : {statement} := by\n{proof}"


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _environment_imports(environment: LeanEnvironment) -> tuple[str, ...]:
    return ("Mathlib",) if environment is LeanEnvironment.MATHLIB else ("Init",)


def _environment_digest(
    environment: LeanEnvironment,
    installation: LeanCheckerInstallation,
) -> str:
    return _digest(
        {
            "environment": environment.value,
            "imports": list(_environment_imports(environment)),
            "lean_version": installation.lean_version,
            "lean_commit": installation.lean_commit,
            "mathlib_commit": installation.mathlib_commit,
        }
    )


def _source_digest(statement: str, tactic_prefix: tuple[str, ...]) -> str:
    return _digest(
        {
            "statement": statement,
            "tactic_prefix": list(tactic_prefix),
            "replay_command": _proof_state_command(
                statement=statement,
                proof_prefix=tactic_prefix,
            ),
        }
    )


def _state_digest_data(
    *,
    environment: LeanEnvironment,
    environment_digest: str,
    source_digest: str,
    statement: str,
    tactic_prefix: tuple[str, ...],
    normalized_goals: tuple[str, ...],
    completed: bool,
    imports: tuple[str, ...],
    lean_version: str,
    lean_commit: str,
    mathlib_commit: str | None,
) -> dict[str, Any]:
    return {
        "environment": environment.value,
        "environment_digest": environment_digest,
        "source_digest": source_digest,
        "statement": statement,
        "tactic_prefix": list(tactic_prefix),
        "normalized_goals": list(normalized_goals),
        "completed": completed,
        "imports": list(imports),
        "lean_version": lean_version,
        "lean_commit": lean_commit,
        "mathlib_commit": mathlib_commit,
    }


def _state_payload(
    *,
    environment: LeanEnvironment,
    environment_digest: str,
    statement: str,
    tactic_prefix: tuple[str, ...],
    normalized_goals: tuple[str, ...],
    installation: LeanCheckerInstallation,
) -> LeanProofStateArtifact:
    source_digest = _source_digest(statement, tactic_prefix)
    imports = _environment_imports(environment)
    completed = len(normalized_goals) == 0
    digest_data = _state_digest_data(
        environment=environment,
        environment_digest=environment_digest,
        source_digest=source_digest,
        statement=statement,
        tactic_prefix=tactic_prefix,
        normalized_goals=normalized_goals,
        completed=completed,
        imports=imports,
        lean_version=installation.lean_version,
        lean_commit=installation.lean_commit,
        mathlib_commit=installation.mathlib_commit,
    )
    return LeanProofStateArtifact(
        **digest_data,
        state_digest=_digest(digest_data),
    )


def _state_digest_payload(state: LeanProofStateArtifact) -> str:
    return _digest(
        _state_digest_data(
            environment=state.environment,
            environment_digest=state.environment_digest,
            source_digest=state.source_digest,
            statement=state.statement,
            tactic_prefix=state.tactic_prefix,
            normalized_goals=state.normalized_goals,
            completed=state.completed,
            imports=state.imports,
            lean_version=state.lean_version,
            lean_commit=state.lean_commit,
            mathlib_commit=state.mathlib_commit,
        )
    )


__all__ = [
    "_digest",
    "_environment_digest",
    "_environment_imports",
    "_proof_state_command",
    "_source_digest",
    "_state_digest_data",
    "_state_digest_payload",
    "_state_payload",
]
