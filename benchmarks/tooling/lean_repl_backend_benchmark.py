"""Compare clean and persistent Lean validated replay with pyperf.

Choose one benchmark cell with environment variables so pyperf can forward the
exact identity to worker processes::

    JACOBIAN_LEAN_BENCH_ENVIRONMENT=MATHLIB \
    JACOBIAN_LEAN_BENCH_BACKEND=clean \
    JACOBIAN_LEAN_BENCH_PREFIX_LENGTH=32 \
      uv run --locked python -m benchmarks.tooling.lean_repl_backend_benchmark \
        --inherit-environ \
          JACOBIAN_LEAN_BENCH_ENVIRONMENT,JACOBIAN_LEAN_BENCH_BACKEND,JACOBIAN_LEAN_BENCH_PREFIX_LENGTH \
        --loops 1 --values 3 --processes 1 --warmups 1 -o /tmp/result.json

The benchmark changes no agent-facing operation.  Correctness assertions run
around every measured transition; the JSON result remains evaluation evidence.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pyperf

from benchmarks.tooling.command_runner import (
    git_head_sha,
    git_tracked_worktree_is_clean,
)
from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.artifacts import _proof_state_command
from jacobian.lean_frontend.repl import LeanExplorationReplRuntime, LeanReplPolicy
from jacobian.lean_frontend.repl_protocol import LeanReplProofStepResponse

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "lean"
_BACKENDS = ("clean", "persistent")
_PREFIX_LENGTHS = (1, 8, 16, 32)


def _source_sha() -> str:
    if not git_tracked_worktree_is_clean(_ROOT):
        raise SystemExit(
            "Lean REPL benchmark requires a clean tracked worktree; commit or "
            "stash source changes before running"
        )
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise SystemExit("cannot bind Lean REPL benchmark to the source revision")
    return revision


def _setting(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"{name} is required")
    return value


def _installation(environment: LeanEnvironment) -> LeanCheckerInstallation:
    artifact_uri = "artifact://sha256/" + (
        "a" * 64 if environment is LeanEnvironment.CORE else "b" * 64
    )
    return LeanCheckerInstallation(
        environment=environment,
        lean_version="4.31.0",
        lean_commit="68218e876d2a38b1985b8590fff244a83c321783",
        import_name=(None if environment is LeanEnvironment.CORE else "Mathlib"),
        mathlib_commit=(
            None
            if environment is LeanEnvironment.CORE
            else "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
        ),
        allowed_axioms=(),
        checker_timeout_seconds=180,
        semantics_uri=artifact_uri,
        claim_schema_uri=artifact_uri,
        candidate_schema_uri=artifact_uri,
        certificate_schema_uri=artifact_uri,
        checker_id=None,
    )


def _corpus_digest(
    *,
    environment: LeanEnvironment,
    backend: str,
    prefix_length: int,
    command: str,
) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonicalize_json(
                {
                    "environment": environment.value,
                    "backend": backend,
                    "prefix_length": prefix_length,
                    "command": command,
                    "tactic": "trivial",
                    "contract": "validated-command-reconstruction-and-one-tactic",
                }
            )
        ).hexdigest()
    )


def main() -> None:
    source_sha = _source_sha()
    try:
        environment = LeanEnvironment(_setting("JACOBIAN_LEAN_BENCH_ENVIRONMENT"))
    except ValueError as exc:
        raise SystemExit("benchmark environment must be CORE or MATHLIB") from exc
    backend = _setting("JACOBIAN_LEAN_BENCH_BACKEND")
    if backend not in _BACKENDS:
        raise SystemExit("benchmark backend must be clean or persistent")
    try:
        prefix_length = int(_setting("JACOBIAN_LEAN_BENCH_PREFIX_LENGTH"))
    except ValueError as exc:
        raise SystemExit("benchmark prefix length must be an integer") from exc
    if prefix_length not in _PREFIX_LENGTHS:
        raise SystemExit(f"benchmark prefix length must be one of {_PREFIX_LENGTHS}")

    installations = {member: _installation(member) for member in LeanEnvironment}
    runtime = LeanExplorationReplRuntime(
        _RUNTIME,
        installations,
        policy=LeanReplPolicy(
            max_requests=100_000,
            max_age_seconds=3_600,
            max_rss_kb=(
                9 * 1024 * 1024
                if environment is LeanEnvironment.MATHLIB
                else 7 * 1024 * 1024
            ),
            timeout_seconds=180,
        ),
    )
    command = _proof_state_command(
        statement="True",
        proof_prefix=("skip",) * prefix_length,
    )

    def replay(loops: int) -> float:
        started = time.perf_counter()
        for _ in range(loops):
            responses = (
                runtime.execute_clean(
                    command=command,
                    tactic="trivial",
                    environment=environment,
                )
                if backend == "clean"
                else runtime.execute_persistent_validated(
                    command=command,
                    tactic="trivial",
                    environment=environment,
                )
            )
            transition = responses[2]
            if (
                not isinstance(transition, LeanReplProofStepResponse)
                or transition.proof_status != "Completed"
                or transition.goals
            ):
                raise RuntimeError("Lean benchmark transition failed correctness check")
        return time.perf_counter() - started

    runner = pyperf.Runner(
        loops=1,
        values=3,
        processes=1,
        warmups=1,
        min_time=0.1,
        metadata={
            "jacobian_source_sha": source_sha,
            "lean_environment": environment.value,
            "lean_backend": backend,
            "proof_prefix_length": prefix_length,
            "lean_version": installations[environment].lean_version,
            "lean_commit": installations[environment].lean_commit,
            "mathlib_commit": installations[environment].mathlib_commit or "none",
            "corpus_digest": _corpus_digest(
                environment=environment,
                backend=backend,
                prefix_length=prefix_length,
                command=command,
            ),
        },
    )
    try:
        runner.bench_time_func(
            f"lean_repl_{environment.value.casefold()}_{backend}_prefix_{prefix_length}",
            replay,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
