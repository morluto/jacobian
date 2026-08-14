"""Compare cold, indexed, cached, and persistent Lean declaration lookup.

Run one pyperf cell at a time so worker processes inherit the exact cell::

    JACOBIAN_LEAN_DECL_BENCH_ENVIRONMENT=MATHLIB \
    JACOBIAN_LEAN_DECL_BENCH_BACKEND=warm \
    JACOBIAN_LEAN_DECL_BENCH_OPERATION=inspect \
      uv run --locked python -m benchmarks.tooling.lean_declaration_backend_benchmark \
        --inherit-environ \
          JACOBIAN_LEAN_DECL_BENCH_ENVIRONMENT,JACOBIAN_LEAN_DECL_BENCH_BACKEND,JACOBIAN_LEAN_DECL_BENCH_OPERATION \
        --loops 1 --values 3 --processes 1 --warmups 1 -o /tmp/result.json

Every measured result is checked against the same typed declaration contract.
The persistent cell is an evaluation-only candidate and does not change the
agent-facing atomic operation surface or the production backend selection.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, cast

import pyperf  # type: ignore[import-untyped]

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    git_head_sha,
    git_tracked_worktree_is_clean,
)
from jacobian.canonical import canonicalize_json  # noqa: E402
from jacobian.contracts.lean import LeanEnvironment  # noqa: E402
from jacobian.contracts.operations import (  # noqa: E402
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.lean_frontend.declaration_protocol import (  # noqa: E402
    LeanDeclarationInspectPayload,
    LeanDeclarationInspectQuery,
    LeanDeclarationPayload,
    LeanDeclarationQuery,
    LeanDeclarationSearchPayload,
    LeanDeclarationSearchQuery,
)
from jacobian.lean_frontend.declarations import (  # noqa: E402
    LeanSubprocessDeclarationBackend,
)
from jacobian.providers.lean_runtime import lean_provider_runtime  # noqa: E402
from jacobian_checkers import lean4  # noqa: E402

_BACKENDS = ("cold", "indexed", "warm", "persistent")
_OPERATIONS = ("search", "inspect")
type BackendName = Literal["cold", "indexed", "warm", "persistent"]
type OperationName = Literal["search", "inspect"]


def _source_sha() -> str:
    if not git_tracked_worktree_is_clean(_ROOT):
        raise SystemExit(
            "Lean declaration benchmark requires a clean tracked worktree; "
            "commit or stash source changes before running"
        )
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise SystemExit(
            "cannot bind Lean declaration benchmark to the source revision"
        )
    return cast(str, revision)


def _setting(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"{name} is required")
    return value


def _runtime() -> tuple[Path, Path, ProviderObservation]:
    lean_executable, mathlib_runtime = lean4.inspect_runtime(require_mathlib=True)
    profiles: dict[str, dict[str, Any]] = {
        LeanEnvironment.CORE.value: {
            "lean_version": lean4.LEAN_VERSION,
            "lean_commit": lean4.LEAN_COMMIT,
            "mathlib_commit": None,
        },
        LeanEnvironment.MATHLIB.value: {
            "lean_version": lean4.LEAN_VERSION,
            "lean_commit": lean4.LEAN_COMMIT,
            "mathlib_commit": lean4.MATHLIB_COMMIT,
        },
    }
    runtime = lean_provider_runtime(profiles=profiles, checker_ids=())
    if runtime.availability is not ProviderAvailability.AVAILABLE:
        raise SystemExit("the pinned Lean/Mathlib runtime is unavailable")
    if mathlib_runtime is None:
        raise SystemExit("the pinned Mathlib project is unavailable")
    return lean_executable, mathlib_runtime, runtime


def _settings() -> tuple[LeanEnvironment, BackendName, OperationName]:
    try:
        environment = LeanEnvironment(_setting("JACOBIAN_LEAN_DECL_BENCH_ENVIRONMENT"))
    except ValueError as exc:
        raise SystemExit("benchmark environment must be CORE or MATHLIB") from exc
    backend = _setting("JACOBIAN_LEAN_DECL_BENCH_BACKEND")
    if backend not in _BACKENDS:
        raise SystemExit(f"benchmark backend must be one of {_BACKENDS}")
    operation = _setting("JACOBIAN_LEAN_DECL_BENCH_OPERATION")
    if operation not in _OPERATIONS:
        raise SystemExit(f"benchmark operation must be one of {_OPERATIONS}")
    return environment, cast(BackendName, backend), cast(OperationName, operation)


def _query(
    environment: LeanEnvironment,
    operation: OperationName,
    *,
    prime: bool = False,
) -> LeanDeclarationQuery:
    name = (
        "Nat.mul"
        if prime and environment is LeanEnvironment.CORE
        else "Nat.add"
        if environment is LeanEnvironment.CORE
        else "Nat.add_comm"
        if prime
        else "irrational_sqrt_two"
    )
    target_modules = ("Init",) if environment is LeanEnvironment.CORE else ()
    if operation == "inspect":
        return LeanDeclarationInspectQuery(
            declaration_name=name,
            target_module_prefixes=target_modules,
        )
    return LeanDeclarationSearchQuery(
        name_contains=name,
        type_constants=(),
        namespace_prefixes=(),
        target_module_prefixes=target_modules,
        kinds=(),
        limit=1,
    )


def _assert_result(
    query: LeanDeclarationQuery,
    payload: LeanDeclarationPayload,
) -> None:
    if isinstance(query, LeanDeclarationInspectQuery):
        if not isinstance(payload, LeanDeclarationInspectPayload):
            raise RuntimeError("Lean declaration benchmark returned the wrong payload")
        if payload.declaration.name != query.declaration_name:
            raise RuntimeError("Lean declaration benchmark inspected the wrong name")
        return
    if not isinstance(payload, LeanDeclarationSearchPayload):
        raise RuntimeError("Lean declaration benchmark returned the wrong payload")
    needle = query.name_contains
    if (
        needle is None
        or not payload.declarations
        or needle not in payload.declarations[0].name
    ):
        raise RuntimeError("Lean declaration benchmark search missed its target")


def _corpus_digest(
    environment: LeanEnvironment,
    backend: BackendName,
    operation: OperationName,
    query: LeanDeclarationQuery,
) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonicalize_json(
                {
                    "environment": environment.value,
                    "backend": backend,
                    "operation": operation,
                    "query": query.model_dump(mode="json"),
                    "contract": "one-atomic-typed-declaration-query",
                }
            )
        ).hexdigest()
    )


def main() -> None:
    source_sha = _source_sha()
    environment, backend, operation_name = _settings()

    lean_executable, mathlib_runtime, runtime = _runtime()
    query = _query(environment, operation_name)
    cache_directory = tempfile.TemporaryDirectory(
        prefix="jacobian-lean-declaration-benchmark-"
    )
    cache_root = Path(cache_directory.name)

    def create_backend(
        *,
        persistent: bool = False,
        result_cache: bool = True,
        portable_cache: bool = True,
    ) -> LeanSubprocessDeclarationBackend:
        return LeanSubprocessDeclarationBackend(
            lean_executable=lean_executable,
            mathlib_runtime=mathlib_runtime,
            provider_runtime=runtime,
            cache_root=cache_root if portable_cache else None,
            session_backend="persistent" if persistent else "clean",
            cache_results=result_cache,
        )

    retained: LeanSubprocessDeclarationBackend | None = None
    if backend == "indexed":
        primer = create_backend(result_cache=False)
        try:
            primed = primer.query(environment, _query(environment, "search"))
            _assert_result(_query(environment, "search"), primed.payload)
        finally:
            primer.close()
    elif backend == "warm":
        retained = create_backend()
        primed = retained.query(environment, query)
        _assert_result(query, primed.payload)
    elif backend == "persistent":
        retained = create_backend(persistent=True, result_cache=False)
        prime_query = _query(environment, operation_name, prime=True)
        primed = retained.query(environment, prime_query)
        _assert_result(prime_query, primed.payload)

    def lookup(loops: int) -> float:
        started = time.perf_counter()
        for _ in range(loops):
            current = retained
            if current is None:
                current = create_backend(
                    result_cache=False,
                    portable_cache=backend == "indexed",
                )
            try:
                result = current.query(environment, query)
                _assert_result(query, result.payload)
            finally:
                if retained is None:
                    current.close()
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
            "declaration_operation": operation_name,
            "lean_version": lean4.LEAN_VERSION,
            "lean_commit": lean4.LEAN_COMMIT,
            "mathlib_commit": (
                lean4.MATHLIB_COMMIT
                if environment is LeanEnvironment.MATHLIB
                else "none"
            ),
            "corpus_digest": _corpus_digest(
                environment,
                backend,
                operation_name,
                query,
            ),
        },
    )
    try:
        runner.bench_time_func(
            f"lean_declaration_{environment.value.casefold()}_{operation_name}_{backend}",
            lookup,
        )
    finally:
        if retained is not None:
            retained.close()
        cache_directory.cleanup()


if __name__ == "__main__":
    main()
