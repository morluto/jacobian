"""Pinned Lean declaration session lifecycle and protocol behavior."""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jacobian.contracts.lean import LeanDeclarationSearchStopReason, LeanEnvironment
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.lean_frontend.declaration_protocol import (
    LeanDeclarationBackendResult,
    LeanDeclarationInspectPayload,
    LeanDeclarationInspectQuery,
    LeanDeclarationPayload,
    LeanDeclarationQuery,
    LeanDeclarationSearchPayload,
    LeanDeclarationSearchQuery,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanSubprocessDeclarationBackend,
    _parse_persistent_response,
    _parse_session_response,
    _ReusableLeanQuerySession,
    _seal_declaration_index,
)
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.providers.lean_runtime import (
    lean_portable_semantic_runtime_digest,
    lean_semantic_runtime_digest,
)

_RUNTIME = ProviderObservation(
    provider="jacobian.lean4",
    availability=ProviderAvailability.AVAILABLE,
    version="4.31.0",
    digest="sha256:" + "b" * 64,
    digest_kind=ProviderDigestKind.EXECUTABLE,
    platform="test",
    install_tier=ProviderInstallTier.T3,
    license_id="Apache-2.0",
    features=("CORE", "MATHLIB"),
    configuration={
        "profiles": {
            "CORE": {
                "lean_version": "4.31.0",
                "lean_commit": "lean-commit",
                "mathlib_commit": None,
            },
            "MATHLIB": {
                "lean_version": "4.31.0",
                "lean_commit": "lean-commit",
                "mathlib_commit": "mathlib-commit",
            },
        }
    },
)


def _search_query() -> LeanDeclarationSearchQuery:
    return LeanDeclarationSearchQuery(
        name_contains="Nat",
        type_constants=(),
        namespace_prefixes=(),
        target_module_prefixes=("Init",),
        kinds=(),
        limit=10,
    )


def _inspect_query() -> LeanDeclarationInspectQuery:
    return LeanDeclarationInspectQuery(
        declaration_name="Nat.add",
        target_module_prefixes=("Init",),
    )


def _search_payload() -> LeanDeclarationSearchPayload:
    return LeanDeclarationSearchPayload(
        operation="search",
        declarations=(),
        scanned_declarations=0,
        stop_reason=LeanDeclarationSearchStopReason.EXHAUSTED,
    )


def _inspect_payload() -> LeanDeclarationInspectPayload:
    return LeanDeclarationInspectPayload.model_validate(
        {
            "operation": "inspect",
            "declaration": {
                "name": "Nat.add",
                "type": "Nat → Nat → Nat",
                "kind": "DEFINITION",
                "namespace": "Nat",
                "docstring": None,
                "source": None,
                "match_reasons": [],
            },
        }
    )


@dataclass
class RecordingSession:
    responses: list[LeanDeclarationPayload | LeanDeclarationBackendError]
    requests: list[LeanDeclarationQuery] = field(default_factory=list)
    closed: bool = False
    active: int = 0
    max_active: int = 0
    request_started: threading.Event | None = None
    release_request: threading.Event | None = None
    after_request: Any = None
    activity_lock: threading.Lock = field(default_factory=threading.Lock)

    def request(
        self, query: LeanDeclarationQuery, *, timeout_seconds: int
    ) -> LeanDeclarationPayload:
        del timeout_seconds
        with self.activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.requests.append(query)
            if self.request_started is not None:
                self.request_started.set()
            if self.release_request is not None:
                assert self.release_request.wait(timeout=2)
            response = self.responses.pop(0)
            if isinstance(response, LeanDeclarationBackendError):
                raise response
            return response
        finally:
            if self.after_request is not None:
                self.after_request()
            with self.activity_lock:
                self.active -= 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True


class RecordingSubprocessBackend(LeanSubprocessDeclarationBackend):
    def __init__(
        self,
        *,
        lean_executable: Path,
        provider_runtime: ProviderObservation,
        sessions: list[RecordingSession],
    ) -> None:
        super().__init__(
            lean_executable=lean_executable,
            mathlib_runtime=None,
            provider_runtime=provider_runtime,
        )
        self.pending_sessions = sessions
        self.started_sessions: list[RecordingSession] = []

    def _start_session(
        self, environment: LeanEnvironment, environment_digest: str
    ) -> RecordingSession:
        del environment, environment_digest
        session = self.pending_sessions.pop(0)
        self.started_sessions.append(session)
        return session


def _recording_backend(
    tmp_path: Path, sessions: list[RecordingSession]
) -> tuple[RecordingSubprocessBackend, Path]:
    executable = tmp_path / "lean"
    executable.write_bytes(b"pinned lean executable")
    digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    runtime = _RUNTIME.model_copy(update={"digest": digest, "features": ("CORE",)})
    return RecordingSubprocessBackend(
        lean_executable=executable, provider_runtime=runtime, sessions=sessions
    ), executable


def test_environment_identity_fails_closed_if_the_lean_executable_changes(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"pinned lean executable")
    digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    runtime = _RUNTIME.model_copy(update={"digest": digest, "features": ("CORE",)})
    backend = LeanSubprocessDeclarationBackend(
        lean_executable=executable, mathlib_runtime=None, provider_runtime=runtime
    )
    assert backend.environment_digest(LeanEnvironment.CORE).startswith("sha256:")
    executable.write_bytes(b"changed lean executable")
    with pytest.raises(LeanDeclarationBackendError) as raised:
        backend.environment_digest(LeanEnvironment.CORE)
    assert raised.value.code == "LEAN_ENVIRONMENT_CHANGED"


def test_portable_semantic_digest_excludes_only_the_deployment_root() -> None:
    first: dict[str, Any] = {
        "contract": "jacobian.lean.semantic-runtime/v1",
        "executable": {"digest": "sha256:" + "a" * 64},
        "mathlib_project": {
            "root": "/opt/jacobian/release-a/lean",
            "lake_digest": "sha256:" + "b" * 64,
            "loaded_modules": [{"path": "Mathlib.olean", "digest": "content"}],
        },
    }
    second: dict[str, Any] = {
        **first,
        "mathlib_project": {
            **first["mathlib_project"],
            "root": "/srv/jacobian/release-b/lean",
        },
    }

    assert lean_semantic_runtime_digest(first) != lean_semantic_runtime_digest(second)
    assert lean_portable_semantic_runtime_digest(
        first
    ) == lean_portable_semantic_runtime_digest(second)


def test_persistent_declaration_backend_uses_pinned_adjacent_lake(
    tmp_path: Path,
) -> None:
    toolchain_bin = tmp_path / "toolchain" / "bin"
    toolchain_bin.mkdir(parents=True)
    lean = toolchain_bin / "lean"
    lake = toolchain_bin / "lake"
    lean.write_bytes(b"lean")
    lake.write_bytes(b"lake")
    runtime = tmp_path / "mathlib-runtime"
    repl = runtime / ".lake/packages/repl/.lake/build/bin/repl"
    repl.parent.mkdir(parents=True)
    repl.write_bytes(b"repl")
    backend = LeanSubprocessDeclarationBackend(
        lean_executable=lean,
        mathlib_runtime=runtime,
        provider_runtime=_RUNTIME,
        session_backend="persistent",
    )

    command, cwd = backend._persistent_command()

    assert command == [str(lake.resolve()), "env", str(repl.resolve())]
    assert cwd == runtime


def test_mathlib_declaration_environment_authorizes_manifest_checkouts(
    tmp_path: Path,
) -> None:
    lean_bin = tmp_path / "toolchain" / "bin"
    lean_bin.mkdir(parents=True)
    lean = lean_bin / "lean"
    lake = lean_bin / "lake"
    lean.write_bytes(b"pinned lean executable")
    lake.write_bytes(b"pinned lake executable")
    runtime = tmp_path / "runtime"
    package_root = runtime / ".lake" / "packages"
    (package_root / "mathlib").mkdir(parents=True)
    (package_root / "batteries").mkdir()
    (runtime / "lake-manifest.json").write_text(
        json.dumps(
            {
                "packagesDir": ".lake/packages",
                "packages": [
                    {"name": "mathlib", "type": "git"},
                    {"name": "batteries", "type": "git"},
                ],
            }
        ),
        encoding="utf-8",
    )
    digest = "sha256:" + hashlib.sha256(lean.read_bytes()).hexdigest()
    backend = LeanSubprocessDeclarationBackend(
        lean_executable=lean,
        mathlib_runtime=runtime,
        provider_runtime=_RUNTIME.model_copy(update={"digest": digest}),
    )

    environment = backend._process_environment(LeanEnvironment.MATHLIB, tmp_path)

    assert environment["GIT_CONFIG_COUNT"] == "2"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert environment["GIT_CONFIG_VALUE_0"] == str(package_root / "mathlib")
    assert environment["GIT_CONFIG_KEY_1"] == "safe.directory"
    assert environment["GIT_CONFIG_VALUE_1"] == str(package_root / "batteries")


def test_subprocess_backend_reuses_one_pinned_session_per_environment(
    tmp_path: Path,
) -> None:
    session = RecordingSession(
        responses=[
            _search_payload(),
            _inspect_payload(),
        ]
    )
    backend, _ = _recording_backend(tmp_path, [session])
    searched = backend.query(LeanEnvironment.CORE, _search_query())
    inspected = backend.query(LeanEnvironment.CORE, _inspect_query())
    assert len(backend.started_sessions) == 1
    assert [request.operation for request in session.requests] == [
        "search",
        "inspect",
    ]
    assert searched.environment_digest == inspected.environment_digest
    backend.close()
    assert session.closed


def test_subprocess_backend_finalizer_closes_sessions_after_garbage_collection(
    tmp_path: Path,
) -> None:
    session = RecordingSession(responses=[_search_payload()])
    backend, _ = _recording_backend(tmp_path, [session])
    backend.query(LeanEnvironment.CORE, _search_query())
    sessions = backend._sessions
    reference = weakref.ref(backend)

    del backend
    gc.collect()

    assert reference() is None
    assert session.closed
    assert not sessions


def test_subprocess_backend_close_detaches_finalizer(tmp_path: Path) -> None:
    session = RecordingSession(responses=[_search_payload()])
    backend, _ = _recording_backend(tmp_path, [session])
    backend.query(LeanEnvironment.CORE, _search_query())

    backend.close()

    assert session.closed
    assert not backend._finalizer.alive


def test_subprocess_backend_serializes_concurrent_session_requests(
    tmp_path: Path,
) -> None:
    request_started = threading.Event()
    release_request = threading.Event()
    session = RecordingSession(
        responses=[
            _search_payload(),
            _search_payload(),
        ],
        request_started=request_started,
        release_request=release_request,
    )
    backend, _ = _recording_backend(tmp_path, [session])
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(backend.query, LeanEnvironment.CORE, _search_query())
        assert request_started.wait(timeout=1)
        second_started = threading.Event()

        def query_second() -> LeanDeclarationBackendResult:
            second_started.set()
            return backend.query(LeanEnvironment.CORE, _search_query())

        second = executor.submit(query_second)
        assert second_started.wait(timeout=1)
        release_request.set()
        results = (first.result(timeout=2), second.result(timeout=2))
    assert len(results) == 2
    assert len(backend.started_sessions) == 1
    assert session.max_active == 1
    backend.close()


def test_subprocess_backend_rejects_result_if_environment_changes_during_query(
    tmp_path: Path,
) -> None:
    session = RecordingSession(responses=[_search_payload()])
    backend, executable = _recording_backend(tmp_path, [session])
    session.after_request = lambda: executable.write_bytes(b"changed executable")
    with pytest.raises(LeanDeclarationBackendError) as raised:
        backend.query(LeanEnvironment.CORE, _search_query())
    assert raised.value.code == "LEAN_ENVIRONMENT_CHANGED"
    assert session.closed


def test_subprocess_backend_discards_timed_out_session_before_retry(
    tmp_path: Path,
) -> None:
    timed_out = RecordingSession(
        responses=[
            LeanDeclarationBackendError(
                "LEAN_QUERY_TIMEOUT", "Lean declaration discovery timed out."
            )
        ]
    )
    replacement = RecordingSession(responses=[_search_payload()])
    backend, _ = _recording_backend(tmp_path, [timed_out, replacement])
    with pytest.raises(LeanDeclarationBackendError) as raised:
        backend.query(LeanEnvironment.CORE, _search_query())
    result = backend.query(LeanEnvironment.CORE, _search_query())
    assert raised.value.code == "LEAN_QUERY_TIMEOUT"
    assert timed_out.closed
    assert len(backend.started_sessions) == 2
    assert result.payload.operation == "search"
    backend.close()


def test_session_protocol_rejects_mismatched_request_identity() -> None:
    line = 'JACOBIAN_DECLARATION_RESULT {"request_id":"stale","payload":{"operation":"search"}}'
    with pytest.raises(LeanDeclarationBackendError) as raised:
        _parse_session_response(line, expected_request_id="current")
    assert raised.value.code == "LEAN_QUERY_PROTOCOL_ERROR"


def test_session_protocol_preserves_exact_missing_declaration_failure() -> None:
    line = 'JACOBIAN_DECLARATION_ERROR {"request_id":"current","code":"LEAN_DECLARATION_NOT_FOUND","message":"declaration not found: Missing.name"}'
    with pytest.raises(LeanDeclarationBackendError) as raised:
        _parse_session_response(line, expected_request_id="current")
    assert raised.value.code == "LEAN_DECLARATION_NOT_FOUND"
    assert raised.value.message == "Lean did not find the exact requested declaration."


def _clean_session(
    tmp_path: Path,
    *,
    cache_results: bool = True,
    index_cache_path: Path | None = None,
) -> _ReusableLeanQuerySession:
    return _ReusableLeanQuerySession(
        command=[str(tmp_path / "lean")],
        cwd=tmp_path,
        process_environment={},
        source="import Init.Prelude",
        memory_mb="1024",
        isolated_home=True,
        environment_digest="sha256:" + "c" * 64,
        index_cache_path=index_cache_path,
        cache_results=cache_results,
    )


def _inspect_process_result(request_id: str) -> ProcessResult:
    payload = {
        "request_id": request_id,
        "payload": _inspect_payload().model_dump(mode="json"),
    }
    return ProcessResult(
        termination=ProcessTermination.EXITED,
        returncode=0,
        stdout=(
            "JACOBIAN_DECLARATION_RESULT "
            + json.dumps(payload, separators=(",", ":"))
            + "\n"
        ).encode(),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
    )


def test_clean_session_reuses_exact_typed_metadata_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def execute(request: Any) -> ProcessResult:
        nonlocal calls
        calls += 1
        query = json.loads(
            Path(request.environment["JACOBIAN_LEAN_QUERY_FILE"]).read_text()
        )
        return _inspect_process_result(query["request_id"])

    monkeypatch.setattr("jacobian.lean_frontend.declarations.execute_process", execute)
    session = _clean_session(tmp_path)

    first = session.request(_inspect_query(), timeout_seconds=1)
    second = session.request(_inspect_query(), timeout_seconds=1)

    assert first is second
    assert calls == 1
    session.close()


def test_clean_session_can_disable_result_cache_for_backend_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def execute(request: Any) -> ProcessResult:
        nonlocal calls
        calls += 1
        query = json.loads(
            Path(request.environment["JACOBIAN_LEAN_QUERY_FILE"]).read_text()
        )
        return _inspect_process_result(query["request_id"])

    monkeypatch.setattr("jacobian.lean_frontend.declarations.execute_process", execute)
    session = _clean_session(tmp_path, cache_results=False)

    session.request(_inspect_query(), timeout_seconds=1)
    session.request(_inspect_query(), timeout_seconds=1)

    assert calls == 2
    session.close()


def test_clean_session_restores_portable_environment_bound_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "state" / "core.index"
    seen_queries: list[dict[str, Any]] = []

    def execute(request: Any) -> ProcessResult:
        query_path = Path(request.environment["JACOBIAN_LEAN_QUERY_FILE"])
        query = json.loads(query_path.read_text())
        seen_queries.append(query)
        index_path = Path(request.environment["JACOBIAN_LEAN_INDEX_FILE"])
        if not index_path.exists():
            index_path.write_text(
                "sha256:" + "c" * 64 + "\nNat.add\tInit.Prelude\tDEFINITION\n",
                encoding="utf-8",
            )
        payload = {
            "request_id": query["request_id"],
            "payload": _search_payload().model_dump(mode="json"),
        }
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=(
                "JACOBIAN_DECLARATION_RESULT "
                + json.dumps(payload, separators=(",", ":"))
                + "\n"
            ).encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr("jacobian.lean_frontend.declarations.execute_process", execute)
    first = _clean_session(tmp_path, index_cache_path=cache_path)
    first.request(_search_query(), timeout_seconds=1)
    first.close()
    assert cache_path.is_file()

    second = _clean_session(tmp_path, index_cache_path=cache_path)
    second.request(_search_query(), timeout_seconds=1)
    second.close()

    assert seen_queries[0]["scanned_declarations_total"] is None
    assert seen_queries[1]["candidate_names"] == ["Nat.add"]
    assert seen_queries[1]["candidate_scan_positions"] == [1]
    assert seen_queries[1]["scanned_declarations_total"] == 1


def test_clean_session_ignores_corrupt_portable_catalog(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_text(
        "sha256:" + "c" * 64 + "\nmalformed-row\n",
        encoding="utf-8",
    )

    session = _clean_session(tmp_path, index_cache_path=cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    session.close()


def test_clean_session_ignores_row_boundary_truncated_catalog(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_text(
        "sha256:" + "c" * 64 + "\nNat.add\tInit.Prelude\tDEFINITION\n",
        encoding="utf-8",
    )

    session = _clean_session(tmp_path, index_cache_path=cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    session.close()


def test_clean_session_ignores_catalog_with_mismatched_content_digest(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    environment_digest = "sha256:" + "c" * 64
    cache_path.write_text(
        environment_digest + "\nNat.add\tInit.Prelude\tDEFINITION\n",
        encoding="utf-8",
    )
    _seal_declaration_index(
        cache_path,
        environment_digest=environment_digest,
    )
    cache_path.write_bytes(cache_path.read_bytes().replace(b"Nat.add", b"Nat.mul"))

    session = _clean_session(tmp_path, index_cache_path=cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    session.close()


def test_clean_session_bounds_corrupt_cache_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"x" * 65)
    monkeypatch.setattr("jacobian.lean_frontend.declarations._MAX_INDEX_BYTES", 64)

    session = _clean_session(tmp_path, index_cache_path=cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    session.close()


def test_persistent_response_requires_one_typed_marker() -> None:
    response = {
        "env": 1,
        "messages": [
            {
                "pos": {"line": 1, "column": 0},
                "severity": "info",
                "data": (
                    "JACOBIAN_DECLARATION_RESULT "
                    + json.dumps(
                        {
                            "request_id": "current",
                            "payload": _inspect_payload().model_dump(mode="json"),
                        }
                    )
                ),
            }
        ],
    }

    payload = _parse_persistent_response(response, expected_request_id="current")

    assert payload == _inspect_payload()
