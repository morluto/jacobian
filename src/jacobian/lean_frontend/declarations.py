"""Read-only declaration discovery over pinned Lean environments."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.lean import (
    LeanDeclarationInspectOutput,
    LeanDeclarationInspectRequest,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDependencyGraphArtifact,
    LeanDependencyGraphRequest,
    LeanEnvironment,
)
from jacobian.lean_frontend.declaration_protocol import (
    LeanDeclarationBackendResult,
    LeanDeclarationDependenciesPayload,
    LeanDeclarationDependenciesQuery,
    LeanDeclarationErrorEnvelope,
    LeanDeclarationInspectPayload,
    LeanDeclarationInspectQuery,
    LeanDeclarationPayload,
    LeanDeclarationQuery,
    LeanDeclarationResultEnvelope,
    LeanDeclarationSearchPayload,
    LeanDeclarationSearchQuery,
)
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandResponse,
    LeanReplErrorResponse,
)
from jacobian.process_policy import (
    BoundedInteractiveProcess,
    InteractiveProcessError,
    InteractiveProcessRequest,
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.providers.lean_runtime import (
    LeanRuntimeIdentityError,
    lean_mathlib_git_config,
    lean_portable_semantic_runtime_digest,
    lean_semantic_runtime_digest,
    require_lean_semantic_runtime_identity,
)
from jacobian.worker_environment import worker_environment

_LOGGER = logging.getLogger(__name__)
_RESULT_PREFIX = "JACOBIAN_DECLARATION_RESULT "
_ERROR_PREFIX = "JACOBIAN_DECLARATION_ERROR "
_MAX_STDOUT_BYTES = 2 * 1024 * 1024
_MAX_STDERR_BYTES = 128 * 1024
_MAX_CATALOG_CANDIDATES = 10_000
_MAX_CATALOG_NAME_BYTES = 2 * 1024 * 1024
_MAX_INDEX_BYTES = 128 * 1024 * 1024
_RESULT_CACHE_ENTRIES = 128
_RESULT_CACHE_BYTES = 32 * 1024 * 1024
_PERSISTENT_MAX_REQUESTS = 64
_PERSISTENT_MAX_AGE_SECONDS = 600.0
_QUERY_SOURCE = Path(__file__).with_name("_lean_declaration_query.lean")
_IMPORT_TOKEN = "{{JACOBIAN_IMPORT}}"
_ENTRYPOINT_MARKER = "-- JACOBIAN_DECLARATION_ENTRYPOINT"
_INDEX_FORMAT = "jacobian.lean.declaration-index/v2"
_INDEX_FOOTER_PREFIX = "# jacobian-declaration-index"


class LeanDeclarationBackend(Protocol):
    """The process boundary needed by typed declaration discovery."""

    def environment_digest(self, environment: LeanEnvironment) -> str: ...

    def query(
        self,
        environment: LeanEnvironment,
        query: LeanDeclarationQuery,
    ) -> LeanDeclarationBackendResult: ...


class LeanDeclarationBackendError(RuntimeError):
    """A bounded backend failure safe for capability diagnostic mapping."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _TypedPayloadCache:
    """Bounded LRU of immutable, validated declaration payloads."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._entries: OrderedDict[str, tuple[LeanDeclarationPayload, int]] = (
            OrderedDict()
        )
        self._retained_bytes = 0

    def get(self, query: LeanDeclarationQuery) -> LeanDeclarationPayload | None:
        if not self._enabled or not _cacheable_query(query):
            return None
        key = _query_cache_key(query)
        cached = self._entries.get(key)
        if cached is None:
            return None
        self._entries.move_to_end(key)
        return cached[0]

    def put(
        self,
        query: LeanDeclarationQuery,
        payload: LeanDeclarationPayload,
    ) -> None:
        if not self._enabled or not _cacheable_query(query):
            return
        serialized = canonicalize_json(payload.model_dump(mode="json"))
        size = len(serialized)
        if size > _RESULT_CACHE_BYTES:
            return
        key = _query_cache_key(query)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._retained_bytes -= previous[1]
        self._entries[key] = (payload, size)
        self._retained_bytes += size
        while (
            len(self._entries) > _RESULT_CACHE_ENTRIES
            or self._retained_bytes > _RESULT_CACHE_BYTES
        ):
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self._retained_bytes -= evicted_size


class _DeclarationQuerySession(Protocol):
    def request(
        self,
        query: LeanDeclarationQuery,
        *,
        timeout_seconds: int,
    ) -> LeanDeclarationPayload: ...

    def close(self) -> None: ...


class _ReusableLeanQuerySession:
    """Run clean bounded queries with a reusable index and typed result cache."""

    def __init__(
        self,
        *,
        command: list[str],
        cwd: Path,
        process_environment: dict[str, str],
        source: str,
        memory_mb: str,
        isolated_home: bool,
        environment_digest: str,
        index_cache_path: Path | None,
        cache_results: bool,
    ) -> None:
        self._closed = False
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="jacobian-lean-declarations-"
        )
        temporary_root = Path(self._temporary_directory.name)
        self._source_path = temporary_root / "declaration_query.lean"
        self._source_path.write_text(source, encoding="utf-8")
        self._request_path = temporary_root / "query.json"
        self._index_path = temporary_root / "declarations.index"
        self._index_digest: str | None = None
        self._index_cache_path = index_cache_path
        self._payload_cache = _TypedPayloadCache(enabled=cache_results)
        self._environment_digest = environment_digest
        self._command = command
        self._cwd = cwd
        self._memory_mb = memory_mb
        self._process_environment = dict(process_environment)
        self._process_environment.update(
            {
                "JACOBIAN_LEAN_ENVIRONMENT_DIGEST": environment_digest,
                "JACOBIAN_LEAN_INDEX_FILE": str(self._index_path),
                "JACOBIAN_LEAN_QUERY_FILE": str(self._request_path),
            }
        )
        if isolated_home:
            self._process_environment["HOME"] = str(temporary_root)
        self._restore_index_cache()

    def request(
        self,
        query: LeanDeclarationQuery,
        *,
        timeout_seconds: int,
    ) -> LeanDeclarationPayload:
        if self._closed:
            raise _query_failed()
        self._check_index_identity()
        cached = self._payload_cache.get(query)
        if cached is not None:
            return cached
        request_id = uuid.uuid4().hex
        wire_query = self._catalog_query(query)
        wire_payload = {
            **wire_query.model_dump(mode="json"),
            "request_id": request_id,
        }
        try:
            self._request_path.write_bytes(canonicalize_json(wire_payload))
            result = execute_process(
                ProcessRequest(
                    executable=self._command[0],
                    arguments=(
                        *self._command[1:],
                        str(self._source_path),
                        "-t",
                        "0",
                        "-T",
                        "1000000000",
                        "-M",
                        self._memory_mb,
                        "-j",
                        "1",
                        "--trust=0",
                    ),
                    environment=self._process_environment,
                    cwd=str(self._cwd),
                    timeout_seconds=float(timeout_seconds),
                    stdin_bytes=b"",
                    stdout_limit_bytes=_MAX_STDOUT_BYTES,
                    stderr_limit_bytes=_MAX_STDERR_BYTES,
                )
            )
        except OSError as exc:
            raise _query_failed() from exc
        if result.termination is ProcessTermination.TIMED_OUT:
            raise LeanDeclarationBackendError(
                "LEAN_QUERY_TIMEOUT",
                (
                    "Lean declaration discovery exceeded the "
                    f"{timeout_seconds}-second per-query budget."
                ),
            )
        if result.termination is not ProcessTermination.EXITED:
            raise _query_failed()
        if len(result.stdout) > _MAX_STDOUT_BYTES:
            raise _structured_output_limit()
        if len(result.stderr) > _MAX_STDERR_BYTES:
            raise _diagnostic_output_limit()
        if result.returncode != 0:
            _LOGGER.warning(
                "Lean declaration query failed: %s",
                (result.stdout + result.stderr)
                .decode("utf-8", errors="replace")
                .strip(),
            )
            raise _query_failed()
        output = _parse_process_response(
            result.stdout,
            expected_request_id=request_id,
        )
        self._record_or_check_index(query)
        self._payload_cache.put(query, output)
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._temporary_directory.cleanup()

    def _check_index_identity(self) -> None:
        if self._index_digest is None:
            return
        try:
            current_digest = _sha256_file(self._index_path)
        except OSError as exc:
            raise _index_changed() from exc
        if current_digest != self._index_digest:
            raise _index_changed()

    def _record_or_check_index(self, query: LeanDeclarationQuery) -> None:
        if not isinstance(query, LeanDeclarationSearchQuery):
            return
        try:
            if self._index_digest is None:
                _seal_declaration_index(
                    self._index_path,
                    environment_digest=self._environment_digest,
                )
            current_digest = _sha256_file(self._index_path)
            _validate_declaration_index(
                self._index_path,
                environment_digest=self._environment_digest,
            )
        except (OSError, ValueError) as exc:
            raise _protocol_error() from exc
        if self._index_digest is None:
            self._index_digest = current_digest
            self._publish_index_cache()
        elif current_digest != self._index_digest:
            raise _index_changed()

    def _restore_index_cache(self) -> None:
        cache_path = self._index_cache_path
        if cache_path is None or cache_path.is_symlink() or not cache_path.is_file():
            return
        try:
            _copy_bounded_file(
                cache_path,
                self._index_path,
                max_bytes=_MAX_INDEX_BYTES,
            )
            _validate_declaration_index(
                self._index_path,
                environment_digest=self._environment_digest,
            )
            self._index_digest = _sha256_file(self._index_path)
        except (OSError, ValueError):
            self._index_path.unlink(missing_ok=True)
            self._index_digest = None

    def _publish_index_cache(self) -> None:
        cache_path = self._index_cache_path
        if cache_path is None:
            return
        temporary_path: Path | None = None
        try:
            cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{cache_path.name}.",
                dir=cache_path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with self._index_path.open("rb") as source:
                    shutil.copyfileobj(source, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.chmod(0o600)
            temporary_path.replace(cache_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _catalog_query(
        self,
        query: LeanDeclarationQuery,
    ) -> LeanDeclarationQuery:
        return _catalog_query_from_index(
            query,
            index_path=self._index_path,
            environment_digest=self._environment_digest,
            index_ready=self._index_digest is not None,
        )


class _PersistentLeanQuerySession:
    """Bounded benchmark candidate retaining one imported Lean environment."""

    def __init__(
        self,
        *,
        command: list[str],
        cwd: Path,
        process_environment: dict[str, str],
        base_command: str,
        memory_mb: str,
        environment_digest: str,
        cache_results: bool,
    ) -> None:
        self._closed = False
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="jacobian-lean-declarations-persistent-"
        )
        temporary_root = Path(self._temporary_directory.name)
        self._request_path = temporary_root / "query.json"
        self._index_path = temporary_root / "declarations.index"
        self._index_digest: str | None = None
        self._environment_digest = environment_digest
        self._command = command
        self._cwd = cwd
        self._base_command = base_command
        self._process_environment = dict(process_environment)
        self._process_environment.update(
            {
                "JACOBIAN_LEAN_ENVIRONMENT_DIGEST": environment_digest,
                "JACOBIAN_LEAN_INDEX_FILE": str(self._index_path),
                "JACOBIAN_LEAN_QUERY_FILE": str(self._request_path),
            }
        )
        self._max_rss_kb = int(memory_mb) * 1024
        self._process: BoundedInteractiveProcess | None = None
        self._base_env: int | None = None
        self._started_at = 0.0
        self._requests = 0
        self._payload_cache = _TypedPayloadCache(enabled=cache_results)

    def request(
        self,
        query: LeanDeclarationQuery,
        *,
        timeout_seconds: int,
    ) -> LeanDeclarationPayload:
        if self._closed:
            raise _query_failed()
        cached = self._payload_cache.get(query)
        if cached is not None:
            return cached
        self._check_index_identity()
        self._ensure_process(timeout_seconds)
        process = self._process
        if process is None:
            raise _query_failed()
        request_id = uuid.uuid4().hex
        wire_query = _catalog_query_from_index(
            query,
            index_path=self._index_path,
            environment_digest=self._environment_digest,
            index_ready=self._index_digest is not None,
        )
        wire_payload = {
            **wire_query.model_dump(mode="json"),
            "request_id": request_id,
        }
        self._request_path.write_bytes(canonicalize_json(wire_payload))
        try:
            decoded = process.exchange(
                {
                    "cmd": "#jacobian_declaration_query",
                    "env": self._base_env,
                },
                timeout_seconds=float(timeout_seconds),
            )
        except InteractiveProcessError as exc:
            self._stop_process()
            detail = str(exc)
            if "timed out" in detail:
                raise LeanDeclarationBackendError(
                    "LEAN_QUERY_TIMEOUT",
                    (
                        "Lean declaration discovery exceeded the "
                        f"{timeout_seconds}-second per-query budget."
                    ),
                ) from exc
            if "stderr limit" in detail or "memory limit" in detail:
                raise LeanDeclarationBackendError(
                    "LEAN_QUERY_OUTPUT_LIMIT",
                    "The persistent Lean declaration backend exceeded its resource budget.",
                ) from exc
            raise _query_failed() from exc
        output = _parse_persistent_response(
            decoded,
            expected_request_id=request_id,
        )
        self._record_or_check_index(query)
        self._requests += 1
        self._payload_cache.put(query, output)
        return output

    def _check_index_identity(self) -> None:
        if self._index_digest is None:
            return
        try:
            current_digest = _sha256_file(self._index_path)
        except OSError as exc:
            raise _index_changed() from exc
        if current_digest != self._index_digest:
            raise _index_changed()

    def _record_or_check_index(self, query: LeanDeclarationQuery) -> None:
        if not isinstance(query, LeanDeclarationSearchQuery):
            return
        try:
            if self._index_digest is None:
                _seal_declaration_index(
                    self._index_path,
                    environment_digest=self._environment_digest,
                )
            _validate_declaration_index(
                self._index_path,
                environment_digest=self._environment_digest,
            )
            current_digest = _sha256_file(self._index_path)
        except (OSError, ValueError) as exc:
            raise _protocol_error() from exc
        if self._index_digest is None:
            self._index_digest = current_digest
        elif current_digest != self._index_digest:
            raise _index_changed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_process()
        self._temporary_directory.cleanup()

    def _ensure_process(self, timeout_seconds: int) -> None:
        if self._process is not None and (
            not self._process.is_running
            or self._requests >= _PERSISTENT_MAX_REQUESTS
            or time.monotonic() - self._started_at >= _PERSISTENT_MAX_AGE_SECONDS
        ):
            self._stop_process()
        if self._process is not None:
            return
        process = BoundedInteractiveProcess(
            InteractiveProcessRequest(
                executable=self._command[0],
                arguments=tuple(self._command[1:]),
                environment=self._process_environment,
                cwd=str(self._cwd.resolve(strict=True)),
                startup_timeout_seconds=float(timeout_seconds),
                read_timeout_seconds=float(timeout_seconds),
                shutdown_timeout_seconds=2.0,
                stderr_limit_bytes=_MAX_STDERR_BYTES,
                base_command=self._base_command,
                max_rss_kb=self._max_rss_kb,
            )
        )
        try:
            process.start()
        except InteractiveProcessError as exc:
            raise _query_failed() from exc
        base_response = process.base_response
        try:
            parsed_base = LeanReplCommandResponse.model_validate(base_response)
        except ValidationError as exc:
            process.close()
            raise _protocol_error() from exc
        if any(message.severity == "error" for message in parsed_base.messages):
            process.close()
            raise _query_failed()
        self._process = process
        self._base_env = parsed_base.env
        self._started_at = time.monotonic()
        self._requests = 0

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._base_env = None
        if process is not None:
            process.close()


@dataclass(frozen=True, slots=True)
class _SessionEntry:
    environment_digest: str
    session: _DeclarationQuerySession


def _close_declaration_sessions(
    sessions: dict[LeanEnvironment, _SessionEntry],
    session_locks: dict[LeanEnvironment, threading.Lock],
) -> None:
    """Release declaration sessions retained past their backend's lifetime."""
    for environment, entry in list(sessions.items()):
        with session_locks[environment]:
            entry.session.close()
    sessions.clear()


class LeanSubprocessDeclarationBackend:
    """Reuse typed metadata across bounded, environment-bound sessions."""

    def __init__(
        self,
        *,
        lean_executable: Path,
        mathlib_runtime: Path | None,
        provider_runtime: CapabilityProviderRuntime,
        cache_root: Path | None = None,
        session_backend: Literal["clean", "persistent"] = "clean",
        cache_results: bool = True,
    ) -> None:
        if session_backend not in {"clean", "persistent"}:
            raise ValueError("declaration session backend must be clean or persistent")
        self.lean_executable = lean_executable
        self.mathlib_runtime = mathlib_runtime
        self.provider_runtime = provider_runtime
        self.cache_root = cache_root
        self.session_backend = session_backend
        self.cache_results = cache_results
        self._source_template = _QUERY_SOURCE.read_text(encoding="utf-8")
        if (
            self._source_template.count(_IMPORT_TOKEN) != 1
            or self._source_template.count(_ENTRYPOINT_MARKER) != 1
        ):
            raise RuntimeError(
                "Lean declaration query source has invalid template markers"
            )
        self._sessions: dict[LeanEnvironment, _SessionEntry] = {}
        self._session_locks = {
            environment: threading.Lock() for environment in LeanEnvironment
        }
        self._finalizer = weakref.finalize(
            self,
            _close_declaration_sessions,
            self._sessions,
            self._session_locks,
        )

    def environment_digest(self, environment: LeanEnvironment) -> str:
        try:
            if _sha256_file(self.lean_executable) != self.provider_runtime.digest:
                raise LeanDeclarationBackendError(
                    "LEAN_ENVIRONMENT_CHANGED",
                    "The pinned Lean executable changed after capability registration.",
                )
            if "semantic_runtime" in self.provider_runtime.configuration:
                require_lean_semantic_runtime_identity(self.provider_runtime)
            return self._compute_environment_digest(environment)
        except LeanRuntimeIdentityError as exc:
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_CHANGED",
                "The pinned Lean semantic environment changed after registration.",
            ) from exc
        except OSError as exc:
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                f"The pinned Lean {environment.value} environment is not installed.",
            ) from exc

    def _resolve_session(
        self,
        environment: LeanEnvironment,
        environment_digest: str,
    ) -> _SessionEntry:
        """Return a valid session entry, discarding stale sessions as needed."""

        entry = self._sessions.get(environment)
        if entry is not None and entry.environment_digest != environment_digest:
            self._discard_session(environment)
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_CHANGED",
                (
                    "The pinned Lean environment changed while an indexed "
                    "declaration session was active."
                ),
            )
        if entry is None:
            session = self._start_session(environment, environment_digest)
            entry = _SessionEntry(
                environment_digest=environment_digest,
                session=session,
            )
            self._sessions[environment] = entry
            try:
                unchanged_after_start = (
                    self.environment_digest(environment) == environment_digest
                )
            except LeanDeclarationBackendError:
                self._discard_session(environment)
                raise
            if not unchanged_after_start:
                self._discard_session(environment)
                raise LeanDeclarationBackendError(
                    "LEAN_ENVIRONMENT_CHANGED",
                    (
                        "The pinned Lean environment changed while the "
                        "declaration index was starting."
                    ),
                )
        return entry

    def _execute_session_request(
        self,
        environment: LeanEnvironment,
        entry: _SessionEntry,
        environment_digest: str,
        query: LeanDeclarationQuery,
    ) -> LeanDeclarationPayload:
        """Run one bounded request, mapping not-found errors to environment changes."""

        try:
            output = entry.session.request(
                query,
                timeout_seconds=self._query_timeout_seconds(environment),
            )
        except LeanDeclarationBackendError as exc:
            if exc.code == "LEAN_DECLARATION_NOT_FOUND":
                try:
                    unchanged = (
                        self.environment_digest(environment) == environment_digest
                    )
                except LeanDeclarationBackendError:
                    unchanged = False
                if unchanged:
                    raise
            self._discard_session(environment)
            if exc.code == "LEAN_DECLARATION_NOT_FOUND":
                raise LeanDeclarationBackendError(
                    "LEAN_ENVIRONMENT_CHANGED",
                    (
                        "The pinned Lean environment changed during "
                        "declaration inspection."
                    ),
                ) from exc
            raise
        return output

    def _validate_session_unchanged(
        self,
        environment: LeanEnvironment,
        environment_digest: str,
    ) -> None:
        """Discard the session and fail when the environment changed after a query."""

        try:
            unchanged_after_query = (
                self.environment_digest(environment) == environment_digest
            )
        except LeanDeclarationBackendError:
            self._discard_session(environment)
            raise
        if not unchanged_after_query:
            self._discard_session(environment)
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_CHANGED",
                "The pinned Lean environment changed during declaration discovery.",
            )

    def query(
        self,
        environment: LeanEnvironment,
        query: LeanDeclarationQuery,
    ) -> LeanDeclarationBackendResult:
        with self._session_locks[environment]:
            environment_digest = self.environment_digest(environment)
            lean_version, lean_commit, mathlib_commit = self._runtime_identity(
                environment
            )
            entry = self._resolve_session(environment, environment_digest)
            output = self._execute_session_request(
                environment,
                entry,
                environment_digest,
                query,
            )
            self._validate_session_unchanged(environment, environment_digest)
            return LeanDeclarationBackendResult(
                environment_digest=environment_digest,
                lean_version=lean_version,
                lean_commit=lean_commit,
                mathlib_commit=mathlib_commit,
                payload=output,
            )

    def _runtime_identity(
        self,
        environment: LeanEnvironment,
    ) -> tuple[str, str, str | None]:
        profiles = self.provider_runtime.configuration.get("profiles")
        profile = (
            profiles.get(environment.value) if isinstance(profiles, dict) else None
        )
        lean_version = self.provider_runtime.version
        lean_commit = profile.get("lean_commit") if isinstance(profile, dict) else None
        mathlib_commit = (
            profile.get("mathlib_commit") if isinstance(profile, dict) else None
        )
        if not isinstance(lean_version, str) or not isinstance(lean_commit, str):
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned Lean runtime identity is incomplete.",
            )
        if environment is LeanEnvironment.MATHLIB and not isinstance(
            mathlib_commit, str
        ):
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned Mathlib runtime identity is incomplete.",
            )
        if environment is LeanEnvironment.CORE:
            mathlib_commit = None
        return lean_version, lean_commit, mathlib_commit

    def close(self) -> None:
        """Terminate all active query sessions."""

        for environment, lock in self._session_locks.items():
            with lock:
                self._discard_session(environment)
        self._finalizer.detach()

    def _start_session(
        self,
        environment: LeanEnvironment,
        environment_digest: str,
    ) -> _DeclarationQuerySession:
        import_name = (
            "Init.Prelude" if environment is LeanEnvironment.CORE else "Mathlib"
        )
        source = self._source_template.replace(_IMPORT_TOKEN, import_name)
        temporary_root = Path(tempfile.gettempdir())
        if self.session_backend == "persistent":
            command, cwd = self._persistent_command()
            process_environment = self._process_environment(
                LeanEnvironment.MATHLIB,
                temporary_root,
            )
            base_command = source.split(_ENTRYPOINT_MARKER, maxsplit=1)[0]
            return _PersistentLeanQuerySession(
                command=command,
                cwd=cwd,
                process_environment=process_environment,
                base_command=base_command,
                memory_mb="8192",
                environment_digest=environment_digest,
                cache_results=self.cache_results,
            )
        command, cwd, memory_mb, _ = self._command(
            environment,
            temporary_root,
        )
        process_environment = self._process_environment(
            environment,
            temporary_root,
        )
        index_environment_digest = self._compute_index_environment_digest(environment)
        return _ReusableLeanQuerySession(
            command=command,
            cwd=cwd,
            process_environment=process_environment,
            source=source,
            memory_mb=memory_mb,
            isolated_home=environment is LeanEnvironment.CORE,
            environment_digest=index_environment_digest,
            index_cache_path=self._index_cache_path(
                environment,
                index_environment_digest,
            ),
            cache_results=self.cache_results,
        )

    def _persistent_command(self) -> tuple[list[str], Path]:
        if self.mathlib_runtime is None:
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned Lean REPL project is unavailable.",
            )
        lake = self.lean_executable.with_name(
            "lake.exe" if self.lean_executable.suffix.lower() == ".exe" else "lake"
        )
        repl = (
            self.mathlib_runtime
            / ".lake"
            / "packages"
            / "repl"
            / ".lake"
            / "build"
            / "bin"
            / ("repl.exe" if os.name == "nt" else "repl")
        )
        if not lake.is_file() or not repl.is_file():
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned persistent Lean REPL backend is unavailable.",
            )
        return (
            [
                str(lake.resolve(strict=True)),
                "env",
                str(repl.resolve(strict=True)),
            ],
            self.mathlib_runtime,
        )

    def _index_cache_path(
        self,
        environment: LeanEnvironment,
        environment_digest: str,
    ) -> Path | None:
        if self.cache_root is None:
            return None
        identity = canonicalize_json(
            {
                "format": _INDEX_FORMAT,
                "environment": environment.value,
                "environment_digest": environment_digest,
            }
        )
        digest = hashlib.sha256(identity).hexdigest()
        return self.cache_root / f"{environment.value.casefold()}-{digest}.index"

    def _discard_session(self, environment: LeanEnvironment) -> None:
        entry = self._sessions.pop(environment, None)
        if entry is not None:
            entry.session.close()

    @staticmethod
    def _query_timeout_seconds(environment: LeanEnvironment) -> int:
        return 40 if environment is LeanEnvironment.CORE else 105

    def _command(
        self,
        environment: LeanEnvironment,
        temporary_root: Path,
    ) -> tuple[list[str], Path, str, int]:
        if environment is LeanEnvironment.CORE:
            return [str(self.lean_executable)], temporary_root, "1024", 40
        if self.mathlib_runtime is None:
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned Lean MATHLIB environment is not installed.",
            )
        lake = self.lean_executable.with_name(
            "lake.exe" if self.lean_executable.suffix.lower() == ".exe" else "lake"
        )
        if not lake.is_file():
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                "The pinned Lake executable for MATHLIB discovery is unavailable.",
            )
        return [str(lake), "env", "lean"], self.mathlib_runtime, "8192", 105

    def _process_environment(
        self,
        environment: LeanEnvironment,
        temporary_root: Path,
    ) -> dict[str, str]:
        lean_bin = str(self.lean_executable.parent)
        if environment is LeanEnvironment.MATHLIB:
            mathlib_runtime = self.mathlib_runtime
            if mathlib_runtime is None:
                raise LeanDeclarationBackendError(
                    "LEAN_ENVIRONMENT_UNAVAILABLE",
                    "The pinned Lean MATHLIB environment is not installed.",
                )
            # MATHLIB needs the host PATH (for elan/lake toolchain discovery)
            # and host HOME (for elan toolchain installs), with the pinned
            # Lean bin directory prepended to PATH.
            existing_path = os.environ.get("PATH", "")
            mathlib_path = (
                f"{lean_bin}{os.pathsep}{existing_path}" if existing_path else lean_bin
            )
            return worker_environment(
                extra_variables=("HOME", "ELAN_HOME"),
                overrides={
                    "PATH": mathlib_path,
                    **lean_mathlib_git_config(mathlib_runtime),
                },
            )
        # CORE uses an isolated HOME and a toolchain-only PATH.
        return worker_environment(
            overrides={
                "PATH": lean_bin,
                "HOME": str(temporary_root),
            },
        )

    def _compute_environment_digest(self, environment: LeanEnvironment) -> str:
        return self._digest_environment_identity(environment, portable=False)

    def _compute_index_environment_digest(
        self,
        environment: LeanEnvironment,
    ) -> str:
        return self._digest_environment_identity(environment, portable=True)

    def _digest_environment_identity(
        self,
        environment: LeanEnvironment,
        *,
        portable: bool,
    ) -> str:
        identity: dict[str, Any] = {
            "contract": "jacobian.lean.environment-manifest/v2",
            "environment": environment.value,
            "import_name": (
                "Init.Prelude" if environment is LeanEnvironment.CORE else "Mathlib"
            ),
            "lean_version": self.provider_runtime.version,
            "platform": self.provider_runtime.platform,
            "provider_digest": self.provider_runtime.digest,
        }
        semantic_runtime = self.provider_runtime.configuration.get("semantic_runtime")
        if isinstance(semantic_runtime, dict):
            digest_semantic_runtime = (
                lean_portable_semantic_runtime_digest
                if portable
                else lean_semantic_runtime_digest
            )
            identity["semantic_runtime_digest"] = digest_semantic_runtime(
                semantic_runtime
            )
        if environment is LeanEnvironment.MATHLIB:
            if self.mathlib_runtime is None:
                raise RuntimeError("cannot identify an unavailable Mathlib environment")
            profile = self.provider_runtime.configuration.get("profiles", {}).get(
                LeanEnvironment.MATHLIB.value,
                {},
            )
            identity.update(
                {
                    "lake_manifest_digest": _sha256_file(
                        self.mathlib_runtime / "lake-manifest.json"
                    ),
                    "lean_toolchain_digest": _sha256_file(
                        self.mathlib_runtime / "lean-toolchain"
                    ),
                    "mathlib_commit": profile.get("mathlib_commit"),
                }
            )
        return "sha256:" + hashlib.sha256(canonicalize_json(identity)).hexdigest()


class LeanDeclarationService:
    """Validate backend JSON into stable typed discovery results."""

    def __init__(self, backend: LeanDeclarationBackend) -> None:
        self.backend = backend

    def close(self) -> None:
        """Release backend-owned declaration sessions when supported."""

        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def search(
        self,
        query: LeanDeclarationSearchRequest,
    ) -> LeanDeclarationSearchOutput:
        type_constants = (
            query.type_pattern.constants if query.type_pattern is not None else ()
        )
        result = self.backend.query(
            query.environment,
            LeanDeclarationSearchQuery(
                name_contains=query.name_contains,
                type_constants=type_constants,
                namespace_prefixes=query.namespace_prefixes,
                target_module_prefixes=(
                    ("Init",) if query.environment is LeanEnvironment.CORE else ()
                ),
                kinds=query.kinds,
                limit=query.result_limit,
            ),
        )
        payload = result.payload
        if not isinstance(payload, LeanDeclarationSearchPayload):
            raise _protocol_error()
        try:
            return LeanDeclarationSearchOutput(
                environment=query.environment,
                environment_digest=result.environment_digest,
                lean_version=result.lean_version,
                lean_commit=result.lean_commit,
                mathlib_commit=result.mathlib_commit,
                query=query,
                declarations=payload.declarations,
                scanned_declarations=payload.scanned_declarations,
                stop_reason=payload.stop_reason,
            )
        except (ValueError, ValidationError) as exc:
            raise _protocol_error() from exc

    def inspect(
        self,
        query: LeanDeclarationInspectRequest,
    ) -> LeanDeclarationInspectOutput:
        result = self.backend.query(
            query.environment,
            LeanDeclarationInspectQuery(
                declaration_name=query.declaration_name,
                target_module_prefixes=(
                    ("Init",) if query.environment is LeanEnvironment.CORE else ()
                ),
            ),
        )
        payload = result.payload
        if not isinstance(payload, LeanDeclarationInspectPayload):
            raise _protocol_error()
        try:
            return LeanDeclarationInspectOutput(
                environment=query.environment,
                environment_digest=result.environment_digest,
                lean_version=result.lean_version,
                lean_commit=result.lean_commit,
                mathlib_commit=result.mathlib_commit,
                query=query,
                declaration=payload.declaration,
            )
        except (ValueError, ValidationError) as exc:
            raise _protocol_error() from exc

    def dependencies(
        self,
        query: LeanDependencyGraphRequest,
    ) -> LeanDependencyGraphArtifact:
        result = self.backend.query(
            query.environment,
            LeanDeclarationDependenciesQuery(
                declaration_name=query.root_declaration,
                target_module_prefixes=(
                    ("Init",) if query.environment is LeanEnvironment.CORE else ()
                ),
                max_depth=query.max_depth,
                max_nodes=query.max_nodes,
            ),
        )
        payload = result.payload
        if not isinstance(payload, LeanDeclarationDependenciesPayload):
            raise _protocol_error()
        try:
            return LeanDependencyGraphArtifact(
                environment=query.environment,
                environment_digest=result.environment_digest,
                query=query,
                nodes=payload.nodes,
                edges=payload.edges,
                frontier=payload.frontier,
                node_budget_exhausted=payload.node_budget_exhausted,
                closure_complete=payload.closure_complete,
            )
        except (ValueError, ValidationError) as exc:
            raise _protocol_error() from exc


def installed_lean_declaration_service(
    provider_runtime: CapabilityProviderRuntime,
    *,
    cache_root: Path | None = None,
) -> LeanDeclarationService:
    """Bind discovery to the same separately validated pinned runtime identity."""

    from jacobian_checkers import lean4

    lean_executable, mathlib_runtime = lean4.inspect_runtime(require_mathlib=True)
    return LeanDeclarationService(
        LeanSubprocessDeclarationBackend(
            lean_executable=lean_executable,
            mathlib_runtime=mathlib_runtime,
            provider_runtime=provider_runtime,
            cache_root=cache_root,
        )
    )


def _parse_process_response(
    stdout: bytes,
    *,
    expected_request_id: str,
) -> LeanDeclarationPayload:
    try:
        lines = stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise _protocol_error() from exc
    responses = [
        line for line in lines if line.startswith((_RESULT_PREFIX, _ERROR_PREFIX))
    ]
    if len(responses) != 1:
        raise _protocol_error()
    return _parse_session_response(
        responses[0],
        expected_request_id=expected_request_id,
    )


def _parse_session_response(
    line: str,
    *,
    expected_request_id: str,
) -> LeanDeclarationPayload:
    response_kind: Literal["result", "error"]
    if line.startswith(_RESULT_PREFIX):
        response_kind = "result"
        serialized = line.removeprefix(_RESULT_PREFIX)
    elif line.startswith(_ERROR_PREFIX):
        response_kind = "error"
        serialized = line.removeprefix(_ERROR_PREFIX)
    else:
        raise _protocol_error()
    try:
        decoded = loads_strict_json(serialized)
    except CanonicalizationError as exc:
        raise _protocol_error() from exc
    if not isinstance(decoded, dict):
        raise _protocol_error()
    return _parse_decoded_response(
        decoded,
        expected_request_id=expected_request_id,
        response_kind=response_kind,
    )


def _parse_persistent_response(
    decoded: dict[str, Any],
    *,
    expected_request_id: str,
) -> LeanDeclarationPayload:
    try:
        response = LeanReplCommandResponse.model_validate(decoded)
    except ValidationError as command_error:
        try:
            LeanReplErrorResponse.model_validate(decoded)
        except ValidationError:
            raise _protocol_error() from command_error
        raise _query_failed() from command_error
    markers = tuple(
        message.data
        for message in response.messages
        if message.data.startswith((_RESULT_PREFIX, _ERROR_PREFIX))
    )
    if len(markers) != 1:
        raise _protocol_error()
    return _parse_session_response(
        markers[0],
        expected_request_id=expected_request_id,
    )


def _parse_decoded_response(
    decoded: dict[str, Any],
    *,
    expected_request_id: str,
    response_kind: Literal["result", "error"] | None = None,
) -> LeanDeclarationPayload:
    if response_kind is None:
        if "payload" in decoded and "code" not in decoded:
            response_kind = "result"
        elif "code" in decoded and "payload" not in decoded:
            response_kind = "error"
        else:
            raise _protocol_error()
    if response_kind == "result":
        try:
            envelope = LeanDeclarationResultEnvelope.model_validate(decoded)
        except ValidationError as exc:
            raise _protocol_error() from exc
        if envelope.request_id != expected_request_id:
            raise _protocol_error()
        return envelope.payload
    try:
        error = LeanDeclarationErrorEnvelope.model_validate(decoded)
    except ValidationError as exc:
        raise _protocol_error() from exc
    if error.request_id != expected_request_id:
        raise _protocol_error()
    if error.code == "LEAN_DECLARATION_NOT_FOUND":
        raise LeanDeclarationBackendError(
            "LEAN_DECLARATION_NOT_FOUND",
            "Lean did not find the exact requested declaration.",
        )
    raise LeanDeclarationBackendError(
        "LEAN_QUERY_FAILED",
        "Lean could not complete declaration discovery in the selected environment.",
    )


def _protocol_error() -> LeanDeclarationBackendError:
    return LeanDeclarationBackendError(
        "LEAN_QUERY_PROTOCOL_ERROR",
        "Lean declaration discovery returned malformed structured output.",
    )


def _structured_output_limit() -> LeanDeclarationBackendError:
    return LeanDeclarationBackendError(
        "LEAN_QUERY_OUTPUT_LIMIT",
        "Lean declaration discovery exceeded its structured output budget.",
    )


def _diagnostic_output_limit() -> LeanDeclarationBackendError:
    return LeanDeclarationBackendError(
        "LEAN_QUERY_OUTPUT_LIMIT",
        "Lean declaration discovery exceeded its diagnostic output budget.",
    )


def _query_failed() -> LeanDeclarationBackendError:
    return LeanDeclarationBackendError(
        "LEAN_QUERY_FAILED",
        "Lean could not complete declaration discovery in the selected environment.",
    )


def _index_changed() -> LeanDeclarationBackendError:
    return LeanDeclarationBackendError(
        "LEAN_QUERY_INDEX_CHANGED",
        "The reusable Lean declaration index changed between bounded queries.",
    )


def _cacheable_query(query: LeanDeclarationQuery) -> bool:
    return isinstance(query, LeanDeclarationSearchQuery | LeanDeclarationInspectQuery)


def _query_cache_key(query: LeanDeclarationQuery) -> str:
    serialized = canonicalize_json(query.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def _catalog_query_from_index(
    query: LeanDeclarationQuery,
    *,
    index_path: Path,
    environment_digest: str,
    index_ready: bool,
) -> LeanDeclarationQuery:
    if (
        not isinstance(query, LeanDeclarationSearchQuery)
        or not index_ready
        or query.name_contains is None
    ):
        return query
    candidates: list[str] = []
    positions: list[int] = []
    candidate_bytes = 0
    scanned = 0
    try:
        with index_path.open(encoding="utf-8") as stream:
            if next(stream).rstrip("\n") != environment_digest:
                raise ValueError("declaration index environment mismatch")
            for line in stream:
                if line.startswith(f"{_INDEX_FOOTER_PREFIX}\t"):
                    break
                name, module, kind = line.rstrip("\n").split("\t")
                if not _prefix_matches(module, query.target_module_prefixes):
                    continue
                scanned += 1
                if (
                    not _prefix_matches(name, query.namespace_prefixes)
                    or (query.kinds and kind not in query.kinds)
                    or query.name_contains not in name
                ):
                    continue
                if query.type_constants or len(candidates) < query.limit:
                    candidates.append(name)
                    positions.append(scanned)
                    candidate_bytes += len(name.encode("utf-8"))
                    if (
                        len(candidates) > _MAX_CATALOG_CANDIDATES
                        or candidate_bytes > _MAX_CATALOG_NAME_BYTES
                    ):
                        return query
    except (OSError, StopIteration, ValueError) as exc:
        raise _index_changed() from exc
    return LeanDeclarationSearchQuery.model_validate(
        {
            **query.model_dump(mode="json"),
            "candidate_names": candidates,
            "candidate_scan_positions": positions,
            "scanned_declarations_total": scanned,
        }
    )


def _validate_declaration_index(
    path: Path,
    *,
    environment_digest: str,
) -> None:
    _scan_declaration_index(
        path,
        environment_digest=environment_digest,
        require_footer=True,
    )


def _seal_declaration_index(
    path: Path,
    *,
    environment_digest: str,
) -> None:
    row_count, content_digest = _scan_declaration_index(
        path,
        environment_digest=environment_digest,
        require_footer=False,
    )
    footer = f"{_INDEX_FOOTER_PREFIX}\t{row_count}\t{content_digest}\n".encode()
    if path.stat().st_size + len(footer) > _MAX_INDEX_BYTES:
        raise ValueError("declaration index exceeds its size bound")
    with path.open("ab") as stream:
        stream.write(footer)
        stream.flush()
        os.fsync(stream.fileno())
    _validate_declaration_index(path, environment_digest=environment_digest)


def _scan_declaration_index(
    path: Path,
    *,
    environment_digest: str,
    require_footer: bool,
) -> tuple[int, str]:
    size = path.stat().st_size
    if size == 0 or size > _MAX_INDEX_BYTES:
        raise ValueError("declaration index exceeds its size bound")
    content_hasher = hashlib.sha256()
    row_count = 0
    footer: tuple[str, str] | None = None
    with path.open("rb") as stream:
        header = _read_index_header(stream, environment_digest)
        content_hasher.update(header)
        for raw_line in stream:
            line = _decode_index_line(raw_line, label="row")
            footer = _parse_index_footer(line)
            if footer is not None:
                _require_index_eof(stream)
                break
            _validate_index_row(line)
            content_hasher.update(raw_line)
            row_count += 1
    content_digest = "sha256:" + content_hasher.hexdigest()
    if not require_footer:
        if footer is not None:
            raise ValueError("declaration index is already sealed")
        return row_count, content_digest
    _validate_index_footer(footer, row_count, content_digest)
    return row_count, content_digest


def _read_index_header(stream: BinaryIO, environment_digest: str) -> bytes:
    header = stream.readline()
    decoded_header = _decode_index_line(header, label="header")
    if decoded_header != environment_digest:
        raise ValueError("declaration index environment mismatch")
    return header


def _decode_index_line(raw_line: bytes, *, label: str) -> str:
    if not raw_line.endswith(b"\n"):
        raise ValueError(f"declaration index {label} is truncated")
    try:
        return raw_line[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("declaration index is not UTF-8") from exc


def _parse_index_footer(line: str) -> tuple[str, str] | None:
    if not line.startswith(f"{_INDEX_FOOTER_PREFIX}\t"):
        return None
    fields = line.split("\t")
    if len(fields) != 3:
        raise ValueError("declaration index footer is malformed")
    return fields[1], fields[2]


def _require_index_eof(stream: BinaryIO) -> None:
    if stream.read(1):
        raise ValueError("declaration index footer is not final")


def _validate_index_row(line: str) -> None:
    fields = line.split("\t")
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError("declaration index row is malformed")


def _validate_index_footer(
    footer: tuple[str, str] | None,
    row_count: int,
    content_digest: str,
) -> None:
    if footer is None:
        raise ValueError("declaration index footer is missing")
    if footer != (str(row_count), content_digest):
        raise ValueError("declaration index footer does not match its rows")


def _prefix_matches(value: str, prefixes: tuple[str, ...]) -> bool:
    return not prefixes or any(
        value == prefix or value.startswith(f"{prefix}.") for prefix in prefixes
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_bounded_file(source: Path, destination: Path, *, max_bytes: int) -> None:
    copied = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(min(1024 * 1024, max_bytes - copied + 1)):
            copied += len(chunk)
            if copied > max_bytes:
                raise ValueError("declaration index exceeds its size bound")
            output_stream.write(chunk)
