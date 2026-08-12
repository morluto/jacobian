"""Read-only declaration discovery over pinned Lean environments."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.providers.lean_runtime import (
    LeanRuntimeIdentityError,
    lean_mathlib_git_config,
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
_QUERY_SOURCE = Path(__file__).with_name("_lean_declaration_query.lean")
_IMPORT_TOKEN = "{{JACOBIAN_IMPORT}}"


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


class _DeclarationQuerySession(Protocol):
    def request(
        self,
        query: LeanDeclarationQuery,
        *,
        timeout_seconds: int,
    ) -> LeanDeclarationPayload: ...

    def close(self) -> None: ...


class _ReusableLeanQuerySession:
    """Run bounded queries while reusing one environment-bound name index."""

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

    def request(
        self,
        query: LeanDeclarationQuery,
        *,
        timeout_seconds: int,
    ) -> LeanDeclarationPayload:
        if self._closed:
            raise _query_failed()
        self._check_index_identity()
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
            current_digest = _sha256_file(self._index_path)
            self._validate_index_header()
        except OSError as exc:
            raise _protocol_error() from exc
        if self._index_digest is None:
            self._index_digest = current_digest
        elif current_digest != self._index_digest:
            raise _index_changed()

    def _catalog_query(
        self,
        query: LeanDeclarationQuery,
    ) -> LeanDeclarationQuery:
        if (
            not isinstance(query, LeanDeclarationSearchQuery)
            or self._index_digest is None
            or query.name_contains is None
        ):
            return query
        target_modules = query.target_module_prefixes
        namespaces = query.namespace_prefixes
        kinds = set(query.kinds)
        name_contains = query.name_contains
        type_constants = query.type_constants
        limit = query.limit
        candidates: list[str] = []
        positions: list[int] = []
        candidate_bytes = 0
        scanned = 0
        try:
            with self._index_path.open(encoding="utf-8") as stream:
                if next(stream).rstrip("\n") != self._environment_digest:
                    raise ValueError("declaration index environment mismatch")
                for line in stream:
                    name, module, kind = line.rstrip("\n").split("\t")
                    if not _prefix_matches(module, target_modules):
                        continue
                    scanned += 1
                    if (
                        not _prefix_matches(name, namespaces)
                        or (kinds and kind not in kinds)
                        or name_contains not in name
                    ):
                        continue
                    if type_constants or len(candidates) < limit:
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

    def _validate_index_header(self) -> None:
        with self._index_path.open(encoding="utf-8") as stream:
            if stream.readline().rstrip("\n") != self._environment_digest:
                raise OSError("declaration index environment mismatch")


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
    """Reuse one catalog across bounded processes for each validated environment."""

    def __init__(
        self,
        *,
        lean_executable: Path,
        mathlib_runtime: Path | None,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.lean_executable = lean_executable
        self.mathlib_runtime = mathlib_runtime
        self.provider_runtime = provider_runtime
        self._source_template = _QUERY_SOURCE.read_text(encoding="utf-8")
        if self._source_template.count(_IMPORT_TOKEN) != 1:
            raise RuntimeError(
                "Lean declaration query source has an invalid import token"
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
        command, cwd, memory_mb, _ = self._command(
            environment,
            temporary_root,
        )
        process_environment = self._process_environment(
            environment,
            temporary_root,
        )
        return _ReusableLeanQuerySession(
            command=command,
            cwd=cwd,
            process_environment=process_environment,
            source=source,
            memory_mb=memory_mb,
            isolated_home=environment is LeanEnvironment.CORE,
            environment_digest=environment_digest,
        )

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
            identity["semantic_runtime_digest"] = lean_semantic_runtime_digest(
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
) -> LeanDeclarationService:
    """Bind discovery to the same separately validated pinned runtime identity."""

    from jacobian_checkers import lean4

    lean_executable, mathlib_runtime = lean4.inspect_runtime(require_mathlib=True)
    return LeanDeclarationService(
        LeanSubprocessDeclarationBackend(
            lean_executable=lean_executable,
            mathlib_runtime=mathlib_runtime,
            provider_runtime=provider_runtime,
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
