"""Read-only declaration discovery over pinned Lean environments."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import uuid
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
    LeanDeclarationRecord,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDeclarationSearchStopReason,
    LeanDependencyGraphArtifact,
    LeanDependencyGraphRequest,
    LeanEnvironment,
)

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
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


class LeanDeclarationBackendError(RuntimeError):
    """A bounded backend failure safe for capability diagnostic mapping."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _DeclarationQuerySession(Protocol):
    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]: ...

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
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if self._closed:
            raise _query_failed()
        self._check_index_identity()
        request_id = uuid.uuid4().hex
        wire_payload = {
            **payload,
            **self._catalog_fields(payload),
            "request_id": request_id,
        }
        try:
            self._request_path.write_bytes(canonicalize_json(wire_payload))
            completed = subprocess.run(
                [
                    *self._command,
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
                ],
                cwd=self._cwd,
                env=self._process_environment,
                check=False,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise LeanDeclarationBackendError(
                "LEAN_QUERY_TIMEOUT",
                (
                    "Lean declaration discovery exceeded the "
                    f"{timeout_seconds}-second per-query budget."
                ),
            ) from exc
        except OSError as exc:
            raise _query_failed() from exc
        if len(completed.stdout) > _MAX_STDOUT_BYTES:
            raise _structured_output_limit()
        if len(completed.stderr) > _MAX_STDERR_BYTES:
            raise _diagnostic_output_limit()
        if completed.returncode != 0:
            _LOGGER.warning(
                "Lean declaration query failed: %s",
                (completed.stdout + completed.stderr)
                .decode("utf-8", errors="replace")
                .strip(),
            )
            raise _query_failed()
        output = _parse_process_response(
            completed.stdout,
            expected_request_id=request_id,
        )
        self._record_or_check_index(payload)
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

    def _record_or_check_index(self, payload: dict[str, Any]) -> None:
        if payload.get("operation") != "search":
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

    def _catalog_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            payload.get("operation") != "search"
            or self._index_digest is None
            or payload.get("name_contains") is None
        ):
            return {
                "candidate_names": [],
                "candidate_scan_positions": [],
                "scanned_declarations_total": None,
            }
        target_modules = tuple(payload.get("target_module_prefixes", ()))
        namespaces = tuple(payload.get("namespace_prefixes", ()))
        kinds = set(payload.get("kinds", ()))
        name_contains = str(payload["name_contains"])
        type_constants = tuple(payload.get("type_constants", ()))
        limit = int(payload["limit"])
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
                            return {
                                "candidate_names": [],
                                "candidate_scan_positions": [],
                                "scanned_declarations_total": None,
                            }
        except (OSError, StopIteration, ValueError) as exc:
            raise _index_changed() from exc
        return {
            "candidate_names": candidates,
            "candidate_scan_positions": positions,
            "scanned_declarations_total": scanned,
        }

    def _validate_index_header(self) -> None:
        with self._index_path.open(encoding="utf-8") as stream:
            if stream.readline().rstrip("\n") != self._environment_digest:
                raise OSError("declaration index environment mismatch")


@dataclass(frozen=True, slots=True)
class _SessionEntry:
    environment_digest: str
    session: _DeclarationQuerySession


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

    def environment_digest(self, environment: LeanEnvironment) -> str:
        try:
            if _sha256_file(self.lean_executable) != self.provider_runtime.digest:
                raise LeanDeclarationBackendError(
                    "LEAN_ENVIRONMENT_CHANGED",
                    "The pinned Lean executable changed after capability registration.",
                )
            return self._compute_environment_digest(environment)
        except OSError as exc:
            raise LeanDeclarationBackendError(
                "LEAN_ENVIRONMENT_UNAVAILABLE",
                f"The pinned Lean {environment.value} environment is not installed.",
            ) from exc

    def query(
        self,
        environment: LeanEnvironment,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._session_locks[environment]:
            environment_digest = self.environment_digest(environment)
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
            try:
                output = entry.session.request(
                    payload,
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
            output["_environment_digest"] = environment_digest
            return output

    def close(self) -> None:
        """Terminate all active query sessions."""

        for environment, lock in self._session_locks.items():
            with lock:
                self._discard_session(environment)

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
        existing_path = os.environ.get("PATH")
        path = (
            f"{lean_bin}{os.pathsep}{existing_path}"
            if existing_path is not None
            else lean_bin
        )
        runtime_home = (
            os.environ.get("HOME", str(temporary_root))
            if environment is LeanEnvironment.MATHLIB
            else str(temporary_root)
        )
        return {
            "HOME": runtime_home,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": path,
        }

    def _compute_environment_digest(self, environment: LeanEnvironment) -> str:
        identity: dict[str, Any] = {
            "contract": "jacobian.lean.environment-manifest/v1",
            "environment": environment.value,
            "import_name": (
                "Init.Prelude" if environment is LeanEnvironment.CORE else "Mathlib"
            ),
            "lean_version": self.provider_runtime.version,
            "platform": self.provider_runtime.platform,
            "provider_digest": self.provider_runtime.digest,
        }
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
            list(query.type_pattern.constants) if query.type_pattern is not None else []
        )
        raw = self.backend.query(
            query.environment,
            {
                "operation": "search",
                "declaration_name": None,
                "name_contains": query.name_contains,
                "type_constants": type_constants,
                "namespace_prefixes": list(query.namespace_prefixes),
                "target_module_prefixes": (
                    ["Init"] if query.environment is LeanEnvironment.CORE else []
                ),
                "kinds": [kind.value for kind in query.kinds],
                "limit": query.result_limit,
                "max_depth": 0,
                "max_nodes": 1,
            },
        )
        _require_operation(raw, "search")
        environment_digest = (
            raw["_environment_digest"]
            if "_environment_digest" in raw
            else self.backend.environment_digest(query.environment)
        )
        try:
            return LeanDeclarationSearchOutput(
                environment=query.environment,
                environment_digest=environment_digest,
                query=query,
                declarations=tuple(
                    LeanDeclarationRecord.model_validate(item)
                    for item in raw["declarations"]
                ),
                scanned_declarations=raw["scanned_declarations"],
                stop_reason=LeanDeclarationSearchStopReason(raw["stop_reason"]),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise _protocol_error() from exc

    def inspect(
        self,
        query: LeanDeclarationInspectRequest,
    ) -> LeanDeclarationInspectOutput:
        raw = self.backend.query(
            query.environment,
            {
                "operation": "inspect",
                "declaration_name": query.declaration_name,
                "name_contains": None,
                "type_constants": [],
                "namespace_prefixes": [],
                "target_module_prefixes": (
                    ["Init"] if query.environment is LeanEnvironment.CORE else []
                ),
                "kinds": [],
                "limit": 1,
                "max_depth": 0,
                "max_nodes": 1,
            },
        )
        _require_operation(raw, "inspect")
        environment_digest = (
            raw["_environment_digest"]
            if "_environment_digest" in raw
            else self.backend.environment_digest(query.environment)
        )
        try:
            return LeanDeclarationInspectOutput(
                environment=query.environment,
                environment_digest=environment_digest,
                query=query,
                declaration=LeanDeclarationRecord.model_validate(raw["declaration"]),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise _protocol_error() from exc

    def dependencies(
        self,
        query: LeanDependencyGraphRequest,
    ) -> LeanDependencyGraphArtifact:
        raw = self.backend.query(
            query.environment,
            {
                "operation": "dependencies",
                "declaration_name": query.root_declaration,
                "name_contains": None,
                "type_constants": [],
                "namespace_prefixes": [],
                "target_module_prefixes": (
                    ["Init"] if query.environment is LeanEnvironment.CORE else []
                ),
                "kinds": [],
                "limit": 1,
                "max_depth": query.max_depth,
                "max_nodes": query.max_nodes,
            },
        )
        _require_operation(raw, "dependencies")
        environment_digest = (
            raw["_environment_digest"]
            if "_environment_digest" in raw
            else self.backend.environment_digest(query.environment)
        )
        try:
            return LeanDependencyGraphArtifact(
                environment=query.environment,
                environment_digest=environment_digest,
                query=query,
                nodes=raw["nodes"],
                edges=raw["edges"],
                frontier=raw["frontier"],
                node_budget_exhausted=raw["node_budget_exhausted"],
                closure_complete=raw["closure_complete"],
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    if line.startswith(_RESULT_PREFIX):
        response_kind = "result"
        serialized = line.removeprefix(_RESULT_PREFIX)
    elif line.startswith(_ERROR_PREFIX):
        response_kind = "error"
        serialized = line.removeprefix(_ERROR_PREFIX)
    else:
        raise _protocol_error()
    try:
        envelope = loads_strict_json(serialized)
    except CanonicalizationError as exc:
        raise _protocol_error() from exc
    if not isinstance(envelope, dict) or envelope.get("request_id") != (
        expected_request_id
    ):
        raise _protocol_error()
    if response_kind == "result":
        if set(envelope) != {"request_id", "payload"} or not isinstance(
            envelope["payload"], dict
        ):
            raise _protocol_error()
        return envelope["payload"]
    if (
        set(envelope) != {"request_id", "code", "message"}
        or not isinstance(envelope["code"], str)
        or not isinstance(envelope["message"], str)
    ):
        raise _protocol_error()
    if envelope["code"] == "LEAN_DECLARATION_NOT_FOUND":
        raise LeanDeclarationBackendError(
            "LEAN_DECLARATION_NOT_FOUND",
            "Lean did not find the exact requested declaration.",
        )
    raise LeanDeclarationBackendError(
        "LEAN_QUERY_FAILED",
        "Lean could not complete declaration discovery in the selected environment.",
    )


def _require_operation(payload: dict[str, Any], expected: str) -> None:
    if payload.get("operation") != expected:
        raise _protocol_error()


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


def _prefix_matches(value: str, prefixes: tuple[Any, ...]) -> bool:
    return not prefixes or any(
        isinstance(prefix, str) and (value == prefix or value.startswith(f"{prefix}."))
        for prefix in prefixes
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
