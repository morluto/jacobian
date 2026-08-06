"""Pinned Lean declaration session lifecycle and protocol behavior."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanSubprocessDeclarationBackend,
    _parse_session_response,
)

_RUNTIME = CapabilityProviderRuntime(
    provider="jacobian.lean4",
    availability=CapabilityProviderAvailability.AVAILABLE,
    version="4.31.0",
    digest="sha256:" + "b" * 64,
    digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
    platform="test",
    install_tier=CapabilityInstallTier.T3,
    license_id="Apache-2.0",
    features=("CORE", "MATHLIB"),
)


@dataclass
class RecordingSession:
    responses: list[dict[str, Any] | LeanDeclarationBackendError]
    requests: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    active: int = 0
    max_active: int = 0
    request_started: threading.Event | None = None
    release_request: threading.Event | None = None
    after_request: Any = None
    activity_lock: threading.Lock = field(default_factory=threading.Lock)

    def request(
        self, payload: dict[str, Any], *, timeout_seconds: int
    ) -> dict[str, Any]:
        del timeout_seconds
        with self.activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.requests.append(payload)
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
        self.closed = True


class RecordingSubprocessBackend(LeanSubprocessDeclarationBackend):
    def __init__(
        self,
        *,
        lean_executable: Path,
        provider_runtime: CapabilityProviderRuntime,
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


def test_subprocess_backend_reuses_one_pinned_session_per_environment(
    tmp_path: Path,
) -> None:
    session = RecordingSession(
        responses=[
            {"operation": "search", "declarations": []},
            {"operation": "inspect", "declaration": {"name": "Nat.add"}},
        ]
    )
    backend, _ = _recording_backend(tmp_path, [session])
    searched = backend.query(LeanEnvironment.CORE, {"operation": "search"})
    inspected = backend.query(LeanEnvironment.CORE, {"operation": "inspect"})
    assert len(backend.started_sessions) == 1
    assert [request["operation"] for request in session.requests] == [
        "search",
        "inspect",
    ]
    assert searched["_environment_digest"] == inspected["_environment_digest"]
    backend.close()
    assert session.closed


def test_subprocess_backend_serializes_concurrent_session_requests(
    tmp_path: Path,
) -> None:
    request_started = threading.Event()
    release_request = threading.Event()
    session = RecordingSession(
        responses=[
            {"operation": "search", "declarations": []},
            {"operation": "search", "declarations": []},
        ],
        request_started=request_started,
        release_request=release_request,
    )
    backend, _ = _recording_backend(tmp_path, [session])
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            backend.query, LeanEnvironment.CORE, {"operation": "search"}
        )
        assert request_started.wait(timeout=1)
        second_started = threading.Event()

        def query_second() -> dict[str, Any]:
            second_started.set()
            return backend.query(LeanEnvironment.CORE, {"operation": "search"})

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
    session = RecordingSession(responses=[{"operation": "search"}])
    backend, executable = _recording_backend(tmp_path, [session])
    session.after_request = lambda: executable.write_bytes(b"changed executable")
    with pytest.raises(LeanDeclarationBackendError) as raised:
        backend.query(LeanEnvironment.CORE, {"operation": "search"})
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
    replacement = RecordingSession(responses=[{"operation": "search"}])
    backend, _ = _recording_backend(tmp_path, [timed_out, replacement])
    with pytest.raises(LeanDeclarationBackendError) as raised:
        backend.query(LeanEnvironment.CORE, {"operation": "search"})
    result = backend.query(LeanEnvironment.CORE, {"operation": "search"})
    assert raised.value.code == "LEAN_QUERY_TIMEOUT"
    assert timed_out.closed
    assert len(backend.started_sessions) == 2
    assert result["operation"] == "search"
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
