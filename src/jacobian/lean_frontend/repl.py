"""Bounded Lean REPL transport and reusable clean-session lifecycle."""

from __future__ import annotations

import shutil
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jacobian.contracts.lean import LeanEnvironment
from jacobian.process_policy import (
    BoundedInteractiveProcess,
    InteractiveProcessError,
    InteractiveProcessRequest,
)
from jacobian.references import LeanCheckerInstallation
from jacobian.worker_environment import worker_environment

_RESOURCE_POLL_SECONDS = 0.1


def _repl_process_environment() -> dict[str, str]:
    return worker_environment(
        extra_variables=("HOME", "PATH", "ELAN_HOME"),
    )


@dataclass(frozen=True, slots=True)
class LeanReplPolicy:
    """Bounds one reusable exploratory REPL process."""

    max_requests: int = 16
    max_age_seconds: float = 600
    max_rss_kb: int = 7 * 1024 * 1024
    timeout_seconds: float = 180

    def __post_init__(self) -> None:
        if self.max_requests < 1:
            raise ValueError("max_requests must be positive")
        if self.max_age_seconds <= 0 or self.timeout_seconds <= 0:
            raise ValueError("REPL time bounds must be positive")
        if self.max_rss_kb < 0:
            raise ValueError("max_rss_kb cannot be negative")


class PersistentLeanRepl:
    """Serialized, bounded client for one exploratory Lean REPL process."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        cwd: Path,
        base_command: str | None,
        policy: LeanReplPolicy,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._base_command = base_command
        self._policy = policy
        self._lock = threading.Lock()
        self._process: BoundedInteractiveProcess | None = None
        self._base_env: int | None = None
        self._started_at = 0.0
        self._requests = 0

    def execute(
        self,
        *,
        command: str,
        tactic: str,
        pickle_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one independent command and tactic from the immutable base env."""

        with self._lock:
            self._ensure_process()
            command_request: dict[str, Any] = {"cmd": command}
            if self._base_env is not None:
                command_request["env"] = self._base_env
            command_response = self._exchange(command_request)
            proof_state = _single_proof_state(command_response)
            tactic_response = self._exchange(
                {"tactic": tactic, "proofState": proof_state}
            )
            if pickle_path is not None and not _response_errors(tactic_response):
                successor = tactic_response.get("proofState")
                if not isinstance(successor, int):
                    raise RuntimeError(
                        "Lean REPL did not return a successor proof state"
                    )
                pickled = self._exchange(
                    {"proofState": successor, "pickleTo": str(pickle_path)}
                )
                if _response_errors(pickled):
                    raise RuntimeError("Lean REPL could not pickle the proof state")
            self._requests += 1
            return command_response, tactic_response

    def execute_validated(
        self,
        *,
        command: str,
        tactic: str,
        pickle_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Reconstruct, inspect, then advance one state in this process."""

        with self._lock:
            self._ensure_process()
            command_request: dict[str, Any] = {"cmd": command}
            if self._base_env is not None:
                command_request["env"] = self._base_env
            command_response = self._exchange(command_request)
            proof_state = _single_proof_state(command_response)
            validation_response = self._exchange(
                {"tactic": "skip", "proofState": proof_state}
            )
            validated_state = validation_response.get("proofState")
            if not isinstance(validated_state, int):
                raise RuntimeError("Lean REPL did not return the validated proof state")
            tactic_response = self._exchange(
                {"tactic": tactic, "proofState": validated_state}
            )
            if pickle_path is not None and not _response_errors(tactic_response):
                successor = tactic_response.get("proofState")
                if not isinstance(successor, int):
                    raise RuntimeError(
                        "Lean REPL did not return a successor proof state"
                    )
                pickled = self._exchange(
                    {"proofState": successor, "pickleTo": str(pickle_path)}
                )
                if _response_errors(pickled):
                    raise RuntimeError("Lean REPL could not pickle the proof state")
            self._requests += 1
            return command_response, validation_response, tactic_response

    def close(self) -> None:
        """Stop the process and discard all retained snapshots."""

        with self._lock:
            self._stop_process()

    def _ensure_process(self) -> None:
        if self._process is not None and self._expired():
            self._stop_process()
        if self._process is not None:
            return
        process = BoundedInteractiveProcess(
            InteractiveProcessRequest(
                executable=self._command[0],
                arguments=self._command[1:],
                environment=_repl_process_environment(),
                cwd=str(self._cwd.resolve(strict=True)),
                startup_timeout_seconds=self._policy.timeout_seconds,
                read_timeout_seconds=self._policy.timeout_seconds,
                shutdown_timeout_seconds=2.0,
                stderr_limit_bytes=128 * 1024,
                base_command=self._base_command,
                max_rss_kb=self._policy.max_rss_kb,
            )
        )
        try:
            process.start()
        except InteractiveProcessError as exc:
            raise RuntimeError("Lean REPL could not start") from exc
        self._process = process
        self._started_at = time.monotonic()
        self._requests = 0
        self._base_env = None
        if self._base_command is not None:
            response = process.base_response
            if response is None:
                self._stop_process()
                raise RuntimeError("Lean REPL did not return a base response")
            base_env = response.get("env")
            if not isinstance(base_env, int):
                self._stop_process()
                raise RuntimeError("Lean REPL did not return a base environment")
            self._base_env = base_env

    def _expired(self) -> bool:
        assert self._process is not None
        if not self._process.is_running:
            return True
        if self._requests >= self._policy.max_requests:
            return True
        return time.monotonic() - self._started_at >= self._policy.max_age_seconds

    def _exchange(self, request: Mapping[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError("Lean REPL is unavailable")
        try:
            return process.exchange(request)
        except InteractiveProcessError as exc:
            self._stop_process()
            detail = str(exc)
            if "timed out" in detail:
                raise RuntimeError("Lean REPL timed out") from exc
            if "stderr limit" in detail:
                raise RuntimeError("Lean REPL exceeded its output limit") from exc
            if "memory limit" in detail:
                raise RuntimeError("Lean REPL exceeded its memory limit") from exc
            raise RuntimeError("Lean REPL stopped before returning a result") from exc

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._base_env = None
        if process is not None:
            process.close()


def _single_proof_state(response: Mapping[str, Any]) -> int:
    sorries = response.get("sorries")
    if (
        not isinstance(sorries, list)
        or len(sorries) != 1
        or not isinstance(sorries[0], Mapping)
        or not isinstance(sorries[0].get("proofState"), int)
    ):
        errors = _response_errors(response)
        if errors:
            raise RuntimeError(
                "Lean did not expose one replayable proof state: " + "; ".join(errors)
            )
        raise RuntimeError("Lean did not expose one replayable proof state")
    proof_state = sorries[0]["proofState"]
    assert isinstance(proof_state, int)
    return proof_state


class LeanExplorationReplRuntime:
    """Own bounded REPL sessions used only by exploratory capabilities."""

    def __init__(
        self,
        runtime: Path,
        installations: Mapping[LeanEnvironment, LeanCheckerInstallation],
        *,
        policy: LeanReplPolicy | None = None,
    ) -> None:
        self._runtime = runtime
        self._installations = installations
        self._policy = policy or LeanReplPolicy()
        self._sessions: dict[LeanEnvironment, PersistentLeanRepl] = {}
        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._finalizer = weakref.finalize(self, _close_repls, self._sessions)

    def execute(
        self,
        *,
        command: str,
        tactic: str,
        environment: LeanEnvironment,
        pickle_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Serialize exploration and reuse only an environment's base snapshot."""

        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Lean exploration runtime is closing")
            session = self._sessions.get(environment)
            if session is None:
                session = self._create_session(environment)
                self._sessions[environment] = session
            return session.execute(
                command=command,
                tactic=tactic,
                pickle_path=pickle_path,
            )

    def execute_clean(
        self,
        *,
        command: str,
        tactic: str,
        environment: LeanEnvironment,
        pickle_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Replay and apply in a new process that is always discarded."""

        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Lean exploration runtime is closing")
            session = self._create_session(environment)
            try:
                if pickle_path is None:
                    return session.execute_validated(
                        command=command,
                        tactic=tactic,
                    )
                return session.execute_validated(
                    command=command,
                    tactic=tactic,
                    pickle_path=pickle_path,
                )
            finally:
                session.close()

    def close(self) -> None:
        """Stop every exploration process without affecting independent checkers."""

        with self._lock:
            if self._closed:
                return
            self._closing = True
            sessions = tuple(self._sessions.items())
        failures: list[Exception] = []
        for environment, session in sessions:
            try:
                session.close()
            except Exception as exc:
                failures.append(exc)
            else:
                with self._lock:
                    self._sessions.pop(environment, None)
        if failures:
            raise ExceptionGroup("Lean exploration sessions failed to close", failures)
        with self._lock:
            self._finalizer.detach()
            self._closed = True
            self._closing = False

    def _create_session(self, environment: LeanEnvironment) -> PersistentLeanRepl:
        elan = shutil.which("elan")
        if elan is None:
            raise RuntimeError("elan is unavailable")
        installation = self._installations[environment]
        repl = (
            self._runtime
            / ".lake"
            / "packages"
            / "repl"
            / ".lake"
            / "build"
            / "bin"
            / "repl"
        )
        if not repl.is_file():
            raise RuntimeError(
                "the pinned Lean REPL is unavailable; run `lake build repl` in lean/"
            )
        policy = self._policy
        if environment is LeanEnvironment.CORE:
            policy = replace(policy, timeout_seconds=min(policy.timeout_seconds, 30))
        return PersistentLeanRepl(
            command=(
                elan,
                "run",
                f"leanprover/lean4:v{installation.lean_version}",
                "lake",
                "env",
                str(repl),
            ),
            cwd=self._runtime,
            base_command=(
                "import Mathlib" if environment is LeanEnvironment.MATHLIB else None
            ),
            policy=policy,
        )


def _close_repls(
    sessions: Mapping[LeanEnvironment, PersistentLeanRepl],
) -> None:
    for session in sessions.values():
        session.close()


def _response_errors(response: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    for key in ("error", "message"):
        value = response.get(key)
        if isinstance(value, str):
            errors.append(value)
    messages = response.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if (
                isinstance(item, Mapping)
                and str(item.get("severity", "")).lower() == "error"
            ):
                data = item.get("data")
                if isinstance(data, str):
                    errors.append(data)
    status = str(response.get("proofStatus", "")).lower()
    if status in {"error", "failed", "failure"} and not errors:
        errors.append(f"Lean tactic returned proof status {response['proofStatus']!r}")
    return tuple(errors)


__all__ = ["LeanExplorationReplRuntime", "LeanReplPolicy", "PersistentLeanRepl"]
