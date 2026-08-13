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

from pydantic import ValidationError

from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.process_environment import (
    lean_elan_worker_environment,
)
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandRequest,
    LeanReplCommandResponse,
    LeanReplErrorResponse,
    LeanReplExecution,
    LeanReplPickleProofStateRequest,
    LeanReplPickleProofStateResponse,
    LeanReplPickleResponse,
    LeanReplProofResponse,
    LeanReplProofStepRequest,
    LeanReplProofStepResponse,
    LeanReplRequest,
    LeanReplResponse,
    LeanReplValidatedExecution,
)
from jacobian.process_policy import (
    BoundedInteractiveProcess,
    InteractiveProcessError,
    InteractiveProcessRequest,
)
from jacobian.providers.lean_runtime import lean_mathlib_git_config

_RESOURCE_POLL_SECONDS = 0.1
_DEFAULT_CORE_MAX_RSS_KB = 7 * 1024 * 1024
_DEFAULT_MATHLIB_MAX_RSS_KB = 9 * 1024 * 1024


def _repl_process_environment(runtime: Path) -> dict[str, str]:
    overrides = dict(
        lean_mathlib_git_config(runtime)
        if (runtime / "lake-manifest.json").is_file()
        else {}
    )
    return lean_elan_worker_environment(overrides=overrides)


@dataclass(frozen=True, slots=True)
class LeanReplPolicy:
    """Bounds one reusable exploratory REPL process."""

    max_requests: int = 16
    max_age_seconds: float = 600
    max_rss_kb: int = _DEFAULT_CORE_MAX_RSS_KB
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
    ) -> LeanReplExecution:
        """Run one independent command and tactic from the immutable base env."""

        with self._lock:
            self._ensure_process()
            command_request = LeanReplCommandRequest(
                cmd=command,
                env=self._base_env,
            )
            command_response = self._exchange_command(command_request)
            proof_state = _single_proof_state(command_response)
            if not isinstance(command_response, LeanReplCommandResponse):
                raise RuntimeError("Lean REPL returned an invalid command response")
            tactic_response = self._exchange_proof(
                LeanReplProofStepRequest(tactic=tactic, proof_state=proof_state)
            )
            if pickle_path is not None and not _response_errors(tactic_response):
                if not isinstance(tactic_response, LeanReplProofStepResponse):
                    raise RuntimeError(
                        "Lean REPL did not return a successor proof state"
                    )
                pickled = self._exchange_pickle(
                    LeanReplPickleProofStateRequest(
                        proof_state=tactic_response.proof_state,
                        pickle_to=str(pickle_path),
                    )
                )
                if not isinstance(pickled, LeanReplPickleProofStateResponse):
                    raise RuntimeError("Lean REPL could not pickle the proof state")
            self._requests += 1
            return command_response, tactic_response

    def execute_validated(
        self,
        *,
        command: str,
        tactic: str,
        pickle_path: Path | None = None,
    ) -> LeanReplValidatedExecution:
        """Reconstruct, inspect, then advance one state in this process."""

        with self._lock:
            self._ensure_process()
            command_request = LeanReplCommandRequest(
                cmd=command,
                env=self._base_env,
            )
            command_response = self._exchange_command(command_request)
            proof_state = _single_proof_state(command_response)
            if not isinstance(command_response, LeanReplCommandResponse):
                raise RuntimeError("Lean REPL returned an invalid command response")
            validation_response = self._exchange_proof(
                LeanReplProofStepRequest(tactic="skip", proof_state=proof_state)
            )
            if not isinstance(validation_response, LeanReplProofStepResponse):
                raise RuntimeError("Lean REPL did not return the validated proof state")
            tactic_response = self._exchange_proof(
                LeanReplProofStepRequest(
                    tactic=tactic,
                    proof_state=validation_response.proof_state,
                )
            )
            if pickle_path is not None and not _response_errors(tactic_response):
                if not isinstance(tactic_response, LeanReplProofStepResponse):
                    raise RuntimeError(
                        "Lean REPL did not return a successor proof state"
                    )
                pickled = self._exchange_pickle(
                    LeanReplPickleProofStateRequest(
                        proof_state=tactic_response.proof_state,
                        pickle_to=str(pickle_path),
                    )
                )
                if not isinstance(pickled, LeanReplPickleProofStateResponse):
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
                environment=_repl_process_environment(self._cwd),
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
            parsed = _parse_command_response(response)
            if not isinstance(parsed, LeanReplCommandResponse):
                self._stop_process()
                raise RuntimeError(
                    f"Lean REPL rejected its base command: {parsed.message}"
                )
            self._base_env = parsed.env

    def _expired(self) -> bool:
        if self._process is None:
            raise RuntimeError("Lean REPL process is unexpectedly None")
        if not self._process.is_running:
            return True
        if self._requests >= self._policy.max_requests:
            return True
        return time.monotonic() - self._started_at >= self._policy.max_age_seconds

    def _exchange_command(
        self,
        request: LeanReplCommandRequest,
    ) -> LeanReplCommandResponse | LeanReplErrorResponse:
        return _parse_command_response(self._exchange(request))

    def _exchange_proof(
        self,
        request: LeanReplProofStepRequest,
    ) -> LeanReplProofResponse:
        return _parse_proof_response(self._exchange(request))

    def _exchange_pickle(
        self,
        request: LeanReplPickleProofStateRequest,
    ) -> LeanReplPickleResponse:
        return _parse_pickle_response(self._exchange(request))

    def _exchange(self, request: LeanReplRequest) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError("Lean REPL is unavailable")
        try:
            return process.exchange(
                request.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
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


def _single_proof_state(
    response: LeanReplCommandResponse | LeanReplErrorResponse,
) -> int:
    if not isinstance(response, LeanReplCommandResponse):
        raise RuntimeError(
            "Lean did not expose one replayable proof state: " + response.message
        )
    if len(response.sorries) != 1 or response.sorries[0].proof_state is None:
        errors = _response_errors(response)
        if errors:
            raise RuntimeError(
                "Lean did not expose one replayable proof state: " + "; ".join(errors)
            )
        raise RuntimeError("Lean did not expose one replayable proof state")
    return response.sorries[0].proof_state


class LeanExplorationReplRuntime:
    """Own bounded REPL sessions used only by exploratory operations."""

    def __init__(
        self,
        runtime: Path,
        installations: Mapping[LeanEnvironment, LeanCheckerInstallation],
        *,
        policy: LeanReplPolicy | None = None,
    ) -> None:
        self._runtime = runtime
        self._installations = installations
        self._policy = policy
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
    ) -> LeanReplExecution:
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
    ) -> LeanReplValidatedExecution:
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

    def execute_persistent_validated(
        self,
        *,
        command: str,
        tactic: str,
        environment: LeanEnvironment,
        pickle_path: Path | None = None,
    ) -> LeanReplValidatedExecution:
        """Replay and apply through a retained bounded environment session.

        This backend candidate is intentionally not wired to an agent-facing
        operation.  It exists so evaluation can compare the same validated
        transition contract against clean-process replay without changing the
        atomic operation surface.
        """

        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Lean exploration runtime is closing")
            session = self._sessions.get(environment)
            if session is None:
                session = self._create_session(environment)
                self._sessions[environment] = session
            return session.execute_validated(
                command=command,
                tactic=tactic,
                pickle_path=pickle_path,
            )

    def close(self) -> None:
        """Stop every exploration process without affecting independent checkers."""

        with self._lock:
            if self._closed:
                return
            self._closing = True
            sessions = tuple(self._sessions.items())
        failures: list[BaseException] = []
        for environment, session in sessions:
            try:
                session.close()
            except BaseException as exc:
                failures.append(exc)
            else:
                with self._lock:
                    self._sessions.pop(environment, None)
        if failures:
            exception_failures = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exception_failures) == len(failures):
                raise ExceptionGroup(
                    "Lean exploration sessions failed to close", exception_failures
                )
            raise BaseExceptionGroup(
                "Lean exploration sessions failed to close", failures
            )
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
        policy = self._policy or LeanReplPolicy(
            max_rss_kb=(
                _DEFAULT_MATHLIB_MAX_RSS_KB
                if environment is LeanEnvironment.MATHLIB
                else _DEFAULT_CORE_MAX_RSS_KB
            )
        )
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


def _parse_command_response(
    response: Mapping[str, Any],
) -> LeanReplCommandResponse | LeanReplErrorResponse:
    if set(response) == {"message"}:
        try:
            return LeanReplErrorResponse.model_validate(response)
        except ValidationError as exc:
            raise RuntimeError("Lean REPL returned a malformed error response") from exc
    try:
        return LeanReplCommandResponse.model_validate(response)
    except ValidationError as exc:
        raise RuntimeError("Lean REPL returned a malformed command response") from exc


def _parse_proof_response(response: Mapping[str, Any]) -> LeanReplProofResponse:
    if set(response) == {"message"}:
        try:
            return LeanReplErrorResponse.model_validate(response)
        except ValidationError as exc:
            raise RuntimeError("Lean REPL returned a malformed error response") from exc
    try:
        return LeanReplProofStepResponse.model_validate(response)
    except ValidationError as exc:
        raise RuntimeError("Lean REPL returned a malformed proof response") from exc


def _parse_pickle_response(response: Mapping[str, Any]) -> LeanReplPickleResponse:
    if set(response) == {"message"}:
        try:
            return LeanReplErrorResponse.model_validate(response)
        except ValidationError as exc:
            raise RuntimeError("Lean REPL returned a malformed error response") from exc
    try:
        return LeanReplPickleProofStateResponse.model_validate(response)
    except ValidationError as exc:
        raise RuntimeError("Lean REPL returned a malformed pickle response") from exc


def _response_errors(response: LeanReplResponse) -> tuple[str, ...]:
    if isinstance(response, LeanReplErrorResponse):
        return (response.message,)
    errors = [
        message.data for message in response.messages if message.severity == "error"
    ]
    if isinstance(response, LeanReplProofStepResponse):
        status = response.proof_status.casefold()
        if (
            status in {"error", "failed", "failure"} or status.startswith("error:")
        ) and not errors:
            errors.append(
                f"Lean tactic returned proof status {response.proof_status!r}"
            )
    return tuple(errors)


__all__ = ["LeanExplorationReplRuntime", "LeanReplPolicy", "PersistentLeanRepl"]
