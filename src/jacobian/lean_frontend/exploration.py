"""Pinned, replayable exploratory Lean capabilities."""

from __future__ import annotations

import hashlib
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.capabilities import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import (
    LeanPremiseCandidate,
    LeanPremiseRetrievalArtifact,
    LeanPremiseRetrievalOutput,
    LeanPremiseRetrievalRequest,
    LeanProofStateArtifact,
    LeanProofStateOutput,
    LeanProofStateRequest,
    LeanProofStateTransitionArtifact,
    LeanProofSuccessorState,
    LeanTacticDiagnostic,
    LeanTypedGoal,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.references import LeanCheckerInstallation
from jacobian.schema_registry import SchemaRegistry
from jacobian.store import ArtifactStore, StoreError

_FORBIDDEN = re.compile(
    r"\b(?:admit|axiom|elab|import|macro|native_decide|opaque|run_tac|"
    r"set_option|sorry|syntax|unsafe)\b|#",
    re.IGNORECASE,
)
_SUGGESTION = re.compile(
    r"Try this:\s*\n\s*\[apply\]\s*(?P<tactic>[^\r\n]+)",
)
_DECLARATION = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_']+)+\b")
_RESOURCE_POLL_SECONDS = 0.1


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
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
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
        self._responses = queue.Queue()
        responses = self._responses
        self._process = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        self._started_at = time.monotonic()
        self._requests = 0
        self._base_env = None
        self._reader_thread = threading.Thread(
            target=self._read_responses,
            args=(self._process, responses),
            name="jacobian-lean-repl-reader",
            daemon=True,
        )
        self._reader_thread.start()
        if self._base_command is not None:
            response = self._exchange({"cmd": self._base_command})
            base_env = response.get("env")
            if not isinstance(base_env, int):
                self._stop_process()
                raise RuntimeError("Lean REPL did not return a base environment")
            self._base_env = base_env

    def _expired(self) -> bool:
        assert self._process is not None
        if self._process.poll() is not None:
            return True
        if self._requests >= self._policy.max_requests:
            return True
        if time.monotonic() - self._started_at >= self._policy.max_age_seconds:
            return True
        rss_kb = _process_rss_kb(self._process.pid)
        return self._policy.max_rss_kb > 0 and rss_kb > self._policy.max_rss_kb

    def _exchange(self, request: Mapping[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Lean REPL is unavailable")
        try:
            process.stdin.write(canonicalize_json(request).decode("utf-8") + "\n\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._stop_process()
            raise RuntimeError("Lean REPL stopped before receiving a request") from exc
        deadline = time.monotonic() + self._policy.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_process()
                raise RuntimeError("Lean REPL timed out")
            try:
                response = self._responses.get(
                    timeout=min(_RESOURCE_POLL_SECONDS, remaining)
                )
                break
            except queue.Empty:
                rss_kb = _process_rss_kb(process.pid)
                if self._policy.max_rss_kb > 0 and rss_kb > self._policy.max_rss_kb:
                    self._stop_process()
                    raise RuntimeError("Lean REPL exceeded its memory limit") from None
        if isinstance(response, BaseException):
            self._stop_process()
            raise RuntimeError(
                "Lean REPL stopped before returning a result"
            ) from response
        return response

    def _read_responses(
        self,
        process: subprocess.Popen[str],
        responses: queue.Queue[dict[str, Any] | BaseException],
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            responses.put(RuntimeError("Lean REPL stdout is unavailable"))
            return
        block: list[str] = []
        try:
            for line in stdout:
                if line.strip():
                    block.append(line)
                    continue
                if not block:
                    continue
                value = loads_strict_json("".join(block))
                if not isinstance(value, dict):
                    raise RuntimeError("Lean REPL returned a non-object response")
                responses.put(value)
                block = []
            if block:
                raise RuntimeError("Lean REPL returned an unterminated response")
            responses.put(RuntimeError("Lean REPL exited"))
        except (CanonicalizationError, OSError, RuntimeError) as exc:
            responses.put(exc)

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._base_env = None
        if process is not None:
            if process.stdin is not None:
                with suppress(OSError):
                    process.stdin.close()
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        reader = self._reader_thread
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)
            if reader.is_alive():
                raise RuntimeError("Lean REPL reader did not quiesce")
        self._reader_thread = None


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


def _process_rss_kb(pid: int) -> int:
    """Read current Linux RSS; return zero where procfs is unavailable."""

    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0
    match = re.search(r"^VmRSS:\s+(?P<rss>\d+)\s+kB$", status, re.MULTILINE)
    return int(match.group("rss")) if match else 0


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


@dataclass(frozen=True, slots=True)
class LeanExplorationInstallation:
    semantics_uri: str
    state_schema_uri: str
    transition_schema_uri: str
    retrieval_schema_uri: str
    repl: LeanExplorationReplRuntime


@dataclass(frozen=True, slots=True)
class _Resources:
    store: ArtifactStore
    artifacts: ArtifactService
    semantics_uri: str
    state_schema_uri: str
    transition_schema_uri: str
    retrieval_schema_uri: str
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation]
    runtime: Path
    provider_runtime: CapabilityProviderRuntime
    repl: LeanExplorationReplRuntime


def install_lean_exploration_capabilities(
    store: ArtifactStore,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation],
    provider_runtime: CapabilityProviderRuntime,
) -> tuple[
    tuple[LeanProofStateAdapter, LeanPremiseRetrievalAdapter],
    LeanExplorationInstallation,
]:
    """Register replayable exploratory Lean adapters."""

    mathlib = installations[LeanEnvironment.MATHLIB]
    core = installations[LeanEnvironment.CORE]
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-exploration",
        version="1",
        definition={
            "description": (
                "immutable replayable Lean proof states, one-step tactic "
                "transitions, and premise suggestions"
            ),
            "lean_version": core.lean_version,
            "lean_commit": core.lean_commit,
            "mathlib_commit": mathlib.mathlib_commit,
            "state_expiry": "immutable artifacts do not expire",
            "verification": "none; completed source must pass lean.check",
        },
    )
    state_schema_uri = schemas.register(
        name="jacobian.lean4-proof-state",
        version="1",
        schema=LeanProofStateArtifact.model_json_schema(),
    )
    transition_schema_uri = schemas.register(
        name="jacobian.lean4-proof-state-transition",
        version="2",
        schema=LeanProofStateTransitionArtifact.model_json_schema(),
    )
    retrieval_schema_uri = schemas.register(
        name="jacobian.lean4-premise-retrieval",
        version="2",
        schema=LeanPremiseRetrievalArtifact.model_json_schema(),
    )
    runtime = Path(__file__).resolve().parents[3] / "lean"
    repl = LeanExplorationReplRuntime(runtime, installations)
    resources = _Resources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        state_schema_uri=state_schema_uri,
        transition_schema_uri=transition_schema_uri,
        retrieval_schema_uri=retrieval_schema_uri,
        installations=installations,
        runtime=runtime,
        provider_runtime=provider_runtime,
        repl=repl,
    )
    return (
        (
            LeanProofStateAdapter(resources),
            LeanPremiseRetrievalAdapter(resources),
        ),
        LeanExplorationInstallation(
            semantics_uri=semantics_uri,
            state_schema_uri=state_schema_uri,
            transition_schema_uri=transition_schema_uri,
            retrieval_schema_uri=retrieval_schema_uri,
            repl=repl,
        ),
    )


class LeanProofStateAdapter:
    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof_state.apply_tactic",
            version="2",
            title="Apply one Lean tactic to a replayable proof state",
            description=(
                "Reconstruct and validate an immutable proof state in a clean "
                "Lean process, apply one tactic, and return every durable "
                "successor state or structured rejection diagnostics."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanProofStateRequest.model_json_schema(),
            output_schema=LeanProofStateOutput.model_json_schema(),
            tags=("lean", "proof-state", "tactic", "exploration"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="close_true_with_trivial",
                    description=(
                        "Apply trivial to a replayable proof state for True; "
                        "a completed transition still requires lean.check."
                    ),
                    mode=CapabilityMode.EXPLORE,
                    input=LeanProofStateRequest.model_validate(
                        {
                            "environment": "CORE",
                            "statement": "True",
                            "tactic": "trivial",
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanProofStateRequest.model_validate(request.input)
            if validated.statement is not None:
                _validate_source_parts(
                    validated.statement,
                    (*validated.proof_prefix, validated.tactic),
                )
            else:
                _validate_source_parts("True", (validated.tactic,))
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_TRANSITION_REQUEST",
                    stage="request_validation",
                    message="The Lean statement or tactic sequence is invalid.",
                    hint=(
                        "Use one proposition and bounded tactic bodies without "
                        "commands, imports, declarations, sorry, or run_tac."
                    ),
                )
            ) from exc
        started = time.monotonic()
        installation = self.resources.installations[validated.environment]
        environment_digest = _environment_digest(
            validated.environment,
            installation,
        )
        if validated.state_uri is None:
            assert validated.statement is not None
            statement = validated.statement
            proof_prefix = validated.proof_prefix
            bound_state = None
        else:
            bound_state = self._load_bound_state(
                validated.state_uri,
                expected_environment=validated.environment,
                expected_environment_digest=environment_digest,
            )
            if bound_state.completed:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="LEAN_PROOF_STATE_COMPLETED",
                        stage="state_validation",
                        message="The supplied proof state has no remaining goals.",
                        hint=(
                            "Send the complete statement and proof to lean.check; "
                            "no further tactic transition is applicable."
                        ),
                    )
                )
            if len(bound_state.tactic_prefix) >= 64:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="LEAN_PROOF_STATE_PREFIX_LIMIT",
                        stage="state_validation",
                        message="The replayable proof state reached the 64-tactic limit.",
                        hint=(
                            "Submit a complete proof to lean.check or begin a new "
                            "bounded exploration."
                        ),
                    )
                )
            statement = bound_state.statement
            proof_prefix = bound_state.tactic_prefix
            _validate_source_parts(statement, (*proof_prefix, validated.tactic))
        command = _proof_state_command(
            statement=statement,
            proof_prefix=proof_prefix,
        )
        with tempfile.TemporaryDirectory(prefix="jacobian-lean-proof-state-") as root:
            pickle_path = Path(root) / "proof-state.pickle"
            responses = self.resources.repl.execute_clean(
                command=command,
                tactic=validated.tactic,
                environment=validated.environment,
                pickle_path=pickle_path,
            )
            command_response, validation_response, tactic_response = responses
            reconstruction_errors = (
                *_response_errors(command_response),
                *_response_errors(validation_response),
            )
            if reconstruction_errors:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="LEAN_STATE_RECONSTRUCTION_FAILED",
                        stage="state_reconstruction",
                        message=(
                            "Lean could not reconstruct the bound proof state: "
                            f"{reconstruction_errors[0][:500]}"
                        ),
                        hint=(
                            "Recreate the state from the current pinned environment; "
                            "a reconstruction failure is not a proof conclusion."
                        ),
                    )
                )
            tactic_errors = _response_errors(tactic_response)
            accepted = not tactic_errors
            typed_goals: tuple[LeanTypedGoal, ...] = ()
            if accepted:
                try:
                    typed_goals = _extract_typed_goals(
                        self.resources,
                        pickle_path=pickle_path,
                        request=validated,
                    )
                except RuntimeError as exc:
                    raise CapabilityInvocationError(
                        CapabilityDiagnostic(
                            code="LEAN_PROOF_STATE_EXTRACTION_FAILED",
                            stage="proof_state_extraction",
                            message=(
                                "Lean could not produce the bounded typed successor "
                                "proof state."
                            ),
                            hint=(
                                "Retry with smaller goal/context bounds or verify "
                                "that the pinned proof-state helper is installed."
                            ),
                        )
                    ) from exc
        replayed_goals = _normalized_response_goals(validation_response)
        if bound_state is not None and replayed_goals != bound_state.normalized_goals:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="STALE_LEAN_PROOF_STATE",
                    stage="state_validation",
                    message=(
                        "The clean replay produced goals different from the "
                        "state artifact."
                    ),
                    hint=(
                        "Recreate the state under the current source and "
                        "environment before applying another tactic."
                    ),
                )
            )
        if bound_state is None:
            input_state_payload = _state_payload(
                environment=validated.environment,
                environment_digest=environment_digest,
                statement=statement,
                tactic_prefix=proof_prefix,
                normalized_goals=replayed_goals,
                installation=installation,
            )
            input_state_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.state_schema_uri,
                semantics_uri=self.resources.semantics_uri,
                payload=input_state_payload.model_dump(mode="json"),
                summary="replayable immutable Lean proof state",
            )
            input_state_uri = input_state_artifact.artifact_uri
            input_state = input_state_payload
        else:
            assert validated.state_uri is not None
            input_state_uri = validated.state_uri
            input_state = bound_state

        diagnostics = _tactic_diagnostics(responses)
        successor_states: tuple[LeanProofSuccessorState, ...] = ()
        successor_artifact_uris: tuple[str, ...] = ()
        goals: tuple[str, ...] = ()
        completed = False
        if accepted:
            goals = _normalized_response_goals(tactic_response)
            proof_status = tactic_response.get("proofStatus")
            if (proof_status == "Completed") != (len(goals) == 0):
                raise RuntimeError(
                    "Lean REPL returned inconsistent completion and goals"
                )
            successor_payload = _state_payload(
                environment=validated.environment,
                environment_digest=environment_digest,
                statement=statement,
                tactic_prefix=(*proof_prefix, validated.tactic),
                normalized_goals=goals,
                installation=installation,
            )
            successor_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.state_schema_uri,
                semantics_uri=self.resources.semantics_uri,
                payload=successor_payload.model_dump(mode="json"),
                parents=(input_state_uri,),
                summary="successor immutable Lean proof state",
            )
            completed = successor_payload.completed
            successor_states = (
                LeanProofSuccessorState(
                    state_uri=successor_artifact.artifact_uri,
                    state_digest=successor_payload.state_digest,
                    normalized_goals=goals,
                    completed=completed,
                ),
            )
            successor_artifact_uris = (successor_artifact.artifact_uri,)

        replay_source = "\n  ".join((*proof_prefix, validated.tactic))
        transition_source_digest = _source_digest(
            statement,
            (*proof_prefix, validated.tactic),
        )
        messages = tuple(diagnostic.message for diagnostic in diagnostics)
        artifact_payload = LeanProofStateTransitionArtifact(
            environment=validated.environment,
            environment_digest=environment_digest,
            source_digest=transition_source_digest,
            statement=statement,
            proof_prefix=proof_prefix,
            tactic=validated.tactic,
            input_state_uri=input_state_uri,
            input_state_digest=input_state.state_digest,
            replay_source=replay_source,
            goals=goals,
            typed_goals=typed_goals,
            goal_count=len(goals),
            successor_states=successor_states,
            accepted=accepted,
            completed=completed,
            messages=messages,
            diagnostics=diagnostics,
            lean_version=installation.lean_version,
            lean_commit=installation.lean_commit,
            mathlib_commit=installation.mathlib_commit,
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.transition_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            parents=(input_state_uri, *successor_artifact_uris),
            summary=(
                "accepted replayable Lean tactic transition"
                if accepted
                else "rejected replayable Lean tactic transition"
            ),
        )
        output = LeanProofStateOutput(
            **artifact_payload.model_dump(mode="python"),
            transition_uri=artifact.artifact_uri,
        )
        artifact_uris = (
            input_state_uri,
            *successor_artifact_uris,
            artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
                detail=(
                    None
                    if accepted
                    else "Lean rejected the tactic; no successor state was created"
                ),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "one tactic applied after clean replay and state validation"
                ),
                parameters={
                    "environment": validated.environment.value,
                    "statement": statement,
                    "input_state_digest": input_state.state_digest,
                    "environment_digest": environment_digest,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "Lean returned the complete successor-state list for this "
                    "single tactic application"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "a clean pinned Lean process reconstructed the bound state "
                    "and computed one transition; only lean.check can verify a "
                    "completed theorem"
                ),
            ),
            artifact_uris=artifact_uris,
        )

    def _load_bound_state(
        self,
        state_uri: str,
        *,
        expected_environment: LeanEnvironment,
        expected_environment_digest: str,
    ) -> LeanProofStateArtifact:
        try:
            stored = self.resources.store.get(state_uri)
            if (
                stored.manifest.schema_uri != self.resources.state_schema_uri
                or stored.manifest.semantics_uri != self.resources.semantics_uri
            ):
                raise ValueError("artifact is not a Lean proof state")
            state = LeanProofStateArtifact.model_validate(stored.payload)
        except (StoreError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_PROOF_STATE",
                    stage="state_loading",
                    message="The supplied state artifact is unavailable or invalid.",
                    hint="Use a state URI returned by this capability.",
                )
            ) from exc
        if (
            state.environment is not expected_environment
            or state.environment_digest != expected_environment_digest
            or state.source_digest
            != _source_digest(state.statement, state.tactic_prefix)
            or state.state_digest != _state_digest_payload(state)
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="STALE_LEAN_PROOF_STATE",
                    stage="state_validation",
                    message=(
                        "The proof state no longer matches its source or the "
                        "current pinned Lean environment."
                    ),
                    hint="Recreate the proof state under the current environment.",
                )
            )
        return state


class LeanPremiseRetrievalAdapter:
    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.retrieve.premises",
            version="2",
            title="Retrieve Lean premises",
            description=(
                "Ask pinned Mathlib exact? for bounded candidate tactics at one "
                "explicit proof prefix; an empty result is non-exhaustive."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanPremiseRetrievalRequest.model_json_schema(),
            output_schema=LeanPremiseRetrievalOutput.model_json_schema(),
            tags=("lean", "mathlib", "premise-retrieval", "exploration"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanPremiseRetrievalRequest.model_validate(request.input)
            _validate_source_parts(validated.statement, validated.proof_prefix)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_RETRIEVAL_REQUEST",
                    stage="request_validation",
                    message="The Lean premise-retrieval request is invalid.",
                )
            ) from exc
        started = time.monotonic()
        environment = LeanEnvironment.MATHLIB
        installation = self.resources.installations[environment]
        command = _proof_state_command(
            statement=validated.statement,
            proof_prefix=validated.proof_prefix,
        )
        command_response, tactic_response = _run_repl(
            self.resources,
            command=command,
            tactic="exact?",
            environment=environment,
        )
        command_errors = _response_errors(command_response)
        tactic_errors = _response_errors(tactic_response)
        if command_errors:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_RETRIEVAL_FAILED",
                    stage="premise_retrieval",
                    message=(
                        "Lean rejected the statement or proof prefix: "
                        f"{command_errors[0][:500]}"
                    ),
                    hint="Correct the statement or proof prefix and retry.",
                )
            )
        diagnostics = "\n".join(_response_messages(tactic_response))
        suggestions = [
            match.group("tactic").strip() for match in _SUGGESTION.finditer(diagnostics)
        ][: validated.limit]
        if tactic_errors and not any(
            "`exact?` could not close the goal" in error for error in tactic_errors
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_RETRIEVAL_FAILED",
                    stage="premise_retrieval",
                    message=f"Mathlib exact? failed: {tactic_errors[0][:500]}",
                    hint="Correct the statement or proof prefix and retry.",
                )
            )
        candidates = tuple(
            LeanPremiseCandidate(
                rank=index,
                tactic=suggestion,
                declaration_names=tuple(sorted(set(_DECLARATION.findall(suggestion)))),
                tactic_replayed=(
                    index == 1 and tactic_response.get("proofStatus") == "Completed"
                ),
            )
            for index, suggestion in enumerate(suggestions, start=1)
        )
        artifact_payload = LeanPremiseRetrievalArtifact(
            statement=validated.statement,
            proof_prefix=validated.proof_prefix,
            candidates=candidates,
            goal_context_digest=(
                "sha256:"
                + hashlib.sha256(
                    canonicalize_json(
                        {
                            "environment": "MATHLIB",
                            "statement": validated.statement,
                            "proof_prefix": list(validated.proof_prefix),
                        }
                    )
                ).hexdigest()
            ),
            lean_version=installation.lean_version,
            lean_commit=installation.lean_commit,
            mathlib_commit=installation.mathlib_commit or "",
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.retrieval_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary="non-exhaustive pinned Mathlib premise suggestions",
        )
        output = LeanPremiseRetrievalOutput(
            **artifact_payload.model_dump(mode="python"),
            retrieval_uri=artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one explicit Lean goal under pinned Mathlib exact?",
                parameters={
                    "environment": "MATHLIB",
                    "statement": validated.statement,
                    "limit": validated.limit,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.PARTIAL,
                basis=(
                    "Mathlib exact? suggestions are bounded and non-exhaustive; "
                    "no suggestion is not a proof of absence"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "candidate tactics were emitted by pinned Mathlib exact?; "
                    "they remain unverified until lean.check accepts exact source"
                ),
            ),
            artifact_uris=(artifact.artifact_uri,),
        )


def _validate_source_parts(statement: str, tactics: tuple[str, ...]) -> None:
    if "\n" in statement or "\r" in statement or ":=" in statement:
        raise ValueError("statement must be one Lean expression")
    if _FORBIDDEN.search(statement):
        raise ValueError("statement contains a forbidden command")
    for tactic in tactics:
        if "\x00" in tactic or _FORBIDDEN.search(tactic):
            raise ValueError("tactic contains a forbidden command")


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


def _normalized_response_goals(response: Mapping[str, Any]) -> tuple[str, ...]:
    goals_value = response.get("goals", [])
    if not isinstance(goals_value, list) or any(
        not isinstance(goal, str) for goal in goals_value
    ):
        raise RuntimeError("Lean REPL returned malformed goals")
    return tuple(_normalize_goal(goal) for goal in goals_value)


def _normalize_goal(goal: str) -> str:
    lines = goal.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def _tactic_diagnostics(
    responses: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[LeanTacticDiagnostic, ...]:
    diagnostics: list[LeanTacticDiagnostic] = []
    for response in responses:
        message = response.get("message")
        if isinstance(message, str):
            diagnostics.append(LeanTacticDiagnostic(severity="ERROR", message=message))
        structured = response.get("messages")
        if not isinstance(structured, list):
            continue
        for item in structured:
            if not isinstance(item, Mapping):
                continue
            data = item.get("data")
            if not isinstance(data, str):
                continue
            raw_severity = item.get("severity")
            severity = (
                "ERROR"
                if raw_severity == "error"
                else ("WARNING" if raw_severity == "warning" else "INFO")
            )
            diagnostics.append(
                LeanTacticDiagnostic.model_validate(
                    {"severity": severity, "message": data}
                )
            )
    return tuple(diagnostics)


def _run_repl(
    resources: _Resources,
    *,
    command: str,
    tactic: str,
    environment: LeanEnvironment,
    pickle_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return resources.repl.execute(
        command=command,
        tactic=tactic,
        environment=environment,
        pickle_path=pickle_path,
    )


def _extract_typed_goals(
    resources: _Resources,
    *,
    pickle_path: Path,
    request: LeanProofStateRequest,
) -> tuple[LeanTypedGoal, ...]:
    query_path = pickle_path.with_name("typed-goal-query.json")
    request_id = uuid.uuid4().hex
    query_path.write_bytes(
        canonicalize_json(
            {
                "pickle_path": str(pickle_path),
                "request_id": request_id,
                "max_goals": request.max_goals,
                "max_local_declarations": request.max_local_declarations,
                "max_rendered_bytes": request.max_rendered_bytes,
            }
        )
    )
    environment = dict(os.environ)
    environment["JACOBIAN_LEAN_PROOF_STATE_QUERY"] = str(query_path)
    helper = resources.runtime / ".lake" / "build" / "bin" / "jacobian_lean_proof_state"
    if not helper.is_file():
        raise RuntimeError(
            "the pinned typed proof-state helper is unavailable; "
            "run `lake build jacobian_lean_proof_state` in lean/"
        )
    elan = shutil.which("elan")
    if elan is None:
        raise RuntimeError("elan is unavailable")
    installation = resources.installations[request.environment]
    try:
        completed = subprocess.run(
            [
                elan,
                "run",
                f"leanprover/lean4:v{installation.lean_version}",
                "lake",
                "env",
                helper,
            ],
            cwd=resources.runtime,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    marker = "JACOBIAN_PROOF_STATE_RESULT "
    try:
        lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    responses = [line for line in lines if line.startswith(marker)]
    if completed.returncode != 0 or len(responses) != 1:
        raise RuntimeError("Lean typed proof-state extraction failed")
    try:
        envelope = loads_strict_json(responses[0].removeprefix(marker))
    except CanonicalizationError as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    if (
        not isinstance(envelope, dict)
        or envelope.get("request_id") != request_id
        or not isinstance(envelope.get("payload"), dict)
    ):
        raise RuntimeError("Lean typed proof-state extraction returned invalid JSON")
    payload = envelope["payload"]
    if payload.get("expression_serialization") != "LEAN_PRETTY_PRINTED_EXPR":
        raise RuntimeError("Lean typed proof-state serialization is unsupported")
    try:
        return tuple(
            LeanTypedGoal.model_validate(goal) for goal in payload["typed_goals"]
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise RuntimeError(
            "Lean typed proof-state extraction returned invalid goals"
        ) from exc


def _response_messages(response: Mapping[str, Any]) -> tuple[str, ...]:
    messages: list[str] = []
    message = response.get("message")
    if isinstance(message, str):
        messages.append(message)
    structured = response.get("messages")
    if isinstance(structured, list):
        for item in structured:
            if not isinstance(item, Mapping):
                continue
            data = item.get("data")
            if isinstance(data, str):
                messages.append(data)
    return tuple(messages)


def _response_errors(response: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    message = response.get("message")
    if isinstance(message, str):
        errors.append(message)
    structured = response.get("messages")
    if isinstance(structured, list):
        for item in structured:
            if not isinstance(item, Mapping) or item.get("severity") != "error":
                continue
            data = item.get("data")
            if isinstance(data, str):
                errors.append(data)
    return tuple(errors)


def _runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
