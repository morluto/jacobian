"""Atomic Lean statement proposal and comparison adapters.

Two domain-atomic capabilities, each producing exactly one inspectable
artifact:

* ``lean.statement.propose`` — either type-check one proposed Lean statement
  against an informal claim or directly elaborate one proposition. Returns
  durable environment-bound elaboration details; does NOT certify truth or
  that a formal statement matches an informal claim.

* ``lean.statement.compare`` — compare two Lean statements syntactically
  and by axiom set. Fail-closed: never claims semantic equivalence; if
  elaboration cannot be checked, reports that honestly.

When the Lean binary is not on PATH, each adapter returns honest
unavailable/diagnostic behavior rather than a silent success.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_statement import (
    LeanElaborationDiagnostic,
    LeanElaborationOption,
    LeanStatementComparisonArtifact,
    LeanStatementComparisonOutput,
    LeanStatementComparisonRequest,
    LeanStatementProposalArtifact,
    LeanStatementProposalOutput,
    LeanStatementProposalRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.providers.lean_runtime import (
    LeanRuntimeIdentityError,
    lean_frontend_provider_runtime,
    lean_semantic_runtime_digest,
    require_lean_semantic_runtime_identity,
)
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.worker_environment import worker_environment

# ---------------------------------------------------------------------------
# Security: block dangerous Lean commands in user-supplied text.
# Statements block sorry/admit (the statement is the claim, not a proof).
# Proofs block only structural commands that could escape the proof block.
# ---------------------------------------------------------------------------

_FORBIDDEN_STATEMENT = re.compile(
    r"\b(?:admit|axiom|elab|import|macro|native_decide|opaque|run_tac|"
    r"set_option|sorry|syntax|unsafe)\b|#",
    re.IGNORECASE,
)

_FORBIDDEN_PROOF = re.compile(
    r"\b(?:import|macro|syntax|unsafe|set_option|run_tac|native_decide)\b|#",
    re.IGNORECASE,
)

_ELAPSED_TIMEOUT_SECONDS = 30


def _lean_process_environment(lean_bin: str) -> dict[str, str]:
    lean_bin_dir = str(Path(lean_bin).resolve().parent)
    return worker_environment(path_prefix=lean_bin_dir)


class _LeanUnavailableError(RuntimeError):
    """Lean is not available on PATH or timed out during elaboration."""


# ---------------------------------------------------------------------------
# Elaboration probe — bounded Lean execution via the process_policy gateway.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ElaborationResult:
    elaborates: bool
    sorry_count: int
    messages: tuple[str, ...]
    errors: tuple[str, ...]
    elaborated_expression: str | None = None
    used_imports: tuple[str, ...] = ()
    used_declarations: tuple[str, ...] = ()
    options: tuple[LeanElaborationOption, ...] = ()


_DIRECT_ELABORATION_IMPORTS = ("Init.Prelude",)
_DIRECT_ELABORATION_OPTIONS = (
    LeanElaborationOption(name="pp.all", value="true"),
    LeanElaborationOption(name="pp.explicit", value="true"),
    LeanElaborationOption(name="pp.universes", value="true"),
)
_LEAN_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z0-9_']+)*\b")
_LEAN_MESSAGE_KIND = re.compile(
    r"(?:^|:\s*)(error|warning|info)(?:\([^)]*\))?:",
    re.IGNORECASE,
)
_LEAN_KEYWORDS = frozenset(
    {
        "Prop",
        "Sort",
        "Type",
        "false",
        "fun",
        "let",
        "match",
        "true",
    }
)


def _lean_version_info(
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
) -> tuple[str, str]:
    """Return (version, commit) from ``lean --version``; ``unknown`` on failure."""

    try:
        _require_current_runtime(provider_runtime)
        lean_bin = executable or _lean_executable()
    except _LeanUnavailableError:
        return ("unknown", "unknown")
    try:
        environment = _lean_process_environment(lean_bin)
        result = execute_process(
            ProcessRequest(
                executable=lean_bin,
                arguments=("--version",),
                environment=environment,
                cwd=str(Path.cwd()),
                timeout_seconds=10.0,
                stdin_bytes=b"",
                stdout_limit_bytes=4096,
                stderr_limit_bytes=4096,
            )
        )
    except OSError:
        return ("unknown", "unknown")
    if result.termination is not ProcessTermination.EXITED:
        return ("unknown", "unknown")
    output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode(
        "utf-8", errors="replace"
    )
    version_match = re.search(r"version\s+([^\s,]+)", output)
    commit_match = re.search(r"commit\s+([^\s,)]+)", output)
    return (
        version_match.group(1) if version_match else "unknown",
        commit_match.group(1) if commit_match else "unknown",
    )


def _elaborate_statement(
    statement: str,
    *,
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
    timeout_seconds: int = _ELAPSED_TIMEOUT_SECONDS,
) -> _ElaborationResult:
    """Elaborate ``example : {statement} := by sorry`` via the ``lean`` binary."""

    lean_bin = executable or _lean_executable()
    _validate_statement(statement)
    source = f"example : {statement} := by sorry"
    return _run_lean_source(
        source,
        executable=lean_bin,
        provider_runtime=provider_runtime,
        timeout_seconds=timeout_seconds,
    )


def _elaborate_proposition(
    statement: str,
    *,
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
    timeout_seconds: int = _ELAPSED_TIMEOUT_SECONDS,
) -> _ElaborationResult:
    """Elaborate one expression against expected type ``Prop``."""

    lean_bin = executable or _lean_executable()
    _validate_statement(statement)
    source = "\n".join(
        (
            "set_option pp.all true in",
            "set_option pp.explicit true in",
            "set_option pp.universes true in",
            f"#check ({statement} : Prop)",
        )
    )
    output = _execute_lean_source(
        source,
        executable=lean_bin,
        provider_runtime=provider_runtime,
        timeout_seconds=timeout_seconds,
    )
    messages = tuple(_parse_lean_messages(output))
    errors = tuple(
        message for message in messages if _lean_message_severity(message) == "ERROR"
    )
    expression = None if errors else _parse_elaborated_expression(output)
    if not errors and expression is None:
        errors = ("error: Lean did not emit the elaborated proposition",)
        messages = (*messages, *errors)
    declarations = (
        tuple(
            sorted(
                name
                for name in set(_LEAN_NAME.findall(expression))
                if name not in _LEAN_KEYWORDS
            )
        )
        if expression is not None
        else ()
    )
    return _ElaborationResult(
        elaborates=expression is not None,
        sorry_count=0,
        messages=messages,
        errors=errors,
        elaborated_expression=expression,
        used_imports=_DIRECT_ELABORATION_IMPORTS,
        used_declarations=declarations,
        options=_DIRECT_ELABORATION_OPTIONS,
    )


def _check_proof(
    statement: str,
    proof: str,
    *,
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
    timeout_seconds: int = _ELAPSED_TIMEOUT_SECONDS,
) -> _ElaborationResult:
    """Check whether ``example : {statement} := by {proof}`` elaborates."""

    lean_bin = executable or _lean_executable()
    _validate_statement(statement)
    _validate_proof(proof)
    source = f"example : {statement} := by\n{_indent_proof(proof)}"
    return _run_lean_source(
        source,
        executable=lean_bin,
        provider_runtime=provider_runtime,
        timeout_seconds=timeout_seconds,
    )


def _run_lean_source(
    source: str,
    *,
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
    timeout_seconds: int,
) -> _ElaborationResult:
    output = _execute_lean_source(
        source,
        executable=executable,
        provider_runtime=provider_runtime,
        timeout_seconds=timeout_seconds,
    )
    messages = _parse_lean_messages(output)
    errors = tuple(
        message for message in messages if _lean_message_severity(message) == "ERROR"
    )
    elaborates = len(errors) == 0
    return _ElaborationResult(
        elaborates=elaborates,
        sorry_count=1 if elaborates else 0,
        messages=tuple(messages),
        errors=errors,
    )


def _execute_lean_source(
    source: str,
    *,
    executable: str | None = None,
    provider_runtime: CapabilityProviderRuntime | None = None,
    timeout_seconds: int,
) -> str:
    _require_current_runtime(provider_runtime)
    temp_path: str | None = None
    result: ProcessResult | None = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".lean")
        try:
            handle = os.fdopen(fd, "w")
        except OSError:
            os.close(fd)
            raise
        with handle:
            handle.write(source)
        lean_bin = executable or _lean_executable()
        environment = _lean_process_environment(lean_bin)
        result = execute_process(
            ProcessRequest(
                executable=lean_bin,
                arguments=(temp_path,),
                environment=environment,
                cwd=str(Path(temp_path).parent),
                timeout_seconds=float(timeout_seconds),
                stdin_bytes=b"",
                stdout_limit_bytes=64 * 1024,
                stderr_limit_bytes=64 * 1024,
            )
        )
    except OSError as exc:
        raise _LeanUnavailableError(
            f"The pinned Lean executable could not run: {exc}"
        ) from exc
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    if result.termination is ProcessTermination.TIMED_OUT:
        raise _LeanUnavailableError(f"lean timed out after {timeout_seconds}s")
    if result.termination is ProcessTermination.START_FAILED:
        raise _LeanUnavailableError(
            "The pinned Lean executable could not run: start failed"
        )
    return result.stdout.decode("utf-8", errors="replace") + result.stderr.decode(
        "utf-8", errors="replace"
    )


def _lean_executable() -> str:
    """Resolve the exact pinned Lean executable used by the availability probe."""

    from jacobian_checkers import lean4

    try:
        executable, _ = lean4.inspect_runtime(require_mathlib=False)
    except (OSError, RuntimeError) as exc:
        raise _LeanUnavailableError(
            str(exc)
            or f"The pinned Lean {lean4.LEAN_VERSION} executable is unavailable."
        ) from exc
    return str(executable)


def _require_current_runtime(
    provider_runtime: CapabilityProviderRuntime | None,
) -> None:
    if provider_runtime is None:
        return
    from jacobian.provider_runtime import (
        ProviderRuntimeError,
        require_provider_runtime_unchanged,
    )

    try:
        require_provider_runtime_unchanged(provider_runtime)
        if "semantic_runtime" in provider_runtime.configuration:
            require_lean_semantic_runtime_identity(provider_runtime)
    except (
        LeanRuntimeIdentityError,
        OSError,
        ProviderRuntimeError,
        ValidationError,
    ) as exc:
        raise _LeanUnavailableError(
            "The pinned Lean executable identity changed or became unavailable."
        ) from exc


def _parse_elaborated_expression(output: str) -> str | None:
    diagnostic_match = re.search(
        r"\binfo:\s*(.+?)\s*:\s*Prop(?:\s|$)", output, re.DOTALL
    )
    if diagnostic_match is not None:
        return " ".join(diagnostic_match.group(1).split())
    # Lean's command-line frontend emits successful ``#check`` output as the
    # bare ``<expression> : Prop`` line.  It does not add an ``info:`` prefix.
    plain_match = re.fullmatch(r"\s*(.+?)\s*:\s*Prop\s*", output, re.DOTALL)
    if plain_match is None:
        return None
    return " ".join(plain_match.group(1).split())


def _parse_lean_messages(output: str) -> list[str]:
    messages: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(
            r"^.*:\d+:\d+:\s*(error|warning|info)(?:\([^)]*\))?:",
            stripped,
            re.IGNORECASE,
        ) or re.match(
            r"^(error|warning|info)(?:\([^)]*\))?:",
            stripped,
            re.IGNORECASE,
        ):
            messages.append(stripped)
    return messages


def _lean_message_severity(
    message: str,
) -> Literal["ERROR", "WARNING", "INFO"]:
    match = _LEAN_MESSAGE_KIND.search(message)
    if match is None:
        return "INFO"
    kind = match.group(1).lower()
    if kind == "error":
        return "ERROR"
    if kind == "warning":
        return "WARNING"
    return "INFO"


def _diagnostics(messages: tuple[str, ...]) -> tuple[LeanElaborationDiagnostic, ...]:
    diagnostics: list[LeanElaborationDiagnostic] = []
    for message in messages:
        diagnostics.append(
            LeanElaborationDiagnostic(
                severity=_lean_message_severity(message),
                message=message,
            )
        )
    return tuple(diagnostics)


def _environment_digest(
    *,
    environment: LeanEnvironment,
    lean_version: str,
    lean_commit: str,
    mathlib_commit: str | None,
    imports: tuple[str, ...],
    options: tuple[LeanElaborationOption, ...],
    semantic_runtime_digest: str | None = None,
) -> str:
    payload = {
        "environment": environment.value,
        "lean_version": lean_version,
        "lean_commit": lean_commit,
        "mathlib_commit": mathlib_commit,
        "imports": list(imports),
        "options": [option.model_dump(mode="json") for option in options],
        "semantic_runtime_digest": semantic_runtime_digest,
    }
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _semantic_runtime_digest(runtime: CapabilityProviderRuntime | None) -> str | None:
    if runtime is None:
        return None
    semantic_runtime = runtime.configuration.get("semantic_runtime")
    return (
        lean_semantic_runtime_digest(semantic_runtime)
        if isinstance(semantic_runtime, dict)
        else None
    )


def _indent_proof(proof: str) -> str:
    lines = proof.splitlines()
    return "\n".join(f"  {line}" for line in lines)


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _validate_statement(statement: str) -> None:
    if "\n" in statement or "\r" in statement:
        raise ValueError("statement must be one Lean expression")
    if ":=" in statement:
        raise ValueError("statement must not contain ':='")
    if _FORBIDDEN_STATEMENT.search(statement):
        raise ValueError("statement contains a forbidden command")


def _validate_proof(proof: str) -> None:
    if "\x00" in proof:
        raise ValueError("proof contains a null byte")
    if _FORBIDDEN_PROOF.search(proof):
        raise ValueError("proof contains a dangerous command")


# ---------------------------------------------------------------------------
# Installation metadata.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeanStatementInstallation:
    semantics_uri: str
    proposal_schema_uri: str
    comparison_schema_uri: str


@dataclass(frozen=True, slots=True)
class _Resources:
    store: ArtifactRepository
    artifacts: ArtifactService
    semantics_uri: str
    proposal_schema_uri: str
    comparison_schema_uri: str
    provider_runtime: CapabilityProviderRuntime
    lean_executable: str | None


def install_lean_statement_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    provider_runtime: CapabilityProviderRuntime | None = None,
) -> tuple[
    tuple[
        LeanStatementProposalAdapter,
        LeanStatementCompareAdapter,
    ],
    LeanStatementInstallation,
]:
    """Register schemas and return the three atomic Lean statement adapters."""

    if provider_runtime is None:
        provider_runtime = lean_frontend_provider_runtime()
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-statement",
        version="1",
        definition={
            "description": (
                "atomic Lean statement proposal, direct proposition "
                "elaboration, and statement comparison; "
                "none certify theoremhood, truth, or semantic intent"
            ),
            "verification": (
                "none; elaboration establishes a well-typed Prop expression "
                "in the bound environment but does not establish truth"
            ),
        },
    )
    proposal_schema_uri = schemas.register(
        name="jacobian.lean4-statement-proposal",
        version="2",
        schema=LeanStatementProposalArtifact.model_json_schema(),
    )
    comparison_schema_uri = schemas.register(
        name="jacobian.lean4-statement-comparison",
        version="1",
        schema=LeanStatementComparisonArtifact.model_json_schema(),
    )
    resources = _Resources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        proposal_schema_uri=proposal_schema_uri,
        comparison_schema_uri=comparison_schema_uri,
        provider_runtime=provider_runtime,
        lean_executable=(
            str(provider_runtime.configuration["executable"])
            if provider_runtime.availability is CapabilityProviderAvailability.AVAILABLE
            else None
        ),
    )
    adapters = (
        LeanStatementProposalAdapter(resources),
        LeanStatementCompareAdapter(resources),
    )
    installation = LeanStatementInstallation(
        semantics_uri=semantics_uri,
        proposal_schema_uri=proposal_schema_uri,
        comparison_schema_uri=comparison_schema_uri,
    )
    return adapters, installation


# ---------------------------------------------------------------------------
# Adapter 1: lean.statement.propose
# ---------------------------------------------------------------------------


class LeanStatementProposalAdapter:
    """Type-check one proposed Lean statement; no semantic certification."""

    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.statement.propose",
            version="2",
            title="Propose one Lean statement with type-check status",
            description=(
                "Validate either a proposed statement against an informal "
                "claim or directly elaborate one proposition without an "
                "informal claim. Returns durable environment-bound elaboration "
                "details. It does not establish truth or semantic intent."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanStatementProposalRequest.model_json_schema(),
            output_schema=LeanStatementProposalOutput.model_json_schema(),
            tags=("lean", "statement", "elaboration", "proposal", "proposition"),
            accepted_input_kinds=(
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.FORMAL_PROPOSITION,
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="elaborate_true",
                    description=(
                        "Elaborate the proposition True in the pinned CORE "
                        "environment without assessing its truth."
                    ),
                    mode=CapabilityMode.EXPLORE,
                    input=LeanStatementProposalRequest.model_validate(
                        {
                            "operation": "ELABORATE_PROPOSITION",
                            "environment": "CORE",
                            "proposed_statement": "True",
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
            validated = LeanStatementProposalRequest.model_validate(request.input)
            _validate_statement(validated.proposed_statement)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_STATEMENT_PROPOSAL",
                    stage="request_validation",
                    message="The Lean statement proposal is invalid.",
                    hint=(
                        "Provide one single-line Lean expression without sorry, "
                        "admit, import, or other forbidden commands."
                    ),
                )
            ) from exc
        try:
            elaboration = (
                _elaborate_statement(
                    validated.proposed_statement,
                    executable=self.resources.lean_executable,
                    provider_runtime=self.resources.provider_runtime,
                )
                if validated.operation == "PROPOSE"
                else _elaborate_proposition(
                    validated.proposed_statement,
                    executable=self.resources.lean_executable,
                    provider_runtime=self.resources.provider_runtime,
                )
            )
        except _LeanUnavailableError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_BACKEND_UNAVAILABLE",
                    stage="elaboration",
                    message=str(exc),
                    hint=(
                        "Install Lean or elan and ensure the `lean` binary is on "
                        "PATH, then retry."
                    ),
                )
            ) from exc
        version, commit = _lean_version_info(
            self.resources.lean_executable,
            self.resources.provider_runtime,
        )
        imports = elaboration.used_imports
        options = elaboration.options
        artifact_payload = LeanStatementProposalArtifact(
            operation=validated.operation,
            environment=validated.environment,
            environment_digest=_environment_digest(
                environment=validated.environment,
                lean_version=version,
                lean_commit=commit,
                mathlib_commit=None,
                imports=imports,
                options=options,
                semantic_runtime_digest=_semantic_runtime_digest(
                    self.resources.provider_runtime
                ),
            ),
            informal_claim=validated.informal_claim,
            proposed_statement=validated.proposed_statement,
            elaborates=elaboration.elaborates,
            elaborated_expression=elaboration.elaborated_expression,
            sorry_count=elaboration.sorry_count,
            goals=(),
            messages=elaboration.messages,
            diagnostics=_diagnostics(elaboration.messages),
            used_imports=imports,
            used_declarations=elaboration.used_declarations,
            options=options,
            lean_version=version,
            lean_commit=commit,
            source_locator=validated.source_locator,
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.proposal_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary=(
                "Lean statement operation with elaboration status "
                f"(elaborates={elaboration.elaborates})"
            ),
        )
        output = LeanStatementProposalOutput(
            **artifact_payload.model_dump(mode="python"),
            proposal_uri=artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "one Lean proposition directly elaborated against Prop"
                    if validated.operation == "ELABORATE_PROPOSITION"
                    else "one Lean statement elaborated with sorry as proof"
                ),
                parameters={
                    "operation": validated.operation,
                    "environment": validated.environment.value,
                    "proposed_statement": validated.proposed_statement,
                    "environment_digest": artifact_payload.environment_digest,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "the pinned Lean frontend elaborated the proposition or "
                    "reported complete diagnostics for this bounded request"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "Lean elaboration reports whether the expression is a "
                    "well-typed proposition in the bound environment; it does "
                    "not establish truth, theoremhood, or semantic intent"
                ),
            ),
            artifact_uris=(artifact.artifact_uri,),
        )


# ---------------------------------------------------------------------------
# Adapter 2: lean.statement.compare
# ---------------------------------------------------------------------------


class LeanStatementCompareAdapter:
    """Compare two Lean statements syntactically and by axiom set."""

    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.statement.compare",
            version="1",
            title="Compare two Lean statements and axiom sets (fail-closed)",
            description=(
                "Compare two Lean statements by syntactic identity and axiom "
                "set identity. Optionally elaborates both to report "
                "elaboration status. Never claims semantic equivalence; "
                "fail-closed when elaboration cannot be checked."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanStatementComparisonRequest.model_json_schema(),
            output_schema=LeanStatementComparisonOutput.model_json_schema(),
            tags=("lean", "statement", "comparison", "axiom-set"),
            invocation_examples=(
                example(
                    "core_true_identity",
                    "Compare identical Lean Core propositions.",
                    {
                        "environment": "CORE",
                        "statement_a": "True",
                        "statement_b": "True",
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanStatementComparisonRequest.model_validate(request.input)
            _validate_statement(validated.statement_a)
            _validate_statement(validated.statement_b)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_STATEMENT_COMPARISON",
                    stage="request_validation",
                    message="The Lean statement comparison request is invalid.",
                    hint=(
                        "Provide two single-line Lean expressions without "
                        "sorry, admit, import, or other forbidden commands."
                    ),
                )
            ) from exc
        statements_identical = _normalize_whitespace(
            validated.statement_a
        ) == _normalize_whitespace(validated.statement_b)
        axiom_sets_identical = set(validated.axiom_set_a) == set(validated.axiom_set_b)
        both_elaborate = False
        elaboration_checked = False
        elaboration_messages_a: tuple[str, ...] = ()
        elaboration_messages_b: tuple[str, ...] = ()
        try:
            result_a = _elaborate_statement(
                validated.statement_a,
                executable=self.resources.lean_executable,
                provider_runtime=self.resources.provider_runtime,
            )
            result_b = _elaborate_statement(
                validated.statement_b,
                executable=self.resources.lean_executable,
                provider_runtime=self.resources.provider_runtime,
            )
            elaboration_checked = True
            both_elaborate = result_a.elaborates and result_b.elaborates
            elaboration_messages_a = result_a.messages
            elaboration_messages_b = result_b.messages
        except _LeanUnavailableError:
            pass
        version, commit = _lean_version_info(
            self.resources.lean_executable,
            self.resources.provider_runtime,
        )
        artifact_payload = LeanStatementComparisonArtifact(
            environment=validated.environment,
            statement_a=validated.statement_a,
            statement_b=validated.statement_b,
            axiom_set_a=validated.axiom_set_a,
            axiom_set_b=validated.axiom_set_b,
            statements_identical=statements_identical,
            axiom_sets_identical=axiom_sets_identical,
            both_elaborate=both_elaborate,
            elaboration_checked=elaboration_checked,
            elaboration_messages_a=elaboration_messages_a,
            elaboration_messages_b=elaboration_messages_b,
            lean_version=version,
            lean_commit=commit,
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.comparison_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary=(
                f"Lean statement comparison (identical={statements_identical}, "
                f"axioms_identical={axiom_sets_identical}, "
                f"elaboration_checked={elaboration_checked})"
            ),
        )
        output = LeanStatementComparisonOutput(
            **artifact_payload.model_dump(mode="python"),
            comparison_uri=artifact.artifact_uri,
        )
        completeness_status = (
            CapabilityCompletenessStatus.COMPLETE
            if elaboration_checked
            else CapabilityCompletenessStatus.PARTIAL
        )
        completeness_basis = (
            "both statements were elaborated and compared syntactically and "
            "by axiom set"
            if elaboration_checked
            else "syntactic and axiom-set comparison completed; elaboration "
            "was not checked because Lean is unavailable"
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "syntactic and axiom-set comparison of two Lean statements"
                ),
                parameters={
                    "environment": validated.environment.value,
                    "statement_a": validated.statement_a,
                    "statement_b": validated.statement_b,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=completeness_status,
                basis=completeness_basis,
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "deterministic syntactic and axiom-set comparison; "
                    "this does not certify semantic equivalence of the "
                    "statements"
                ),
            ),
            artifact_uris=(artifact.artifact_uri,),
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
