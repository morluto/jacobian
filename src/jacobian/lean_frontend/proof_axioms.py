"""Bounded, non-authoritative inspection of a Lean proof's axiom closure."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_adapters import parse_capability_input
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_proof_axioms import (
    LeanProofAxiomsArtifact,
    LeanProofAxiomsInspectOutput,
    LeanProofAxiomsInspectRequest,
)
from jacobian.contracts.lean_statement import LeanElaborationDiagnostic
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.providers.lean_runtime import (
    LeanRuntimeIdentityError,
    lean_semantic_runtime_digest,
    require_lean_semantic_runtime_identity,
)
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository

_AXIOM_LINE = re.compile(r"'jacobian_theorem' depends on axioms: \[([^\]]*)\]")
_NO_AXIOMS = "'jacobian_theorem' does not depend on any axioms"
_MESSAGE_KIND = re.compile(r"(?:^|:\s*)(error|warning|info)(?:\([^)]*\))?:", re.I)
_FORBIDDEN_SOURCE_TOKENS = frozenset(
    {
        "admit",
        "axiom",
        "class",
        "def",
        "elab",
        "end",
        "example",
        "import",
        "instance",
        "lemma",
        "macro",
        "namespace",
        "native_decide",
        "opaque",
        "run_tac",
        "section",
        "set_option",
        "sorry",
        "syntax",
        "theorem",
        "unsafe",
    }
)
_FORBIDDEN_PROOF_SOURCE_TOKENS = _FORBIDDEN_SOURCE_TOKENS - {"admit", "sorry"}
_MAX_PROOF_HOLES = 64


@dataclass(frozen=True, slots=True)
class LeanProofAxiomsInstallation:
    semantics_uri: str
    artifact_schema_uri: str


@dataclass(frozen=True, slots=True)
class _InspectionResources:
    artifacts: ArtifactService
    installations: Mapping[LeanEnvironment, Any]
    semantics_uri: str
    artifact_schema_uri: str
    provider_runtime: CapabilityProviderRuntime


def install_lean_proof_axioms_capability(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    installations: Mapping[LeanEnvironment, Any],
    provider_runtime: CapabilityProviderRuntime,
) -> tuple[LeanProofAxiomsAdapter, LeanProofAxiomsInstallation]:
    """Install fact-only proof inspection for the exact pinned Lean runtime."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-proof-axioms",
        version="1",
        definition={
            "description": (
                "reported imports, axiom closure, and proof-hole counts for one "
                "bounded proof source"
            ),
            "verification": "inspection facts only; never verification evidence",
            "scope": "AXIOM_DEPENDENCY_ONLY",
        },
    )
    artifact_schema_uri = schemas.register(
        name="jacobian.lean4-proof-axioms",
        version="1",
        schema=model_schema(LeanProofAxiomsArtifact),
    )
    resources = _InspectionResources(
        artifacts=artifacts,
        installations=installations,
        semantics_uri=semantics_uri,
        artifact_schema_uri=artifact_schema_uri,
        provider_runtime=provider_runtime,
    )
    return (
        LeanProofAxiomsAdapter(resources),
        LeanProofAxiomsInstallation(semantics_uri, artifact_schema_uri),
    )


class LeanProofAxiomsAdapter:
    def __init__(self, resources: _InspectionResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof.axioms.inspect",
            version="1",
            title="Inspect a Lean proof's reported axiom closure",
            description=(
                "Compile one bounded Lean proof in its exact CORE or MATHLIB "
                "environment and report imports, proof-hole counts, and the axiom "
                "closure printed by Lean. This is inspection, not verification."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            input_schema=LeanProofAxiomsInspectRequest.model_json_schema(),
            output_schema=LeanProofAxiomsInspectOutput.model_json_schema(),
            read_only=True,
            tags=("lean", "proof", "axioms", "trust-base", "inspection"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="inspect_true",
                    description="Inspect a proof of True without making a trust decision.",
                    input={
                        "environment": "CORE",
                        "statement": "True",
                        "proof": "by trivial",
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> LeanProofAxiomsInspectRequest:
        try:
            validated = parse_capability_input(
                LeanProofAxiomsInspectRequest, request.input
            )
            _validate_source(validated.statement, validated.proof)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_PROOF_AXIOMS_REQUEST",
                    stage="request_validation",
                    message="The bounded Lean proof-inspection request is invalid.",
                    hint=(
                        "Provide one statement expression and a proof body without "
                        "declarations, imports, or trust-policy commands."
                    ),
                )
            ) from exc
        return validated

    def invoke(self, validated: LeanProofAxiomsInspectRequest) -> OperationProjection:
        if validated.environment not in self.resources.installations:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_ENVIRONMENT_UNAVAILABLE",
                    stage="environment_resolution",
                    message="The requested pinned Lean environment is unavailable.",
                    hint="Install the exact Lean/Mathlib runtime, then retry.",
                )
            )

        if "semantic_runtime" in self.resources.provider_runtime.configuration:
            try:
                require_lean_semantic_runtime_identity(self.resources.provider_runtime)
            except LeanRuntimeIdentityError as exc:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="LEAN_RUNTIME_IDENTITY_UNAVAILABLE",
                        stage="runtime_identity",
                        message=(
                            "The pinned Lean semantic runtime changed or is unavailable."
                        ),
                        hint="Restore the exact authorized Lean runtime, then retry.",
                    )
                ) from exc

        started = time.monotonic()
        inspection = _inspect_source(validated)
        runtime = self.resources.provider_runtime
        if runtime.version is None or runtime.digest is None:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_PROVIDER_IDENTITY_INCOMPLETE",
                    stage="provider_identity",
                    message="The Lean runtime did not provide a pinned version and digest.",
                    hint="Install the exact authorized Lean runtime, then retry.",
                )
            )
        manifest_digest = inspection["package_manifest_digest"]
        environment_digest = _environment_digest(
            validated.environment,
            runtime,
            manifest_digest,
        )
        artifact_payload = LeanProofAxiomsArtifact(
            environment=validated.environment,
            environment_digest=environment_digest,
            provider_runtime_digest=runtime.digest,
            lean_version=runtime.version,
            lean_commit=_lean_commit(
                validated.environment, self.resources.installations
            ),
            mathlib_commit=(
                _mathlib_commit(self.resources.installations, validated.environment)
            ),
            imports=inspection["imports"],
            package_manifest_digest=manifest_digest,
            statement=validated.statement,
            proof=validated.proof,
            elaborated=inspection["elaborated"],
            inspection_complete=inspection["inspection_complete"],
            axioms_reported=inspection["axioms_reported"],
            axioms=inspection["axioms"],
            sorry_count=inspection["sorry_count"],
            admit_count=inspection["admit_count"],
            diagnostics=inspection["diagnostics"],
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.artifact_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary=(
                "Lean proof axiom inspection "
                f"(complete={artifact_payload.inspection_complete})"
            ),
        )
        output = LeanProofAxiomsInspectOutput(
            **artifact_payload.model_dump(mode="python"),
            proof_axioms_uri=artifact.artifact_uri,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=int((time.monotonic() - started) * 1000),
                detail=inspection["detail"],
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(artifact.artifact_uri,),
            ),
        )


def _validate_source(statement: str, proof: str) -> None:
    for value, field_name in ((statement, "statement"), (proof, "proof")):
        if any(marker in value for marker in ("--", "/-", "-/")):
            raise ValueError(f"{field_name} comments are outside the source boundary")
    if _contains_forbidden_source_token(statement):
        raise ValueError("statement contains an unsupported Lean command")
    if _contains_forbidden_source_token(
        proof, forbidden_tokens=_FORBIDDEN_PROOF_SOURCE_TOKENS
    ):
        raise ValueError("proof contains an unsupported Lean command")
    sorry_count, admit_count = _proof_hole_counts(proof)
    if sorry_count + admit_count > _MAX_PROOF_HOLES:
        raise ValueError("proof contains more than 64 proof holes")


def _lean_source_tokens(source: str) -> tuple[str, ...]:
    """Return Lean identifiers and command markers outside literals/comments."""

    tokens: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        if source.startswith("--", index):
            index = _skip_line_comment(source, index, length)
            continue
        if source.startswith("/-", index):
            index = _skip_block_comment(source, index, length)
            continue
        character = source[index]
        if character == '"':
            index = _skip_string_literal(source, index, length)
            continue
        if character == "'" and (
            index == 0 or not _is_identifier_character(source[index - 1])
        ):
            index = _skip_char_literal(source, index, length)
            continue
        if _is_identifier_start(character):
            end = index + 1
            while end < length and _is_identifier_character(source[end]):
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        if character == "#":
            tokens.append(character)
        index += 1
    return tuple(tokens)


def _skip_line_comment(source: str, index: int, length: int) -> int:
    newline = source.find("\n", index + 2)
    return length if newline == -1 else newline + 1


def _skip_block_comment(source: str, index: int, length: int) -> int:
    index += 2
    depth = 1
    while index < length and depth:
        if source.startswith("/-", index):
            depth += 1
            index += 2
        elif source.startswith("-/", index):
            depth -= 1
            index += 2
        else:
            index += 1
    return index


def _skip_string_literal(source: str, index: int, length: int) -> int:
    return _skip_quoted_literal(source, index, length, '"')


def _skip_char_literal(source: str, index: int, length: int) -> int:
    return _skip_quoted_literal(source, index, length, "'")


def _skip_quoted_literal(source: str, index: int, length: int, delimiter: str) -> int:
    index += 1
    while index < length:
        character = source[index]
        if character in "\r\n":
            raise ValueError("Lean quoted literals cannot contain a newline")
        if character == "\\":
            if index + 1 >= length:
                raise ValueError("Lean quoted literal ends with an escape")
            if source[index + 1] in "\r\n":
                raise ValueError("Lean quoted literals cannot contain a newline")
            index += 2
        elif character == delimiter:
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated Lean quoted literal")


def _is_identifier_start(character: str) -> bool:
    return character == "_" or character.isalpha()


def _is_identifier_character(character: str) -> bool:
    return character == "_" or character == "'" or character.isalnum()


def _contains_forbidden_source_token(
    source: str,
    *,
    forbidden_tokens: frozenset[str] = _FORBIDDEN_SOURCE_TOKENS,
) -> bool:
    return any(
        token == "#" or token.casefold() in forbidden_tokens
        for token in _lean_source_tokens(source)
    )


def _proof_hole_counts(source: str) -> tuple[int, int]:
    tokens = _lean_source_tokens(source)
    return (
        sum(token.casefold() == "sorry" for token in tokens),
        sum(token.casefold() == "admit" for token in tokens),
    )


def _inspect_source(request: LeanProofAxiomsInspectRequest) -> dict[str, Any]:
    from jacobian_checkers import lean4

    imports = (
        ("Init.Prelude",)
        if request.environment is LeanEnvironment.CORE
        else ("Mathlib",)
    )
    import_line = f"import {imports[0]}\n"
    source = (
        f"{import_line}"
        "set_option autoImplicit false\n"
        f"theorem jacobian_theorem : ({request.statement}) := {request.proof}\n"
        "#print axioms jacobian_theorem\n"
    )
    try:
        completed = lean4._run_lean(source, environment_name=request.environment.value)
        output = (completed.stdout + completed.stderr).strip()
    except (OSError, RuntimeError, TimeoutError) as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="LEAN_BACKEND_UNAVAILABLE",
                stage="inspection",
                message="The pinned Lean process could not complete inspection.",
                hint="Retry after restoring the exact pinned Lean environment.",
                details={"failure": type(exc).__name__},
            )
        ) from exc

    messages = tuple(_diagnostics(output))
    errors = any(item.severity == "ERROR" for item in messages)
    axioms_match = _AXIOM_LINE.search(output)
    no_axioms = _NO_AXIOMS in output
    axioms_reported = axioms_match is not None or no_axioms
    axioms = (
        tuple(
            sorted(
                item.strip()
                for item in axioms_match.group(1).split(",")
                if item.strip()
            )
        )
        if axioms_match is not None
        else ()
    )
    elaborated = completed.returncode == 0 and not errors
    inspection_complete = elaborated and axioms_reported
    if elaborated and not axioms_reported:
        messages = (
            *messages,
            LeanElaborationDiagnostic(
                severity="ERROR",
                message="Lean completed without a parseable axiom-closure report.",
            ),
        )
    try:
        package_manifest_digest = _manifest_digest(request.environment)
    except (OSError, RuntimeError) as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="LEAN_RUNTIME_IDENTITY_UNAVAILABLE",
                stage="runtime_identity",
                message="The pinned Lean runtime identity could not be inspected.",
                hint="Restore the exact authorized Lean runtime, then retry.",
                details={"failure": type(exc).__name__},
            )
        ) from exc
    sorry_count, admit_count = _proof_hole_counts(request.proof)
    return {
        "imports": imports,
        "package_manifest_digest": package_manifest_digest,
        "elaborated": elaborated,
        "inspection_complete": inspection_complete,
        "axioms_reported": axioms_reported,
        "axioms": axioms,
        "sorry_count": sorry_count,
        "admit_count": admit_count,
        "diagnostics": messages,
        "detail": "complete axiom inspection"
        if inspection_complete
        else "incomplete axiom inspection",
    }


def _diagnostics(output: str) -> tuple[LeanElaborationDiagnostic, ...]:
    diagnostics: list[LeanElaborationDiagnostic] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _MESSAGE_KIND.search(stripped)
        if match is None:
            continue
        severity = cast(
            Literal["ERROR", "WARNING", "INFO"],
            {
                "error": "ERROR",
                "warning": "WARNING",
                "info": "INFO",
            }[match.group(1).lower()],
        )
        diagnostics.append(
            LeanElaborationDiagnostic(severity=severity, message=stripped)
        )
    return tuple(diagnostics)


def _manifest_digest(environment: LeanEnvironment) -> str | None:
    if environment is LeanEnvironment.CORE:
        return None
    from jacobian_checkers import lean4

    _, runtime = lean4.inspect_runtime(require_mathlib=True)
    if runtime is None:
        raise RuntimeError("pinned Mathlib runtime is unavailable")
    return _digest_file(runtime / "lake-manifest.json")


def _environment_digest(
    environment: LeanEnvironment,
    runtime: CapabilityProviderRuntime,
    manifest_digest: str | None,
) -> str:
    semantic_runtime = runtime.configuration.get("semantic_runtime")
    payload = {
        "contract": "jacobian.lean.proof.axioms/v1",
        "environment": environment.value,
        "provider_runtime_digest": runtime.digest,
        "manifest_digest": manifest_digest,
        "semantic_runtime_digest": (
            lean_semantic_runtime_digest(semantic_runtime)
            if isinstance(semantic_runtime, dict)
            else None
        ),
    }
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _lean_commit(
    environment: LeanEnvironment, installations: Mapping[LeanEnvironment, Any]
) -> str:
    installation = installations.get(environment)
    lean_commit = getattr(installation, "lean_commit", None)
    if isinstance(lean_commit, str) and lean_commit:
        return lean_commit
    from jacobian_checkers import lean4

    return lean4.LEAN_COMMIT


def _mathlib_commit(
    installations: Mapping[LeanEnvironment, Any],
    environment: LeanEnvironment,
) -> str | None:
    installation = installations.get(environment)
    return (
        getattr(installation, "mathlib_commit", None)
        if installation is not None
        else None
    )


__all__ = [
    "LeanProofAxiomsAdapter",
    "LeanProofAxiomsInstallation",
    "install_lean_proof_axioms_capability",
]
