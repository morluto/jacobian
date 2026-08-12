"""Bundled adapters for Lean capabilities."""

from __future__ import annotations

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.lean import (
    LeanCheckOutput,
    LeanDeclarationInspectOutput,
    LeanDeclarationInspectRequest,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDependencyGraphOutput,
    LeanDependencyGraphRequest,
    LeanEnvironment,
)
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanDeclarationService,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.schema_registry import model_schema


class LeanCheckAdapter:
    def __init__(
        self,
        lean: LeanService,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.lean = lean
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.check",
            version="2",
            title="Independently check an exact Lean proof",
            description=(
                "Compile and replay one proposition with the pinned CORE or MATHLIB "
                "kernel profile. Statements are single Lean expressions, including "
                "finite-witness let expressions; declarations and trust escapes are "
                "forbidden. Rejections include stable proof-repair diagnostics, "
                "payload-relative source spans, and the bounded backend message."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "proof": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "environment": {"enum": ["CORE", "MATHLIB"]},
                },
                "required": ["statement", "proof"],
                "additionalProperties": False,
            },
            output_schema=LeanCheckOutput.model_json_schema(),
            tags=(
                "lean",
                "proof",
                "checker",
                "verification",
                "core",
                "mathlib",
                "finite-witness",
                "proof-repair",
                "diagnostics",
                "type-mismatch",
                "source-span",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="finite-witness-let",
                    description=(
                        "Check a finite witness encoded as one let expression without "
                        "adding declarations."
                    ),
                    input={
                        "environment": "CORE",
                        "statement": "let n : Nat := 2; n + n = 4",
                        "proof": "rfl",
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        payload = request.input
        checked = self.lean.verify(
            statement=str(payload["statement"]),
            proof=str(payload["proof"]),
            environment=LeanEnvironment(str(payload.get("environment", "CORE"))),
        )
        verified = checked.result.verification_record_uri is not None
        evidence = (checked.certificate_uri,)
        scope_uri = checked.result.scope_uri
        output = LeanCheckOutput(
            conclusion=checked.result.conclusion,
            execution=checked.result.execution,
            input=checked.result.input,
            diagnostics=checked.diagnostics,
            claim_uri=checked.claim_uri,
            candidate_uri=checked.candidate_uri,
            certificate_uri=checked.certificate_uri,
            verification_record_uri=checked.result.verification_record_uri,
            cache_hit=checked.cache_hit,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.result.execution,
            output=output.model_dump(mode="json"),
            verification_record_uri=(
                checked.result.verification_record_uri if verified else None
            ),
            artifact_uris=(
                checked.claim_uri,
                checked.candidate_uri,
                *evidence,
                *((scope_uri,) if scope_uri is not None else ()),
                *(
                    (checked.result.verification_record_uri,)
                    if checked.result.verification_record_uri is not None
                    else ()
                ),
            ),
        )


class LeanDeclarationSearchAdapter:
    def __init__(
        self,
        declarations: LeanDeclarationService,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.declarations = declarations
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.declaration.search",
            version="2",
            title="Search pinned Lean and Mathlib declarations",
            description=(
                "Find theorem, definition, and other declaration names directly in "
                "the installed pinned Lean CORE or MATHLIB environment. Search by a "
                "case-sensitive name substring and/or exact constants occurring in "
                "elaborated declaration types; use this instead of shell-searching "
                "local Mathlib source or caches. Results include portable pinned "
                "Lean/Mathlib version and commit identity."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=model_schema(LeanDeclarationSearchRequest),
            output_schema=model_schema(LeanDeclarationSearchOutput),
            read_only=True,
            tags=(
                "lean",
                "mathlib",
                "declaration",
                "theorem-search",
                "formal-environment",
                "retrieval",
                "premise-discovery",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="find_sqrt_two_irrationality",
                    description=(
                        "Search pinned Mathlib for declarations whose name mentions "
                        "irrational square root."
                    ),
                    input=LeanDeclarationSearchRequest.model_validate(
                        {
                            "environment": "MATHLIB",
                            "name_contains": "irrational_sqrt",
                            "kinds": ["THEOREM"],
                            "result_limit": 10,
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        query = LeanDeclarationSearchRequest.model_validate(request.input)
        try:
            searched = self.declarations.search(query)
        except LeanDeclarationBackendError as exc:
            raise _declaration_invocation_error(exc) from exc
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=searched.model_dump(mode="json"),
        )


class LeanDeclarationInspectAdapter:
    def __init__(
        self,
        declarations: LeanDeclarationService,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.declarations = declarations
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.declaration.inspect",
            version="2",
            title="Inspect an exact Lean or Mathlib declaration",
            description=(
                "Resolve one exact fully qualified declaration name directly in the "
                "installed pinned Lean CORE or MATHLIB environment. Return its "
                "elaborated type, kind, docs, source metadata, environment digest, "
                "and portable pinned Lean/Mathlib version and commit identity."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=model_schema(LeanDeclarationInspectRequest),
            output_schema=model_schema(LeanDeclarationInspectOutput),
            read_only=True,
            tags=(
                "lean",
                "mathlib",
                "declaration",
                "theorem-inspection",
                "formal-environment",
                "retrieval",
                "inspection",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="inspect_sqrt_two_irrationality",
                    description=(
                        "Inspect the exact Mathlib declaration returned by declaration "
                        "search."
                    ),
                    input=LeanDeclarationInspectRequest.model_validate(
                        {
                            "environment": "MATHLIB",
                            "declaration_name": "irrational_sqrt_two",
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        query = LeanDeclarationInspectRequest.model_validate(request.input)
        try:
            inspected = self.declarations.inspect(query)
        except LeanDeclarationBackendError as exc:
            raise _declaration_invocation_error(exc) from exc
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=inspected.model_dump(mode="json"),
        )


class LeanDependencyGraphAdapter:
    def __init__(
        self,
        declarations: LeanDeclarationService,
        provider_runtime: CapabilityProviderRuntime,
        artifacts: ArtifactService,
        *,
        semantics_uri: str,
        dependency_graph_schema_uri: str,
    ) -> None:
        self.declarations = declarations
        self.artifacts = artifacts
        self.semantics_uri = semantics_uri
        self.dependency_graph_schema_uri = dependency_graph_schema_uri
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.declaration.dependencies",
            version="1",
            title="Extract Lean declaration dependencies",
            description=(
                "Extract a bounded dependency subgraph from elaborated declaration "
                "types and values in a pinned Lean environment."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=model_schema(LeanDependencyGraphRequest),
            output_schema=model_schema(LeanDependencyGraphOutput),
            read_only=True,
            tags=("lean", "declaration", "dependency-graph", "formal-artifact"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        query = LeanDependencyGraphRequest.model_validate(request.input)
        try:
            graph = self.declarations.dependencies(query)
        except LeanDeclarationBackendError as exc:
            raise _declaration_invocation_error(exc) from exc
        graph_artifact = self.artifacts.put(
            schema_uri=self.dependency_graph_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=graph.model_dump(mode="json"),
            summary=(
                f"bounded Lean dependency subgraph rooted at {query.root_declaration}"
            ),
        )
        output = LeanDependencyGraphOutput(
            **graph.model_dump(mode="python"),
            dependency_graph_uri=graph_artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            artifact_uris=(graph_artifact.artifact_uri,),
        )


def _declaration_invocation_error(
    error: LeanDeclarationBackendError,
) -> CapabilityInvocationError:
    return CapabilityInvocationError(
        CapabilityDiagnostic(
            code=error.code,
            stage="lean_declaration_query",
            message=error.message,
            hint=(
                "Call math.find for the exact query contract and verify "
                "that the requested pinned environment is installed."
            ),
        )
    )
