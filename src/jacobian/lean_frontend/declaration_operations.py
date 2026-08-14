"""Typed operations for bounded Lean declaration queries."""

from __future__ import annotations

from jacobian.contracts.lean import (
    LeanDeclarationInspectOutput,
    LeanDeclarationInspectRequest,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDependencyGraphArtifact,
    LeanDependencyGraphRequest,
)
from jacobian.contracts.operations import (
    OperationDiagnostic,
    OperationExample,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanDeclarationService,
)
from jacobian.operation_bindings import durable_operation, inline_operation
from jacobian.operation_declarations import OperationDeclaration, OperationDeclarations
from jacobian.operations import (
    OperationRefusalError,
)


def _refusal(error: LeanDeclarationBackendError) -> OperationRefusalError:
    return OperationRefusalError(
        OperationDiagnostic(
            code=error.code,
            stage="lean_declaration_query",
            message=error.message,
            hint=(
                "Call math.find for the exact query contract and verify that the "
                "requested pinned environment is installed."
            ),
        )
    )


def lean_declaration_query_operations(
    declarations: LeanDeclarationService,
) -> OperationDeclarations:
    """Bind search and inspection to one declaration service instance."""

    def search(request: LeanDeclarationSearchRequest) -> LeanDeclarationSearchOutput:
        try:
            return declarations.search(request)
        except LeanDeclarationBackendError as error:
            raise _refusal(error) from error

    def inspect(
        request: LeanDeclarationInspectRequest,
    ) -> LeanDeclarationInspectOutput:
        try:
            return declarations.inspect(request)
        except LeanDeclarationBackendError as error:
            raise _refusal(error) from error

    def dependencies(
        request: LeanDependencyGraphRequest,
    ) -> LeanDependencyGraphArtifact:
        try:
            return declarations.dependencies(request)
        except LeanDeclarationBackendError as error:
            raise _refusal(error) from error

    return (
        inline_operation(
            OperationDeclaration(
                operation_id="lean.declaration.search",
                version="2",
                request_type=LeanDeclarationSearchRequest,
                result_type=LeanDeclarationSearchOutput,
                execute=search,
                title="Search pinned Lean and Mathlib declarations",
                description=(
                    "Find theorem, definition, and other declaration names "
                    "directly in the installed pinned Lean CORE or MATHLIB "
                    "environment. Search by a case-sensitive name substring "
                    "and/or exact constants occurring in elaborated declaration "
                    "types; use this instead of shell-searching local Mathlib "
                    "source or caches. Results include portable pinned "
                    "Lean/Mathlib version and commit identity."
                ),
                tags=(
                    "lean",
                    "mathlib",
                    "declaration",
                    "theorem-search",
                    "formal-environment",
                    "retrieval",
                    "premise-discovery",
                ),
                examples=(
                    OperationExample(
                        name="find_sqrt_two_irrationality",
                        description=(
                            "Find the exact square-root-of-two irrationality "
                            "declaration in pinned Mathlib."
                        ),
                        input=LeanDeclarationSearchRequest.model_validate(
                            {
                                "environment": "MATHLIB",
                                "name_contains": "irrational_sqrt_two",
                                "kinds": ["THEOREM"],
                                "result_limit": 1,
                            }
                        ).model_dump(mode="json"),
                    ),
                ),
            )
        ),
        inline_operation(
            OperationDeclaration(
                operation_id="lean.declaration.inspect",
                version="2",
                request_type=LeanDeclarationInspectRequest,
                result_type=LeanDeclarationInspectOutput,
                execute=inspect,
                title="Inspect an exact Lean or Mathlib declaration",
                description=(
                    "Resolve one exact fully qualified declaration name directly "
                    "in the installed pinned Lean CORE or MATHLIB environment. "
                    "Return its elaborated type, kind, docs, source metadata, "
                    "environment digest, and portable pinned Lean/Mathlib version "
                    "and commit identity."
                ),
                tags=(
                    "lean",
                    "mathlib",
                    "declaration",
                    "theorem-inspection",
                    "formal-environment",
                    "retrieval",
                    "inspection",
                ),
                examples=(
                    OperationExample(
                        name="inspect_sqrt_two_irrationality",
                        description=(
                            "Inspect the exact Mathlib declaration returned by "
                            "declaration search."
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
        ),
        durable_operation(
            OperationDeclaration(
                operation_id="lean.declaration.dependencies",
                version="2",
                request_type=LeanDependencyGraphRequest,
                result_type=LeanDependencyGraphArtifact,
                execute=dependencies,
                title="Extract Lean declaration dependencies",
                description=(
                    "Extract a bounded dependency subgraph from elaborated Lean "
                    "declaration types and values in a pinned environment."
                ),
                tags=(
                    "lean",
                    "declaration",
                    "dependency-graph",
                    "formal-artifact",
                ),
            ),
            resource_reason=(
                "the dependency graph has durable identity and supports later "
                "inspection or replay"
            ),
            preview_type=LeanDependencyGraphArtifact,
            preview=lambda result: result,
            preview_complete=True,
        ),
    )


__all__ = ["lean_declaration_query_operations"]
