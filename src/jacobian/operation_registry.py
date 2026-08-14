"""Lazy resolution of one selected built-in mathematical operation."""

from __future__ import annotations

from typing import Any

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import (
    OperationCatalog,
    OperationCatalogError,
    OperationDeclarationRecord,
    exact_checker_declaration_digest,
    operation_declaration_digest,
)
from jacobian.operation_declarations import OperationDeclarations
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polytope import PolytopeService
from jacobian.portfolio.builtin import load_builtin_operation_module
from jacobian.registry import CheckerRegistry
from jacobian.verification.service import VerificationService

_SELECTED_GRAPH_OPERATIONS = frozenset(
    {
        "graph.construct.explicit",
        "graph.search.atlas",
        "graph.compute.properties",
        "graph.construct.compose",
        "graph.enumerate.nonisomorphic",
        "graph.realize.degree_sequence",
        "graph.compute.neighborhood_independence",
        "graph.degree_sequence.verify",
        "graph.neighborhood_independence.verify",
        "graph.isomorphism.verify",
    }
)
_SELECTED_POLYNOMIAL_OPERATIONS = frozenset(
    {
        "polynomial.expression.normalize",
        "polynomial.expression_normalization.verify",
    }
)
_SELECTED_DIRECT_OPERATIONS = frozenset({"polytope.separate"})
_SELECTED_RESOURCE_OPERATIONS = (
    _SELECTED_GRAPH_OPERATIONS
    | _SELECTED_POLYNOMIAL_OPERATIONS
    | _SELECTED_DIRECT_OPERATIONS
)


def supports_selected_operation(operation_id: str) -> bool:
    """Return whether a non-declaration operation has a narrow selected binder."""

    return operation_id in _SELECTED_RESOURCE_OPERATIONS


class OperationRegistry:
    """Import, verify, and cache only selected built-in declarations."""

    def __init__(
        self,
        catalog: OperationCatalog,
        binder: OperationBinder,
        verification: VerificationService,
        checkers: CheckerRegistry,
        polynomial_expressions: PolynomialExpressionArtifactService,
        polytope: PolytopeService,
    ) -> None:
        self.catalog = catalog
        self.binder = binder
        self.verification = verification
        self.checkers = checkers
        self.polynomial_expressions = polynomial_expressions
        self.polytope = polytope
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        descriptor = self.catalog.inspect(operation_id)
        record = self.catalog.declaration_record(operation_id)
        if descriptor is None or record is None:
            raise OperationCatalogError(f"unknown or hidden operation: {operation_id}")
        try:
            _module_name, operations, checker_declarations = (
                load_builtin_operation_module(record.module)
            )
        except ValueError as exc:
            adapter = self._resolve_selected_resource_operation(
                operation_id,
                descriptor,
                record,
            )
            if adapter is not None:
                return adapter
            raise OperationCatalogError(
                f"operation {operation_id} is not a declared built-in operation"
            ) from exc
        matches = tuple(
            operation
            for operation in operations
            if operation.operation_id == operation_id
        )
        if not matches and any(
            declaration.verification_operation_id == operation_id
            for declaration in checker_declarations
        ):
            return self._resolve_exact_verifier(
                operation_id,
                descriptor,
                record,
                operations,
                checker_declarations,
            )
        if len(matches) != 1:
            raise OperationCatalogError(
                f"operation locator did not resolve exactly once: {operation_id}"
            )
        declaration = matches[0]
        if declaration.version != descriptor.version:
            raise OperationCatalogError(
                f"operation declaration version changed; run `jacobian update`: {operation_id}"
            )
        if operation_declaration_digest(declaration) != record.declaration_digest:
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        bound = self.binder.bind(operations)
        adapter = next(
            candidate
            for candidate in bound.adapters
            if candidate.descriptor.operation_id == operation_id
        )
        if adapter.descriptor.model_dump(mode="json") != descriptor.model_dump(
            mode="json"
        ):
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter

    def _resolve_selected_resource_operation(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
    ) -> OperationAdapter[Any] | None:
        if not supports_selected_operation(operation_id):
            return None
        adapter: OperationAdapter[Any] | None
        if operation_id in _SELECTED_GRAPH_OPERATIONS:
            from jacobian.graphs.installation import bind_selected_graph_operation

            adapter = bind_selected_graph_operation(
                operation_id,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.verification,
                self.checkers,
                self.catalog,
            )
        elif operation_id == "polynomial.expression.normalize":
            from jacobian.sympy_polynomial_normalization import (
                install_sympy_polynomial_normalization_operation,
            )

            if descriptor.provider_runtime is None:
                raise OperationCatalogError(
                    "polynomial provider observation is missing; run `jacobian update`"
                )
            adapter = install_sympy_polynomial_normalization_operation(
                self.polynomial_expressions,
                descriptor.provider_runtime,
            )
        elif operation_id == "polynomial.expression_normalization.verify":
            from jacobian.polynomial_expression_operations import (
                bind_selected_polynomial_expression_checker,
            )

            adapter = bind_selected_polynomial_expression_checker(
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.polynomial_expressions,
                self.verification,
                self.checkers,
                self.catalog,
            )
        elif operation_id == "polytope.separate":
            from jacobian.polytope_operations import PolytopeSeparationAdapter

            adapter = PolytopeSeparationAdapter(self.polytope)
        else:
            adapter = None
        if adapter is None:
            raise OperationCatalogError(
                f"selected operation binder is missing: {operation_id}"
            )
        expected_digest = operation_declaration_digest_from_descriptor(descriptor)
        if expected_digest != record.declaration_digest:
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        if adapter.descriptor.model_dump(mode="json") != descriptor.model_dump(
            mode="json"
        ):
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter

    def _resolve_exact_verifier(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        operations: OperationDeclarations,
        checker_declarations: tuple[ExactReplayCheckerDeclaration, ...],
    ) -> OperationAdapter[Any]:
        from jacobian.exact_domain_checkers import bind_selected_exact_verification

        checker_declaration = next(
            declaration
            for declaration in checker_declarations
            if declaration.verification_operation_id == operation_id
        )
        if (
            exact_checker_declaration_digest(checker_declaration, descriptor)
            != record.declaration_digest
        ):
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        adapter = bind_selected_exact_verification(
            catalog=self.catalog,
            operation_id=operation_id,
            operations=operations,
            declarations=checker_declarations,
            binder=self.binder,
            verification=self.verification,
            checkers=self.checkers,
        )
        if adapter.descriptor.model_dump(mode="json") != descriptor.model_dump(
            mode="json"
        ):
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter


def operation_declaration_digest_from_descriptor(
    descriptor: OperationDescriptor,
) -> str:
    """Match the catalog digest used by retained resource-backed operations."""

    from jacobian.operation_catalog import declaration_digest

    return declaration_digest(
        {
            "operation_id": descriptor.operation_id,
            "version": descriptor.version,
            "input_schema": descriptor.input_schema,
            "output_schema": descriptor.output_schema,
        }
    )


__all__ = ["OperationRegistry", "supports_selected_operation"]
