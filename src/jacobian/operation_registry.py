"""Lazy resolution of one selected built-in mathematical operation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jacobian.builtin_operation_modules import load_builtin_operation_module
from jacobian.checker_operations import AuthorizedChecker
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
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
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
        "polynomial.interval.positivity.decide",
        "polynomial.interval.positivity.verify",
        "polynomial.interval.enclose",
        "polynomial.interval.enclosure.verify",
        "polynomial.map.evaluate",
        "polynomial.map.compute_jacobian",
        "polynomial.map.keller_condition.verify",
        "polynomial.map.collision_witness",
        "polynomial.map.collision.search",
        "polynomial.map.collision.verify",
        "polynomial.map.collision_evidence.verify",
        "polynomial.map.inverse.refute_by_collision",
        "polynomial.identity.verify",
        "polynomial.rational_function.identity.verify",
        "polynomial.map.inverse.candidate_synthesize",
        "polynomial.map.inverse.verify",
        "polynomial.system.solution.verify",
        "polynomial.system.rational_solution.search",
        "polynomial.jacobian_degree_slice.system.materialize",
        "polynomial.nullstellensatz.infeasibility_certificate.verify",
    }
)
_SELECTED_DIRECT_OPERATIONS = frozenset(
    {
        "polytope.separate",
        "finite.coverage.verify",
        "finite_magma.table.enumerate",
        "universal_algebra.evaluate_laws",
        "universal_algebra.search.countermodel",
        "universal_algebra.law_evaluation.verify",
        "sat.cnf.materialize",
        "sat.model.verify",
        "sat.unsat_proof.verify",
        "sat.lrat.verify",
        "smt.unsat_proof.verify",
        "sat.model.find",
        "sat.unsat_proof.find",
        "smt.unsat_proof.find",
    }
)
_SELECTED_LEAN_OPERATIONS = frozenset(
    {
        "lean.check",
        "lean.declaration.dependencies",
        "lean.declaration.inspect",
        "lean.declaration.search",
        "lean.proof.axioms.inspect",
        "lean.proof_edit.validate",
        "lean.proof_state.apply_tactic",
        "lean.proof_state.inspect",
        "lean.proof_state.metavariable_fields",
        "lean.retrieve.premises",
        "lean.statement.compare",
        "lean.statement.propose",
        "lean.term.apply",
    }
)
_SELECTED_RESOURCE_OPERATIONS = (
    _SELECTED_GRAPH_OPERATIONS
    | _SELECTED_POLYNOMIAL_OPERATIONS
    | _SELECTED_DIRECT_OPERATIONS
    | _SELECTED_LEAN_OPERATIONS
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
        sat: SatArtifactService,
        smt: SmtArtifactService,
    ) -> None:
        self.catalog = catalog
        self.binder = binder
        self.verification = verification
        self.checkers = checkers
        self.polynomial_expressions = polynomial_expressions
        self.polytope = polytope
        self.sat = sat
        self.smt = smt
        self._adapters: dict[str, OperationAdapter[Any]] = {}
        self._owned_close: list[Callable[[], None]] = []

    def close(self) -> None:
        """Close lazily selected subprocess-backed services in reverse order."""

        failures: list[Exception] = []
        while self._owned_close:
            close = self._owned_close.pop()
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup(
                "selected operation services failed to close", failures
            )

    def _own(self, resource: object) -> None:
        close = getattr(resource, "close", None)
        if not callable(close):
            raise TypeError("owned selected-operation resource must be closeable")
        self._owned_close.append(close)

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
        adapter = self._bind_selected_resource(operation_id, descriptor)
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

    def _bind_selected_resource(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
    ) -> OperationAdapter[Any] | None:
        if operation_id in _SELECTED_LEAN_OPERATIONS:
            from jacobian.lean_frontend.selected import bind_selected_lean_operation

            adapter = bind_selected_lean_operation(
                operation_id,
                descriptor,
                self.catalog,
                self.binder,
                self.binder.store,
                self.binder.schemas,
                self.verification,
                self.checkers,
                self._own,
            )
        elif operation_id in _SELECTED_GRAPH_OPERATIONS:
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
        elif operation_id in _SELECTED_POLYNOMIAL_OPERATIONS:
            adapter = self._bind_selected_polynomial_operation(operation_id, descriptor)
        elif operation_id == "polytope.separate":
            from jacobian.polytope_operations import PolytopeSeparationAdapter

            adapter = PolytopeSeparationAdapter(self.polytope)
        elif operation_id == "finite.coverage.verify":
            from jacobian.finite_coverage import bind_selected_finite_coverage

            adapter = bind_selected_finite_coverage(
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.verification,
                self.checkers,
                self.catalog,
            )
        elif operation_id in {
            "finite_magma.table.enumerate",
            "universal_algebra.evaluate_laws",
            "universal_algebra.search.countermodel",
            "universal_algebra.law_evaluation.verify",
        }:
            from jacobian.universal_algebra_operations import (
                bind_selected_universal_algebra_operation,
            )

            adapter = bind_selected_universal_algebra_operation(
                operation_id,
                descriptor,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.verification,
                self.checkers,
                self.catalog,
            )
        elif operation_id.startswith(("sat.", "smt.")):
            adapter = self._bind_selected_sat_operation(operation_id, descriptor)
        else:
            adapter = None
        return adapter

    def _bind_selected_polynomial_operation(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
    ) -> OperationAdapter[Any] | None:
        if operation_id == "polynomial.expression.normalize":
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
            return adapter
        if operation_id == "polynomial.expression_normalization.verify":
            from jacobian.polynomial_expression_operations import (
                bind_selected_polynomial_expression_checker,
            )

            return bind_selected_polynomial_expression_checker(
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.polynomial_expressions,
                self.verification,
                self.checkers,
                self.catalog,
            )
        if operation_id in {
            "polynomial.interval.positivity.decide",
            "polynomial.interval.positivity.verify",
        }:
            from jacobian.polynomial_positivity_operations import (
                bind_selected_polynomial_positivity_operation,
            )

            return bind_selected_polynomial_positivity_operation(
                operation_id,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.verification,
                self.checkers,
                self.catalog,
            )
        if operation_id in {
            "polynomial.interval.enclose",
            "polynomial.interval.enclosure.verify",
        }:
            from jacobian.polynomial_interval_operations import (
                bind_selected_polynomial_interval_operation,
            )

            return bind_selected_polynomial_interval_operation(
                operation_id,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.verification,
                self.checkers,
                self.catalog,
            )
        from jacobian.polynomials.installation import (
            bind_selected_polynomial_operation,
        )

        polynomial_adapter = bind_selected_polynomial_operation(
            operation_id,
            self.binder.store,
            self.binder.schemas,
            self.binder.artifacts,
            self.verification,
            self.checkers,
            self.catalog,
        )
        if polynomial_adapter is not None:
            return polynomial_adapter
        from jacobian.polynomial_system_operations import (
            bind_selected_polynomial_system_operation,
        )

        system_adapter = bind_selected_polynomial_system_operation(
            operation_id,
            self.binder.store,
            self.binder.schemas,
            self.binder.artifacts,
            self.verification,
            self.checkers,
            self.catalog,
        )
        if system_adapter is not None:
            return system_adapter
        from jacobian.domains.polynomial_nullstellensatz.core import (
            bind_selected_nullstellensatz_operation,
        )

        if descriptor.provider_runtime is None:
            raise OperationCatalogError(
                "Nullstellensatz provider observation is missing; run `jacobian update`"
            )
        return bind_selected_nullstellensatz_operation(
            operation_id,
            self.binder.store,
            self.binder.schemas,
            self.binder.artifacts,
            self.verification,
            self.checkers,
            self.catalog,
            descriptor.provider_runtime,
        )

    def _bind_selected_sat_operation(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
    ) -> OperationAdapter[Any] | None:
        if operation_id in {"sat.model.find", "sat.unsat_proof.find"}:
            from jacobian.sat_smt.cadical import install_cadical_operations

            if descriptor.provider_runtime is None:
                raise OperationCatalogError(
                    "CaDiCaL provider observation is missing; run `jacobian update`"
                )
            return next(
                adapter
                for adapter in install_cadical_operations(
                    self.sat,
                    descriptor.provider_runtime,
                )
                if adapter.descriptor.operation_id == operation_id
            )
        if operation_id == "smt.unsat_proof.find":
            from jacobian.sat_smt.cvc5 import install_cvc5_operation

            if descriptor.provider_runtime is None:
                raise OperationCatalogError(
                    "cvc5 provider observation is missing; run `jacobian update`"
                )
            return install_cvc5_operation(self.smt, descriptor.provider_runtime)
        if operation_id == "sat.cnf.materialize":
            from jacobian.sat_smt.sat_operations import SatCnfMaterializationAdapter

            return SatCnfMaterializationAdapter(self.sat)
        if operation_id in {"sat.model.verify", "sat.unsat_proof.verify"}:
            from jacobian.sat_smt.sat_operations import (
                bind_selected_sat_verification,
            )

            return bind_selected_sat_verification(
                operation_id,
                descriptor,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.sat,
                self.verification,
                self.checkers,
                self.catalog,
            )
        if operation_id == "sat.lrat.verify":
            from jacobian.sat_smt.sat_lrat import bind_selected_sat_lrat_verifier

            return bind_selected_sat_lrat_verifier(
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.sat,
                self.verification,
                self.checkers,
                self.catalog,
            )
        if operation_id == "smt.unsat_proof.verify":
            from jacobian.sat_smt.smt_operations import (
                bind_selected_smt_unsat_proof_checker,
            )

            return bind_selected_smt_unsat_proof_checker(
                descriptor,
                self.binder.store,
                self.binder.schemas,
                self.binder.artifacts,
                self.smt,
                self.verification,
                self.checkers,
                self.catalog,
            )
        return None

    def _resolve_exact_verifier(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        operations: OperationDeclarations,
        checker_declarations: tuple[AuthorizedChecker, ...],
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
