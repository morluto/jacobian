"""Owned resources for one selected-operation execution runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.catalog_operation_collector import CatalogOperationCollector
from jacobian.operation_binding import OperationBinder
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore


@dataclass(slots=True)
class RuntimeResources:
    """Foundational resources with one explicit close boundary."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    binder: OperationBinder
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    checkers: CheckerRegistry
    operations: CatalogOperationCollector | OperationDispatcher

    def close(self) -> None:
        failures: list[Exception] = []
        close_operations = getattr(self.operations, "close", None)
        for close in (
            close_operations if callable(close_operations) else None,
            self.values.close,
            self.store.close,
        ):
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("runtime resources failed to close", failures)


__all__ = ["RuntimeResources"]
