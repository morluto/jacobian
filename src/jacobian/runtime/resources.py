"""Owned resources for one selected-operation execution runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from jacobian.artifacts import ArtifactService
from jacobian.catalog.collector import CatalogOperationCollector
from jacobian.operation_binding import OperationBinder
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.polynomial_expressions import (
    PolynomialExpressionArtifactService,
    install_polynomial_expression_artifacts,
)
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService, install_sat_artifacts
from jacobian.sat_smt.smt import SmtArtifactService, install_smt_artifacts
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
    checkers: CheckerRegistry
    operations: CatalogOperationCollector | OperationDispatcher | None
    sat: SatArtifactService | None = None
    smt: SmtArtifactService | None = None
    polynomial_expressions: PolynomialExpressionArtifactService | None = None
    _owned_resources: list[object] | None = None

    def ensure_family_artifacts(self) -> None:
        """Install SAT/SMT/polynomial artifact contracts on first family use."""

        if (
            self.sat is not None
            and self.smt is not None
            and self.polynomial_expressions is not None
        ):
            return
        with self.store.transaction():
            if self.sat is None:
                self.sat = install_sat_artifacts(
                    self.store, self.schemas, self.artifacts
                )
            if self.smt is None:
                self.smt = install_smt_artifacts(
                    self.store, self.schemas, self.artifacts
                )
            if self.polynomial_expressions is None:
                self.polynomial_expressions = install_polynomial_expression_artifacts(
                    self.store,
                    self.schemas,
                    self.artifacts,
                )

    def own(self, resource: object) -> None:
        """Add one lazily acquired closeable to the runtime lifecycle."""

        if not callable(getattr(resource, "close", None)):
            raise TypeError("runtime-owned resource must be closeable")
        if self._owned_resources is None:
            self._owned_resources = []
        if all(owned is not resource for owned in self._owned_resources):
            self._owned_resources.append(resource)

    def close(self) -> None:
        failures: list[Exception] = []
        close_operations = getattr(self.operations, "close", None)
        owned_closes = [
            cast(Any, resource).close
            for resource in reversed(self._owned_resources or [])
        ]
        closes: list[object] = [
            close_operations if callable(close_operations) else None,
            *owned_closes,
            self.values.close,
            self.store.close,
        ]
        for close in closes:
            if close is None:
                continue
            assert callable(close)
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("runtime resources failed to close", failures)


__all__ = ["RuntimeResources"]
