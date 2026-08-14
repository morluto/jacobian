"""Small runtime-owned service graph for selected mathematical operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.operation_binding import OperationBinder
from jacobian.operation_service import OperationService
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore
from jacobian.verification.service import VerificationService


@dataclass(slots=True)
class CoreServices:
    """Foundational persistence and registries shared by selected operations."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    binder: OperationBinder
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    checkers: CheckerRegistry
    operations: OperationService

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
            raise ExceptionGroup("runtime services failed to close", failures)


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Retained mathematical services."""

    core: CoreServices
    polytope: PolytopeService
    verification: VerificationService


def build_runtime_services(core: CoreServices) -> RuntimeServices:
    """Construct only services owned by the surviving mathematical product."""

    return RuntimeServices(
        core=core,
        polytope=PolytopeService(core.store, core.schemas),
        verification=VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        ),
    )


__all__ = ["CoreServices", "RuntimeServices", "build_runtime_services"]
