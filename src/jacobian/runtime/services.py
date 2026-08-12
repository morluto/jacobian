"""Small runtime-owned service graph for the mathematical portfolio."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityService
from jacobian.operation_installation import OperationInstaller
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
    """Foundational persistence and registries shared across the portfolio."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    operations: OperationInstaller
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    checkers: CheckerRegistry
    capabilities: CapabilityService

    def close(self) -> None:
        self.values.close()
        self.store.close()


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Retained mathematical services."""

    core: CoreServices
    polytope: PolytopeService
    verification: VerificationService

    def close(self) -> None:
        """Runtime services own no workers beyond portfolio and core owners."""


def build_runtime_services(core: CoreServices) -> RuntimeServices:
    """Construct only services owned by the surviving mathematical product."""

    return RuntimeServices(
        core=core,
        polytope=PolytopeService(core.store, core.schemas),
        verification=VerificationService(
            core.store,
            core.checkers,
            checker_timeout_seconds=105,
        ),
    )


__all__ = ["CoreServices", "RuntimeServices", "build_runtime_services"]
