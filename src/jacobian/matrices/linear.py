"""Durable exact-rational linear systems and candidate solutions.

Ownership: ``jacobian.matrices`` (linear artifact service).
Provides the artifact-backed storage and retrieval layer for rational
linear systems and solution/inconsistency candidates.  Installed during
runtime bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.linear import (
    LinearRationalInconsistencyArtifact,
    LinearRationalResourceBudget,
    LinearRationalSolutionArtifact,
    LinearRationalSystem,
    LinearSystemBinding,
    linear_variable_order_digest,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


class LinearArtifactError(ValueError):
    """A stored value does not satisfy the rational linear-system contract."""


@dataclass(frozen=True, slots=True)
class LinearArtifactInstallation:
    semantics_uri: str
    system_schema_uri: str
    solution_schema_uri: str
    inconsistency_schema_uri: str


@dataclass(frozen=True, slots=True)
class ResolvedLinearSystem:
    artifact: StoredArtifact
    system: LinearRationalSystem
    binding: LinearSystemBinding


@dataclass(frozen=True, slots=True)
class ResolvedLinearSolution:
    artifact: StoredArtifact
    solution: LinearRationalSolutionArtifact
    system_artifact: StoredArtifact
    system: LinearRationalSystem


@dataclass(frozen=True, slots=True)
class ResolvedLinearInconsistency:
    artifact: StoredArtifact
    certificate: LinearRationalInconsistencyArtifact
    system_artifact: StoredArtifact
    system: LinearRationalSystem


class LinearArtifactService:
    """Materialize exact systems and unverified evidence with strict bindings."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        installation: LinearArtifactInstallation,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.installation = installation

    def put_system(
        self, system: LinearRationalSystem | dict[str, Any]
    ) -> ArtifactPutResult:
        """Validate and materialize one canonical rational system."""

        validated = LinearRationalSystem.model_validate(system)
        return self.artifacts.put(
            schema_uri=self.installation.system_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=validated.model_dump(mode="json"),
            summary=(
                "exact rational linear system: "
                f"{len(validated.coefficients.entries)} equations, "
                f"{len(validated.variables)} variables"
            ),
        )

    def resolve_system(self, system_uri: str) -> ResolvedLinearSystem:
        """Resolve one valid system and bind its exact stored identity."""

        try:
            artifact = self.store.get(system_uri)
        except StorageError as exc:
            raise LinearArtifactError(
                "source is not an available rational linear-system artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.system_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise LinearArtifactError("source is not a rational linear-system artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.system_schema_uri,
                artifact.payload,
            )
            system = LinearRationalSystem.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise LinearArtifactError(
                "source is not a valid rational linear-system artifact"
            ) from exc
        binding = LinearSystemBinding(
            system_artifact_uri=artifact.artifact_uri,
            system_object_digest=artifact.manifest.object_digest,
            system_payload_digest=artifact.manifest.payload_digest,
            variable_order_digest=linear_variable_order_digest(system.variables),
            row_count=len(system.coefficients.entries),
            column_count=len(system.variables),
        )
        return ResolvedLinearSystem(
            artifact=artifact,
            system=system,
            binding=binding,
        )

    def put_solution(
        self,
        *,
        system_uri: str,
        values: Sequence[Any],
        producer: CapabilityProviderRuntime,
        resource_budget: LinearRationalResourceBudget | dict[str, Any],
    ) -> ArtifactPutResult:
        """Materialize one total vector candidate without checking the relation."""

        resolved = self.resolve_system(system_uri)
        solution = LinearRationalSolutionArtifact(
            system=resolved.binding,
            values=tuple(values),
            producer=producer,
            resource_budget=LinearRationalResourceBudget.model_validate(
                resource_budget
            ),
        )
        return self.artifacts.put(
            schema_uri=self.installation.solution_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=solution.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri,),
            summary="unverified exact rational solution candidate",
        )

    def resolve_solution(self, solution_uri: str) -> ResolvedLinearSolution:
        """Resolve a vector whose payload and lineage bind one exact system."""

        try:
            artifact = self.store.get(solution_uri)
        except StorageError as exc:
            raise LinearArtifactError(
                "source is not an available rational solution artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.solution_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise LinearArtifactError("source is not a rational solution artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.solution_schema_uri,
                artifact.payload,
            )
            solution = LinearRationalSolutionArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise LinearArtifactError(
                "source is not a valid rational solution artifact"
            ) from exc
        resolved_system = self.resolve_system(solution.system.system_artifact_uri)
        if solution.system != resolved_system.binding:
            raise LinearArtifactError(
                "solution binding does not match its exact rational system"
            )
        if resolved_system.artifact.artifact_uri not in artifact.manifest.parents:
            raise LinearArtifactError(
                "solution is missing its rational linear-system parent"
            )
        return ResolvedLinearSolution(
            artifact=artifact,
            solution=solution,
            system_artifact=resolved_system.artifact,
            system=resolved_system.system,
        )

    def put_inconsistency(
        self,
        *,
        system_uri: str,
        left_witness: Sequence[Any],
        rhs_pairing: Any,
        producer: CapabilityProviderRuntime,
        resource_budget: LinearRationalResourceBudget | dict[str, Any],
    ) -> ArtifactPutResult:
        """Materialize one normalized inconsistency witness without checking it."""

        resolved = self.resolve_system(system_uri)
        certificate = LinearRationalInconsistencyArtifact(
            system=resolved.binding,
            left_witness=tuple(left_witness),
            rhs_pairing=rhs_pairing,
            producer=producer,
            resource_budget=LinearRationalResourceBudget.model_validate(
                resource_budget
            ),
        )
        return self.artifacts.put(
            schema_uri=self.installation.inconsistency_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri,),
            summary="unverified exact rational inconsistency certificate",
        )

    def resolve_inconsistency(
        self,
        certificate_uri: str,
    ) -> ResolvedLinearInconsistency:
        """Resolve a certificate whose payload and lineage bind one exact system."""

        try:
            artifact = self.store.get(certificate_uri)
        except StorageError as exc:
            raise LinearArtifactError(
                "source is not an available rational inconsistency artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.inconsistency_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise LinearArtifactError("source is not a rational inconsistency artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.inconsistency_schema_uri,
                artifact.payload,
            )
            certificate = LinearRationalInconsistencyArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise LinearArtifactError(
                "source is not a valid rational inconsistency artifact"
            ) from exc
        resolved_system = self.resolve_system(certificate.system.system_artifact_uri)
        if certificate.system != resolved_system.binding:
            raise LinearArtifactError(
                "inconsistency binding does not match its exact rational system"
            )
        if resolved_system.artifact.artifact_uri not in artifact.manifest.parents:
            raise LinearArtifactError(
                "inconsistency certificate is missing its rational-system parent"
            )
        return ResolvedLinearInconsistency(
            artifact=artifact,
            certificate=certificate,
            system_artifact=resolved_system.artifact,
            system=resolved_system.system,
        )


def install_linear_artifacts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> LinearArtifactService:
    """Register exact rational system and evidence-candidate contracts."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.linear-rational-system",
        version="1",
        definition={
            "domain": "finite linear systems over the rational field QQ",
            "relations": {
                "solution": "a candidate x supports the exact claim A x = b",
                "inconsistency": (
                    "a left witness y with y^T A = 0 and y^T b = 1 supports "
                    "that A x = b has no solution"
                ),
            },
            "variable_order": (
                "coefficient columns and candidate entries follow the declared "
                "ordered unique variable list"
            ),
            "candidate": (
                "a total exact rational vector bound by payload and lineage to one "
                "system; producing or storing it does not verify the relation"
            ),
            "inconsistency_certificate": (
                "a normalized exact left witness bound by payload and lineage to "
                "one system; producing or storing it does not verify the relation"
            ),
            "not_found": (
                "failure to produce a candidate makes no consistency or "
                "inconsistency claim"
            ),
            "limits": {
                "maximum_rows": 32,
                "maximum_columns": 32,
                "maximum_decimal_digits_per_rational_component": 256,
            },
        },
    )
    installation = LinearArtifactInstallation(
        semantics_uri=semantics_uri,
        system_schema_uri=schemas.register_model(
            name="jacobian.linear-rational-system",
            version="1",
            model=LinearRationalSystem,
        ),
        solution_schema_uri=schemas.register_model(
            name="jacobian.linear-rational-solution",
            version="1",
            model=LinearRationalSolutionArtifact,
        ),
        inconsistency_schema_uri=schemas.register_model(
            name="jacobian.linear-rational-inconsistency",
            version="1",
            model=LinearRationalInconsistencyArtifact,
        ),
    )
    return LinearArtifactService(store, schemas, artifacts, installation)
