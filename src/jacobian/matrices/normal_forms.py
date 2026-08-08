"""Durable integer matrices and Hermite-normal-form candidates.

Ownership: ``jacobian.matrices`` (HNF artifact service).
Provides the artifact-backed storage and retrieval layer for integer
matrices and HNF candidates used by the Python-FLINT HNF producer and
its independent checker.  Installed during runtime bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.matrices import (
    IntegerMatrix,
    MatrixBinding,
    MatrixHermiteNormalFormArtifact,
    MatrixHermiteResourceBudget,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


class MatrixNormalFormArtifactError(ValueError):
    """Stored data does not satisfy the integer-matrix HNF contract."""


@dataclass(frozen=True, slots=True)
class MatrixNormalFormInstallation:
    semantics_uri: str
    matrix_schema_uri: str
    normal_form_schema_uri: str


@dataclass(frozen=True, slots=True)
class ResolvedIntegerMatrix:
    artifact: StoredArtifact
    matrix: IntegerMatrix
    binding: MatrixBinding


@dataclass(frozen=True, slots=True)
class ResolvedHermiteNormalForm:
    artifact: StoredArtifact
    candidate: MatrixHermiteNormalFormArtifact
    matrix_artifact: StoredArtifact
    matrix: IntegerMatrix


class MatrixNormalFormArtifactService:
    """Materialize exact matrices and unverified HNF transformation evidence."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        installation: MatrixNormalFormInstallation,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.installation = installation

    def put_matrix(
        self,
        matrix: IntegerMatrix | dict[str, Any],
    ) -> ArtifactPutResult:
        validated = IntegerMatrix.model_validate(matrix)
        return self.artifacts.put(
            schema_uri=self.installation.matrix_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=validated.model_dump(mode="json"),
            summary=(
                "exact integer matrix: "
                f"{len(validated.entries)} by {len(validated.entries[0])}"
            ),
        )

    def resolve_matrix(self, matrix_uri: str) -> ResolvedIntegerMatrix:
        try:
            artifact = self.store.get(matrix_uri)
        except StorageError as exc:
            raise MatrixNormalFormArtifactError(
                "source is not an available exact integer-matrix artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.matrix_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise MatrixNormalFormArtifactError(
                "source is not an exact integer-matrix artifact"
            )
        try:
            normalized = self.schemas.validate(
                self.installation.matrix_schema_uri,
                artifact.payload,
            )
            matrix = IntegerMatrix.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise MatrixNormalFormArtifactError(
                "source is not a valid exact integer-matrix artifact"
            ) from exc
        binding = MatrixBinding(
            matrix_artifact_uri=artifact.artifact_uri,
            matrix_object_digest=artifact.manifest.object_digest,
            matrix_payload_digest=artifact.manifest.payload_digest,
            row_count=len(matrix.entries),
            column_count=len(matrix.entries[0]),
        )
        return ResolvedIntegerMatrix(
            artifact=artifact,
            matrix=matrix,
            binding=binding,
        )

    def put_hermite_normal_form(
        self,
        *,
        matrix_uri: str,
        normal_form: Sequence[Sequence[Any]],
        transformation: Sequence[Sequence[Any]],
        producer: CapabilityProviderRuntime,
        resource_budget: MatrixHermiteResourceBudget | dict[str, Any],
    ) -> ArtifactPutResult:
        resolved = self.resolve_matrix(matrix_uri)
        candidate = MatrixHermiteNormalFormArtifact(
            source=resolved.binding,
            normal_form=IntegerMatrix(
                entries=tuple(tuple(value for value in row) for row in normal_form)
            ),
            transformation=IntegerMatrix(
                entries=tuple(tuple(value for value in row) for row in transformation)
            ),
            producer=producer,
            resource_budget=MatrixHermiteResourceBudget.model_validate(resource_budget),
        )
        return self.artifacts.put(
            schema_uri=self.installation.normal_form_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=candidate.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri,),
            summary="unverified row Hermite normal form with left transformation",
        )

    def resolve_hermite_normal_form(
        self,
        normal_form_uri: str,
    ) -> ResolvedHermiteNormalForm:
        try:
            artifact = self.store.get(normal_form_uri)
        except StorageError as exc:
            raise MatrixNormalFormArtifactError(
                "source is not an available Hermite-normal-form artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.normal_form_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise MatrixNormalFormArtifactError(
                "source is not a Hermite-normal-form artifact"
            )
        try:
            normalized = self.schemas.validate(
                self.installation.normal_form_schema_uri,
                artifact.payload,
            )
            candidate = MatrixHermiteNormalFormArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise MatrixNormalFormArtifactError(
                "source is not a valid Hermite-normal-form artifact"
            ) from exc
        resolved_matrix = self.resolve_matrix(candidate.source.matrix_artifact_uri)
        if candidate.source != resolved_matrix.binding:
            raise MatrixNormalFormArtifactError(
                "normal-form binding does not match its exact source matrix"
            )
        if resolved_matrix.artifact.artifact_uri not in artifact.manifest.parents:
            raise MatrixNormalFormArtifactError(
                "normal form is missing its exact source-matrix parent"
            )
        return ResolvedHermiteNormalForm(
            artifact=artifact,
            candidate=candidate,
            matrix_artifact=resolved_matrix.artifact,
            matrix=resolved_matrix.matrix,
        )


def install_matrix_normal_form_artifacts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> MatrixNormalFormArtifactService:
    """Register the exact integer row-HNF semantics and artifact contracts."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.integer-matrix-row-hermite-normal-form",
        version="1",
        definition={
            "domain": "nonempty rectangular matrices over the integers ZZ",
            "normal_form": (
                "FLINT row Hermite convention: zero rows last; positive pivots "
                "move strictly right; entries above each pivot are in [0, pivot)"
            ),
            "relation": (
                "a candidate H is equivalent to A only when H = U A and U is "
                "unimodular over ZZ"
            ),
            "candidate": (
                "producer output is computed evidence; only an authorized "
                "independent checker may verify the bound relation and form"
            ),
            "limits": {
                "maximum_rows": 32,
                "maximum_columns": 32,
                "maximum_decimal_digits_per_entry": 256,
            },
        },
    )
    installation = MatrixNormalFormInstallation(
        semantics_uri=semantics_uri,
        matrix_schema_uri=schemas.register_model(
            name="jacobian.exact-integer-matrix",
            version="1",
            model=IntegerMatrix,
        ),
        normal_form_schema_uri=schemas.register_model(
            name="jacobian.matrix-row-hermite-normal-form",
            version="1",
            model=MatrixHermiteNormalFormArtifact,
        ),
    )
    return MatrixNormalFormArtifactService(
        store,
        schemas,
        artifacts,
        installation,
    )
