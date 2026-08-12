"""Registration and materialization for canonical SAT artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.sat import (
    CanonicalCnf,
    SatAssignmentArtifact,
    SatCnfBinding,
    SatProofArtifact,
    SatResourceBudget,
    canonicalize_cnf,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


class SatArtifactError(ValueError):
    """A SAT artifact cannot be created against the requested source."""


@dataclass(frozen=True, slots=True)
class SatArtifactInstallation:
    semantics_uri: str
    cnf_schema_uri: str
    assignment_schema_uri: str
    proof_schema_uri: str


@dataclass(frozen=True, slots=True)
class ResolvedSatCnf:
    artifact: StoredArtifact
    cnf: CanonicalCnf
    binding: SatCnfBinding


@dataclass(frozen=True, slots=True)
class ResolvedSatAssignment:
    artifact: StoredArtifact
    assignment: SatAssignmentArtifact
    cnf_artifact: StoredArtifact


@dataclass(frozen=True, slots=True)
class ResolvedSatProof:
    artifact: StoredArtifact
    proof: SatProofArtifact
    cnf_artifact: StoredArtifact


class SatArtifactService:
    """Materialize SAT instances and unverified evidence with exact bindings."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        installation: SatArtifactInstallation,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.installation = installation

    def put_cnf(
        self,
        *,
        variable_names: Sequence[str],
        clauses: Iterable[Iterable[int]],
    ) -> ArtifactPutResult:
        """Canonicalize and materialize one named CNF instance."""

        cnf = canonicalize_cnf(variable_names=variable_names, clauses=clauses)
        return self.put_canonical_cnf(cnf)

    def put_canonical_cnf(self, cnf: CanonicalCnf) -> ArtifactPutResult:
        """Materialize an already validated canonical CNF value."""

        return self.artifacts.put(
            schema_uri=self.installation.cnf_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=cnf.model_dump(mode="json"),
            summary=(
                f"canonical CNF: {len(cnf.variables)} variables, "
                f"{len(cnf.clauses)} clauses"
            ),
        )

    def resolve_cnf(self, cnf_uri: str) -> ResolvedSatCnf:
        """Resolve one valid canonical CNF and its exact stored identity."""

        try:
            artifact = self.store.get(cnf_uri)
        except StorageError as exc:
            raise SatArtifactError(
                "source is not an available canonical CNF artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.cnf_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise SatArtifactError("source is not a canonical CNF artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.cnf_schema_uri,
                artifact.payload,
            )
            cnf = CanonicalCnf.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise SatArtifactError(
                "source is not a valid canonical CNF artifact"
            ) from exc
        binding = SatCnfBinding(
            cnf_artifact_uri=artifact.artifact_uri,
            cnf_object_digest=artifact.manifest.object_digest,
            cnf_payload_digest=artifact.manifest.payload_digest,
            variable_map_digest=cnf.variable_map_digest,
            dimacs_digest=cnf.dimacs_digest,
            projection_format=cnf.projection_format,
            projection_version=cnf.projection_version,
            variable_count=len(cnf.variables),
            clause_count=len(cnf.clauses),
        )
        return ResolvedSatCnf(
            artifact=artifact,
            cnf=cnf,
            binding=binding,
        )

    def bind_cnf(self, cnf_uri: str) -> SatCnfBinding:
        """Resolve one canonical CNF and bind its exact stored identity."""

        return self.resolve_cnf(cnf_uri).binding

    def put_assignment(
        self,
        *,
        cnf_uri: str,
        values: Sequence[bool],
        producer: CapabilityProviderRuntime,
        resource_budget: SatResourceBudget,
    ) -> ArtifactPutResult:
        """Materialize one total assignment candidate without verifying it."""

        assignment = SatAssignmentArtifact(
            cnf=self.bind_cnf(cnf_uri),
            values=tuple(values),
            producer=producer,
            resource_budget=resource_budget,
        )
        return self.artifacts.put(
            schema_uri=self.installation.assignment_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=assignment.model_dump(mode="json"),
            parents=(cnf_uri,),
            summary="unverified SAT assignment candidate",
        )

    def resolve_assignment(self, assignment_uri: str) -> ResolvedSatAssignment:
        """Resolve an assignment whose payload and lineage bind one exact CNF."""

        try:
            artifact = self.store.get(assignment_uri)
        except StorageError as exc:
            raise SatArtifactError(
                "source is not an available SAT assignment artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.assignment_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise SatArtifactError("source is not a SAT assignment artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.assignment_schema_uri,
                artifact.payload,
            )
            assignment = SatAssignmentArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise SatArtifactError(
                "source is not a valid SAT assignment artifact"
            ) from exc
        binding = self.bind_cnf(assignment.cnf.cnf_artifact_uri)
        if assignment.cnf != binding:
            raise SatArtifactError(
                "SAT assignment binding does not match its exact canonical CNF"
            )
        if binding.cnf_artifact_uri not in artifact.manifest.parents:
            raise SatArtifactError("SAT assignment is missing its canonical CNF parent")
        return ResolvedSatAssignment(
            artifact=artifact,
            assignment=assignment,
            cnf_artifact=self.store.get(binding.cnf_artifact_uri),
        )

    def put_proof(
        self,
        *,
        cnf_uri: str,
        proof: bytes,
        producer: CapabilityProviderRuntime,
        resource_budget: SatResourceBudget,
    ) -> ArtifactPutResult:
        """Preserve raw DRAT bytes without interpreting or verifying them."""

        proof_artifact = SatProofArtifact.from_bytes(
            cnf=self.bind_cnf(cnf_uri),
            proof=proof,
            producer=producer,
            resource_budget=resource_budget,
        )
        return self.artifacts.put(
            schema_uri=self.installation.proof_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=proof_artifact.model_dump(mode="json"),
            parents=(cnf_uri,),
            summary="unverified raw DRAT proof",
        )

    def resolve_proof(self, proof_uri: str) -> ResolvedSatProof:
        """Resolve raw proof bytes whose payload and lineage bind one exact CNF."""

        try:
            artifact = self.store.get(proof_uri)
        except StorageError as exc:
            raise SatArtifactError(
                "source is not an available SAT proof artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.proof_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise SatArtifactError("source is not a SAT proof artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.proof_schema_uri,
                artifact.payload,
            )
            proof = SatProofArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise SatArtifactError("source is not a valid SAT proof artifact") from exc
        binding = self.bind_cnf(proof.cnf.cnf_artifact_uri)
        if proof.cnf != binding:
            raise SatArtifactError(
                "SAT proof binding does not match its exact canonical CNF"
            )
        if binding.cnf_artifact_uri not in artifact.manifest.parents:
            raise SatArtifactError("SAT proof is missing its canonical CNF parent")
        return ResolvedSatProof(
            artifact=artifact,
            proof=proof,
            cnf_artifact=self.store.get(binding.cnf_artifact_uri),
        )


def install_sat_artifacts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> SatArtifactService:
    """Register the SAT artifact boundary without installing a solver or checker."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.sat",
        version="1",
        definition={
            "domain": "propositional Boolean satisfiability in conjunctive normal form",
            "cnf": (
                "variables use contiguous DIMACS IDs ordered by NFC-normalized names; "
                "literals and clauses are unique and canonically ordered; duplicate "
                "literals and clauses are removed; tautological clauses are omitted"
            ),
            "projection": (
                "DIMACS-CNF bytes use jacobian.dimacs.cnf/v1 and are bound by digest"
            ),
            "assignment": (
                "a total unverified candidate bound to the exact CNF artifact, "
                "variable map, projection, producer runtime, scope, and resource budget"
            ),
            "proof": (
                "raw DRAT bytes preserved as canonical base64 and bound to the exact "
                "CNF artifact, projection, format version, producer runtime, scope, "
                "and resource budget; storage alone does not establish UNSAT"
            ),
        },
    )
    installation = SatArtifactInstallation(
        semantics_uri=semantics_uri,
        cnf_schema_uri=schemas.register_model(
            name="jacobian.canonical-cnf",
            version="1",
            model=CanonicalCnf,
        ),
        assignment_schema_uri=schemas.register_model(
            name="jacobian.sat-assignment",
            version="1",
            model=SatAssignmentArtifact,
        ),
        proof_schema_uri=schemas.register_model(
            name="jacobian.sat-proof",
            version="1",
            model=SatProofArtifact,
        ),
    )
    return SatArtifactService(store, schemas, artifacts, installation)
