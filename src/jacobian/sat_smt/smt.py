"""Registration and materialization for pinned-profile SMT artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.operations import ProviderObservation
from jacobian.contracts.smt import (
    SmtAletheProofArtifact,
    SmtLogic,
    SmtProblemArtifact,
    SmtProblemBinding,
    SmtResourceBudget,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


class SmtArtifactError(ValueError):
    """An SMT artifact cannot be created against the requested source."""


@dataclass(frozen=True, slots=True)
class SmtArtifactInstallation:
    semantics_uri: str
    problem_schema_uri: str
    proof_schema_uri: str


@dataclass(frozen=True, slots=True)
class ResolvedSmtProblem:
    artifact: StoredArtifact
    problem: SmtProblemArtifact
    binding: SmtProblemBinding


@dataclass(frozen=True, slots=True)
class ResolvedSmtProof:
    artifact: StoredArtifact
    proof: SmtAletheProofArtifact
    problem_artifact: StoredArtifact


class SmtArtifactService:
    """Materialize exact SMT-LIB inputs and bound unverified Alethe evidence."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        installation: SmtArtifactInstallation,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.installation = installation

    def put_problem(
        self,
        *,
        logic: SmtLogic,
        smtlib_text: str,
    ) -> ArtifactPutResult:
        problem = SmtProblemArtifact.from_text(
            logic=logic,
            smtlib_text=smtlib_text,
        )
        return self.artifacts.put(
            schema_uri=self.installation.problem_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=problem.model_dump(mode="json"),
            summary=f"exact {logic} SMT-LIB single query",
        )

    def resolve_problem(self, problem_uri: str) -> ResolvedSmtProblem:
        try:
            artifact = self.store.get(problem_uri)
        except StorageError as exc:
            raise SmtArtifactError(
                "source is not an available SMT problem artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.problem_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise SmtArtifactError("source is not an SMT problem artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.problem_schema_uri,
                artifact.payload,
            )
            problem = SmtProblemArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise SmtArtifactError(
                "source is not a valid SMT problem artifact"
            ) from exc
        binding = SmtProblemBinding(
            problem_artifact_uri=artifact.artifact_uri,
            problem_object_digest=artifact.manifest.object_digest,
            problem_payload_digest=artifact.manifest.payload_digest,
            logic=problem.logic,
            profile=problem.profile,
            input_language=problem.input_language,
            smtlib_digest=problem.smtlib_digest,
        )
        return ResolvedSmtProblem(
            artifact=artifact,
            problem=problem,
            binding=binding,
        )

    def bind_problem(self, problem_uri: str) -> SmtProblemBinding:
        return self.resolve_problem(problem_uri).binding

    def put_proof(
        self,
        *,
        problem_uri: str,
        proof: bytes,
        producer: ProviderObservation,
        resource_budget: SmtResourceBudget,
    ) -> ArtifactPutResult:
        proof_artifact = SmtAletheProofArtifact.from_bytes(
            problem=self.bind_problem(problem_uri),
            proof=proof,
            producer=producer,
            resource_budget=resource_budget,
        )
        return self.artifacts.put(
            schema_uri=self.installation.proof_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=proof_artifact.model_dump(mode="json"),
            parents=(problem_uri,),
            summary=(
                "unverified raw cvc5 Alethe proof"
                + (
                    f" with {proof_artifact.alethe_hole_count} hole(s)"
                    if proof_artifact.contains_holes
                    else " without lexical hole markers"
                )
            ),
        )

    def resolve_proof(self, proof_uri: str) -> ResolvedSmtProof:
        try:
            artifact = self.store.get(proof_uri)
        except StorageError as exc:
            raise SmtArtifactError(
                "source is not an available SMT proof artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.proof_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise SmtArtifactError("source is not an SMT proof artifact")
        try:
            normalized = self.schemas.validate(
                self.installation.proof_schema_uri,
                artifact.payload,
            )
            proof = SmtAletheProofArtifact.model_validate(normalized)
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise SmtArtifactError("source is not a valid SMT proof artifact") from exc
        binding = self.bind_problem(proof.problem.problem_artifact_uri)
        if proof.problem != binding:
            raise SmtArtifactError(
                "SMT proof binding does not match its exact problem artifact"
            )
        if binding.problem_artifact_uri not in artifact.manifest.parents:
            raise SmtArtifactError("SMT proof is missing its problem parent")
        return ResolvedSmtProof(
            artifact=artifact,
            proof=proof,
            problem_artifact=self.store.get(binding.problem_artifact_uri),
        )


def install_smt_artifacts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> SmtArtifactService:
    """Register the SMT evidence boundary without installing solver or checker."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.smt.qf-unsat",
        version="1",
        definition={
            "domain": (
                "one quantifier-free SMT-LIB 2.6 check-sat query in QF_UF, "
                "QF_LIA, or QF_LRA"
            ),
            "profile": (
                "jacobian.smtlib2.qf-unsat/v1 permits one set-logic, declarations, "
                "assertions, and one final check-sat over exact bounded ASCII bytes"
            ),
            "proof": (
                "raw cvc5 1.3.4 Alethe bytes are bound to the exact problem, "
                "producer runtime, format, and wall-time budget; hole metadata and "
                "storage do not establish UNSAT"
            ),
            "verification": (
                "only a separately authorized compatible checker may promote a "
                "bound proof; no checker is installed by this producer slice"
            ),
        },
    )
    installation = SmtArtifactInstallation(
        semantics_uri=semantics_uri,
        problem_schema_uri=schemas.register_model(
            name="jacobian.smt-problem",
            version="1",
            model=SmtProblemArtifact,
        ),
        proof_schema_uri=schemas.register_model(
            name="jacobian.smt-alethe-proof",
            version="1",
            model=SmtAletheProofArtifact,
        ),
    )
    return SmtArtifactService(store, schemas, artifacts, installation)
