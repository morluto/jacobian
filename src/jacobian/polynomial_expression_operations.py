"""Independent verification for typed polynomial-expression normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from jacobian.artifacts import ArtifactService
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.contracts.polynomial_expressions import (
    PolynomialExpressionNormalizationVerificationOutput,
    PolynomialExpressionNormalizationVerificationRequest,
)
from jacobian.contracts.results import Conclusion, ExecutionStatus
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection
from jacobian.polynomial_expressions import (
    PolynomialExpressionArtifactError,
    PolynomialExpressionArtifactService,
)
from jacobian.polynomials._support import PolynomialOperationResult
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


@dataclass(frozen=True, slots=True)
class PolynomialExpressionCheckerInstallation:
    witness_schema_uri: str
    checker_id: str | None


def register_polynomial_expression_checker_resources(
    schemas: SchemaRegistry,
) -> PolynomialExpressionCheckerInstallation:
    """Register the passive witness contract without checker installation."""

    return PolynomialExpressionCheckerInstallation(
        witness_schema_uri=schemas.register_model(
            name="jacobian.witness-envelope",
            version="1",
            model=WitnessEnvelope,
        ),
        checker_id=None,
    )


def bind_selected_polynomial_expression_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    expressions: PolynomialExpressionArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[Any]:
    """Bind exact normalization verification from persisted checker authority."""

    operation_id = "polynomial.expression_normalization.verify"
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is missing; run `jacobian update`: {operation_id}"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    installation = replace(
        register_polynomial_expression_checker_resources(schemas),
        checker_id=binding.checker_id,
    )
    return PolynomialExpressionNormalizationVerificationAdapter(
        store=store,
        artifacts=artifacts,
        expressions=expressions,
        verification=verification,
        installation=installation,
    )


def install_polynomial_expression_checker(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    expressions: PolynomialExpressionArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    OperationAdapter[Any] | None,
    PolynomialExpressionCheckerInstallation,
]:
    """Install the witness schema and optionally authorize exact AST replay."""

    resources = register_polynomial_expression_checker_resources(schemas)
    checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="typed rational polynomial normalization replay checker",
            entrypoint=(
                "jacobian_checkers.polynomial_expressions:"
                "check_polynomial_expression_normalization"
            ),
            evidence_kind=EvidenceKind.WITNESS,
            format_id="polynomial.expression_normalization",
            format_version="1",
            claim_schema_uris=(expressions.installation.expression_schema_uri,),
            semantics_uris=(expressions.installation.semantics_uri,),
            candidate_schema_uris=(expressions.installation.normalization_schema_uri,),
            reason=(
                "bundled independent exact typed-AST expansion and coefficient checker"
            ),
        ),
        authorize=authorize_checker,
    ).checker_id
    installation = PolynomialExpressionCheckerInstallation(
        witness_schema_uri=resources.witness_schema_uri,
        checker_id=checker_id,
    )
    adapter: OperationAdapter[Any] | None = None
    if checker_id is not None:
        adapter = PolynomialExpressionNormalizationVerificationAdapter(
            store=store,
            artifacts=artifacts,
            expressions=expressions,
            verification=verification,
            installation=installation,
        )
    return adapter, installation


class PolynomialExpressionNormalizationVerificationAdapter:
    """Verify canonical coefficients against every node of one bound AST."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        expressions: PolynomialExpressionArtifactService,
        verification: VerificationService,
        installation: PolynomialExpressionCheckerInstallation,
    ) -> None:
        checker_id = installation.checker_id
        if checker_id is None:
            raise ValueError("polynomial expression checker is not authorized")
        self.store = store
        self.artifacts = artifacts
        self.expressions = expressions
        self.verification = verification
        self.installation = installation
        self._descriptor = OperationDescriptor(
            operation_id="polynomial.expression_normalization.verify",
            version="1",
            title="Verify a typed polynomial normalization",
            description=(
                "Independently expand the complete typed QQ-polynomial AST using "
                "exact standard-library rational arithmetic and compare every "
                "canonical sparse coefficient."
            ),
            provider="jacobian.polynomial-expression",
            provider_runtime=known_provider_runtime(
                "jacobian.polynomial-expression",
                features=(
                    "typed-ast-replay",
                    "exact-rational",
                    "canonical-sparse-polynomial",
                    "clean-process-checker",
                ),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(
                PolynomialExpressionNormalizationVerificationRequest
            ),
            output_schema=model_schema(
                PolynomialExpressionNormalizationVerificationOutput
            ),
            tags=(
                "polynomial",
                "symbolic",
                "normalization",
                "typed-expression",
                "exact-rational",
                "verification",
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(
        self, request: OperationRequest
    ) -> PolynomialExpressionNormalizationVerificationRequest:
        return parse_operation_input(
            PolynomialExpressionNormalizationVerificationRequest,
            request.input,
        )

    def invoke(
        self, validated: PolynomialExpressionNormalizationVerificationRequest
    ) -> OperationProjection:
        try:
            resolved = self.expressions.resolve_normalization(
                validated.normalization_uri
            )
            semantics = self.store.get(self.expressions.installation.semantics_uri)
        except (PolynomialExpressionArtifactError, StorageError) as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_POLYNOMIAL_EXPRESSION_NORMALIZATION",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="normalization_uri",
                    schema_uri=(self.expressions.installation.normalization_schema_uri),
                    expected=(
                        "one canonical sparse polynomial candidate bound by payload "
                        "and lineage to a valid typed QQ-polynomial expression"
                    ),
                    hint=(
                        "Use polynomial.expression.normalize or materialize a candidate "
                        "with the registered normalization schema."
                    ),
                )
            ) from exc

        checker_id = self.installation.checker_id
        if checker_id is None:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="POLYNOMIAL_EXPRESSION_CHECKER_UNAVAILABLE",
                    stage="normalization_verification",
                    message=(
                        "The independent polynomial-expression checker is not installed "
                        "in this runtime."
                    ),
                )
            )
        bindings = EvidenceBindings(
            claim_digest=resolved.expression_artifact.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=resolved.artifact.manifest.object_digest,
        )
        witness = WitnessEnvelope(
            witness_format="polynomial.expression_normalization",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=bindings,
            payload={
                "expression_uri": resolved.expression_artifact.artifact_uri,
                "normalization_uri": resolved.artifact.artifact_uri,
            },
        )
        witness_artifact = self.artifacts.put(
            schema_uri=self.installation.witness_schema_uri,
            semantics_uri=self.expressions.installation.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(
                resolved.expression_artifact.artifact_uri,
                resolved.artifact.artifact_uri,
            ),
            summary="typed polynomial normalization verification witness",
        )
        checked = self.verification.verify_witness(
            claim_uri=resolved.expression_artifact.artifact_uri,
            candidate_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            include_artifact_metadata=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.verification_record_uri is not None
        )
        status: Literal[
            "VERIFIED_NORMALIZATION",
            "REJECTED",
            "TIMEOUT",
            "CANCELLED",
            "ERROR",
        ]
        if verified:
            status = "VERIFIED_NORMALIZATION"
        elif checked.execution.status is ExecutionStatus.COMPLETED:
            status = "REJECTED"
        elif checked.execution.status is ExecutionStatus.TIMEOUT:
            status = "TIMEOUT"
        elif checked.execution.status is ExecutionStatus.CANCELLED:
            status = "CANCELLED"
        else:
            status = "ERROR"
        detail = checked.execution.detail
        if detail is None and checked.input.errors:
            detail = checked.input.errors[0]
        if detail is None:
            detail = (
                "the authorized checker accepted the exact AST normalization"
                if verified
                else "the candidate was not independently accepted"
            )
        output = PolynomialExpressionNormalizationVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            expression_uri=resolved.expression_artifact.artifact_uri,
            normalization_uri=resolved.artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
            verification_record_uri=(
                checked.verification_record_uri if verified else None
            ),
            detail=detail,
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = [
            resolved.expression_artifact.artifact_uri,
            resolved.artifact.artifact_uri,
            witness_artifact.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return PolynomialOperationResult(
            execution=checked.execution,
            value=output,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=tuple(artifact_uris),
        ).project(self.descriptor)
