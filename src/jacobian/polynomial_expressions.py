"""Durable typed polynomial expressions and normalization candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.operations import ProviderObservation
from jacobian.contracts.polynomial_expressions import (
    PolynomialExpressionArtifact,
    PolynomialExpressionBinding,
    PolynomialExpressionNormalizationArtifact,
    PolynomialExpressionResourceBudget,
    analyze_polynomial_expression,
)
from jacobian.contracts.polynomials import SparseRationalPolynomial
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


class PolynomialExpressionArtifactError(ValueError):
    """Stored data does not satisfy the typed polynomial-expression contract."""


@dataclass(frozen=True, slots=True)
class PolynomialExpressionInstallation:
    semantics_uri: str
    expression_schema_uri: str
    normalization_schema_uri: str


@dataclass(frozen=True, slots=True)
class ResolvedPolynomialExpression:
    artifact: StoredArtifact
    expression: PolynomialExpressionArtifact
    binding: PolynomialExpressionBinding


@dataclass(frozen=True, slots=True)
class ResolvedPolynomialNormalization:
    artifact: StoredArtifact
    candidate: PolynomialExpressionNormalizationArtifact
    expression_artifact: StoredArtifact
    expression: PolynomialExpressionArtifact


class PolynomialExpressionArtifactService:
    """Materialize typed expressions and unverified canonicalization evidence."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        installation: PolynomialExpressionInstallation,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.installation = installation

    def put_expression(
        self,
        expression: PolynomialExpressionArtifact | dict[str, Any],
    ) -> ArtifactPutResult:
        validated = PolynomialExpressionArtifact.model_validate(expression)
        analysis = analyze_polynomial_expression(
            validated.expression,
            validated.variables,
        )
        return self.artifacts.put(
            schema_uri=self.installation.expression_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=validated.model_dump(mode="json"),
            summary=(
                "typed rational polynomial expression: "
                f"{len(validated.variables)} variables, "
                f"{analysis.node_count} AST nodes"
            ),
        )

    def resolve_expression(self, expression_uri: str) -> ResolvedPolynomialExpression:
        try:
            artifact = self.store.get(expression_uri)
        except StorageError as exc:
            raise PolynomialExpressionArtifactError(
                "source is not an available typed polynomial-expression artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.expression_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise PolynomialExpressionArtifactError(
                "source is not a typed polynomial-expression artifact"
            )
        try:
            normalized = self.schemas.validate(
                self.installation.expression_schema_uri,
                artifact.payload,
            )
            expression = PolynomialExpressionArtifact.model_validate(normalized)
            analysis = analyze_polynomial_expression(
                expression.expression,
                expression.variables,
            )
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise PolynomialExpressionArtifactError(
                "source is not a valid typed polynomial-expression artifact"
            ) from exc
        binding = PolynomialExpressionBinding(
            expression_artifact_uri=artifact.artifact_uri,
            expression_object_digest=artifact.manifest.object_digest,
            expression_payload_digest=artifact.manifest.payload_digest,
            variables=expression.variables,
            node_count=analysis.node_count,
            depth=analysis.depth,
            expanded_term_upper_bound=analysis.expanded_term_upper_bound,
            coefficient_digit_budget=analysis.coefficient_digit_budget,
        )
        return ResolvedPolynomialExpression(
            artifact=artifact,
            expression=expression,
            binding=binding,
        )

    def put_normalization(
        self,
        *,
        expression_uri: str,
        normalized: SparseRationalPolynomial | dict[str, Any],
        producer: ProviderObservation,
        resource_budget: PolynomialExpressionResourceBudget | dict[str, Any],
    ) -> ArtifactPutResult:
        resolved = self.resolve_expression(expression_uri)
        candidate = PolynomialExpressionNormalizationArtifact(
            source=resolved.binding,
            normalized=SparseRationalPolynomial.model_validate(normalized),
            producer=producer,
            resource_budget=PolynomialExpressionResourceBudget.model_validate(
                resource_budget
            ),
        )
        return self.artifacts.put(
            schema_uri=self.installation.normalization_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=candidate.model_dump(mode="json"),
            parents=(resolved.artifact.artifact_uri,),
            summary="unverified canonical sparse polynomial normalization",
        )

    def resolve_normalization(
        self,
        normalization_uri: str,
    ) -> ResolvedPolynomialNormalization:
        try:
            artifact = self.store.get(normalization_uri)
        except StorageError as exc:
            raise PolynomialExpressionArtifactError(
                "source is not an available polynomial-normalization artifact"
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.normalization_schema_uri
            or artifact.manifest.semantics_uri != self.installation.semantics_uri
        ):
            raise PolynomialExpressionArtifactError(
                "source is not a polynomial-normalization artifact"
            )
        try:
            normalized = self.schemas.validate(
                self.installation.normalization_schema_uri,
                artifact.payload,
            )
            candidate = PolynomialExpressionNormalizationArtifact.model_validate(
                normalized
            )
        except (SchemaRegistryError, ValueError, ValidationError) as exc:
            raise PolynomialExpressionArtifactError(
                "source is not a valid polynomial-normalization artifact"
            ) from exc
        resolved_expression = self.resolve_expression(
            candidate.source.expression_artifact_uri
        )
        if candidate.source != resolved_expression.binding:
            raise PolynomialExpressionArtifactError(
                "normalization binding does not match its exact source expression"
            )
        if resolved_expression.artifact.artifact_uri not in artifact.manifest.parents:
            raise PolynomialExpressionArtifactError(
                "normalization is missing its exact source-expression parent"
            )
        return ResolvedPolynomialNormalization(
            artifact=artifact,
            candidate=candidate,
            expression_artifact=resolved_expression.artifact,
            expression=resolved_expression.expression,
        )


def install_polynomial_expression_artifacts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> PolynomialExpressionArtifactService:
    """Register one bounded typed QQ-expression and normalization contract."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.typed-rational-polynomial-expression-normalization",
        version="1",
        definition={
            "domain": (
                "commutative polynomial rings QQ[x_1,...,x_n] with one explicit "
                "ordered variable tuple"
            ),
            "expression_ast": {
                "version": "1",
                "nodes": [
                    "rational",
                    "variable",
                    "add",
                    "multiply",
                    "negate",
                    "power",
                ],
                "power_exponents": "nonnegative integers; every base to power zero is one",
                "excluded": [
                    "string parsing",
                    "expression division",
                    "functions",
                    "assumptions",
                    "branch-sensitive operations",
                ],
            },
            "normalization": (
                "combine exact rational coefficients by exponent tuple and emit "
                "nonzero terms in descending lexicographic exponent order"
            ),
            "candidate": (
                "SymPy output is computed evidence; only an operator-authorized "
                "independent checker may verify equality with the bound AST"
            ),
            "limits": {
                "maximum_variables": 4,
                "maximum_nodes": 128,
                "maximum_depth": 16,
                "maximum_operands_per_node": 16,
                "maximum_power": 32,
                "maximum_expanded_terms": 1024,
                "maximum_exponent_per_variable": 127,
                "maximum_decimal_digits_per_rational_component": 256,
                "maximum_coefficient_digit_budget": 4096,
            },
        },
    )
    installation = PolynomialExpressionInstallation(
        semantics_uri=semantics_uri,
        expression_schema_uri=schemas.register_model(
            name="jacobian.typed-rational-polynomial-expression",
            version="1",
            model=PolynomialExpressionArtifact,
        ),
        normalization_schema_uri=schemas.register_model(
            name="jacobian.polynomial-expression-normalization",
            version="1",
            model=PolynomialExpressionNormalizationArtifact,
        ),
    )
    return PolynomialExpressionArtifactService(
        store,
        schemas,
        artifacts,
        installation,
    )
