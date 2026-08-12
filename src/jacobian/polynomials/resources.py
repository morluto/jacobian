"""Installation records for polynomial-map capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService


@dataclass(frozen=True, slots=True)
class PolynomialInstallation:
    semantics_uri: str
    identity_semantics_uri: str
    rational_function_identity_semantics_uri: str
    inverse_semantics_uri: str
    map_schema_uri: str
    evaluation_schema_uri: str
    jacobian_schema_uri: str
    claim_schema_uri: str
    jacobian_claim_schema_uri: str
    right_polynomial_schema_uri: str
    left_polynomial_schema_uri: str
    identity_claim_schema_uri: str
    rational_function_left_schema_uri: str
    rational_function_right_schema_uri: str
    rational_function_identity_claim_schema_uri: str
    keller_claim_schema_uri: str
    inverse_collision_claim_schema_uri: str
    inverse_claim_schema_uri: str
    inverse_residual_schema_uri: str
    inverse_synthesis_schema_uri: str
    witness_schema_uri: str
    certificate_schema_uri: str
    collision_checker_id: str | None
    jacobian_checker_id: str | None
    keller_checker_id: str | None
    identity_checker_id: str | None
    rational_function_identity_checker_id: str | None
    inverse_checker_id: str | None
    inverse_collision_checker_id: str | None


@dataclass(frozen=True, slots=True)
class PolynomialResources:
    store: ArtifactRepository
    artifacts: ArtifactService
    verification: VerificationService
    installation: PolynomialInstallation
