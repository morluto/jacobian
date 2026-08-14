"""Build polynomial-map resources and catalog descriptors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel

from jacobian.artifacts import ArtifactService
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, WitnessEnvelope
from jacobian.contracts.polynomials import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialIdentityClaim,
    PolynomialInjectivityClaim,
    PolynomialJacobian,
    PolynomialJacobianClaim,
    PolynomialKellerConditionClaim,
    PolynomialMapCompositionResiduals,
    PolynomialMapEvaluation,
    PolynomialMapInverseClaim,
    PolynomialMapInverseSynthesisArtifact,
    PolynomialNoTwoSidedInverseClaim,
    RationalFunctionArtifact,
    RationalFunctionIdentityClaim,
    RationalPolynomial,
    RationalPolynomialMap,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.polynomials.collision import (
    PolynomialCollisionAdapter,
    PolynomialCollisionSearchAdapter,
    PolynomialCollisionVerifyAdapter,
    PolynomialMapInverseCollisionVerifyAdapter,
)
from jacobian.polynomials.evaluation import (
    PolynomialJacobianAdapter,
    PolynomialKellerConditionVerifyAdapter,
    PolynomialMapEvaluationAdapter,
)
from jacobian.polynomials.identity import PolynomialIdentityAdapter
from jacobian.polynomials.inverse import (
    PolynomialMapInverseSynthesizeAdapter,
    PolynomialMapInverseVerifyAdapter,
)
from jacobian.polynomials.rational_identity import RationalFunctionIdentityAdapter
from jacobian.polynomials.resources import PolynomialContracts, PolynomialResources
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_operations import witness_verification_adapter


def register_polynomial_resources(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
) -> PolynomialContracts:
    """Register passive polynomial-map contracts without checker authorization."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.rational-polynomial-map",
        version="1",
        definition={
            "description": (
                "square sparse polynomial maps over QQ with an explicit variable "
                "order and canonical reduced rational coefficients"
            ),
            "domain": "QQ",
            "map_shape": "square",
            "maximum_dimension": MAX_POLYNOMIAL_VARIABLES,
            "maximum_terms_per_coordinate": 1024,
            "maximum_exponent": 32,
            "maximum_derived_exponent": 127,
            "maximum_jacobian_product_term_estimate": 1024,
        },
    )
    identity_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.sparse-rational-polynomial-ring",
        version="1",
        definition={
            "description": (
                "canonical sparse polynomials over QQ in an explicit ordered "
                "tuple of variables"
            ),
            "coefficient_field": "QQ",
            "maximum_dimension": 4,
            "maximum_terms": 1024,
            "maximum_exponent": 127,
            "monomial_order": "descending lexicographic",
            "zero_terms": "omitted",
        },
    )
    rational_function_identity_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.sparse-rational-function-field",
        version="1",
        definition={
            "description": (
                "bounded fractions of canonical sparse polynomials over QQ in "
                "an explicit ordered tuple of variables"
            ),
            "coefficient_field": "QQ",
            "maximum_dimension": 4,
            "maximum_terms_per_polynomial": 1024,
            "maximum_exponent": 127,
            "maximum_cross_product_term_pairs": 4096,
            "denominators": "nonzero polynomials",
            "equality": "exact polynomial cross multiplication",
            "pointwise_definedness": "outside scope",
        },
    )
    inverse_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.rational-polynomial-map-two-sided-inverse",
        version="1",
        definition={
            "description": (
                "two square sparse polynomial maps over QQ are inverse only when "
                "both ordered exact compositions are identity"
            ),
            "coefficient_field": "QQ",
            "directions": ["inverse_after_forward", "forward_after_inverse"],
            "one_sided_identity": "insufficient",
            "synthesis_scope": "bounded polynomial coefficient ansatz only",
            "bounded_no_candidate": "does not prove noninvertibility",
            "rational_map_inverses": "unsupported",
        },
    )
    models: dict[str, tuple[str, type[BaseModel]]] = {
        "map_schema_uri": ("jacobian.rational-polynomial-map", RationalPolynomialMap),
        "evaluation_schema_uri": (
            "jacobian.polynomial-map-evaluation",
            PolynomialMapEvaluation,
        ),
        "jacobian_schema_uri": ("jacobian.polynomial-jacobian", PolynomialJacobian),
        "claim_schema_uri": (
            "jacobian.polynomial-map-injectivity-claim",
            PolynomialInjectivityClaim,
        ),
        "jacobian_claim_schema_uri": (
            "jacobian.polynomial-jacobian-claim",
            PolynomialJacobianClaim,
        ),
        "right_polynomial_schema_uri": (
            "jacobian.sparse-rational-polynomial-right",
            RationalPolynomial,
        ),
        "left_polynomial_schema_uri": (
            "jacobian.sparse-rational-polynomial-left",
            RationalPolynomial,
        ),
        "identity_claim_schema_uri": (
            "jacobian.polynomial-identity-claim",
            PolynomialIdentityClaim,
        ),
        "rational_function_left_schema_uri": (
            "jacobian.sparse-rational-function-left",
            RationalFunctionArtifact,
        ),
        "rational_function_right_schema_uri": (
            "jacobian.sparse-rational-function-right",
            RationalFunctionArtifact,
        ),
        "rational_function_identity_claim_schema_uri": (
            "jacobian.rational-function-identity-claim",
            RationalFunctionIdentityClaim,
        ),
        "keller_claim_schema_uri": (
            "jacobian.polynomial-map-keller-condition-claim",
            PolynomialKellerConditionClaim,
        ),
        "inverse_collision_claim_schema_uri": (
            "jacobian.polynomial-map-no-two-sided-inverse-claim",
            PolynomialNoTwoSidedInverseClaim,
        ),
        "inverse_claim_schema_uri": (
            "jacobian.polynomial-map-inverse-claim",
            PolynomialMapInverseClaim,
        ),
        "inverse_residual_schema_uri": (
            "jacobian.polynomial-map-composition-residuals",
            PolynomialMapCompositionResiduals,
        ),
        "inverse_synthesis_schema_uri": (
            "jacobian.polynomial-map-inverse-synthesis",
            PolynomialMapInverseSynthesisArtifact,
        ),
        "witness_schema_uri": ("jacobian.witness-envelope", WitnessEnvelope),
        "certificate_schema_uri": (
            "jacobian.certificate-envelope",
            CertificateEnvelope,
        ),
    }
    schema_uris = {
        field: schemas.register(name=name, version="1", schema=model_schema(model))
        for field, (name, model) in models.items()
    }
    return PolynomialContracts(
        semantics_uri=semantics_uri,
        identity_semantics_uri=identity_semantics_uri,
        rational_function_identity_semantics_uri=(
            rational_function_identity_semantics_uri
        ),
        inverse_semantics_uri=inverse_semantics_uri,
        **schema_uris,
        collision_checker_id=None,
        jacobian_checker_id=None,
        keller_checker_id=None,
        identity_checker_id=None,
        rational_function_identity_checker_id=None,
        inverse_checker_id=None,
        inverse_collision_checker_id=None,
    )


def bind_selected_polynomial_operation(
    operation_id: str,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[Any] | None:
    """Bind one polynomial-map operation from passive resources and catalog state."""

    adapter_types: dict[str, Callable[[PolynomialResources], OperationAdapter[Any]]] = {
        "polynomial.map.evaluate": PolynomialMapEvaluationAdapter,
        "polynomial.map.compute_jacobian": PolynomialJacobianAdapter,
        "polynomial.map.keller_condition.verify": PolynomialKellerConditionVerifyAdapter,
        "polynomial.map.collision_witness": PolynomialCollisionAdapter,
        "polynomial.map.collision.search": PolynomialCollisionSearchAdapter,
        "polynomial.map.collision.verify": PolynomialCollisionVerifyAdapter,
        "polynomial.map.inverse.refute_by_collision": (
            PolynomialMapInverseCollisionVerifyAdapter
        ),
        "polynomial.identity.verify": PolynomialIdentityAdapter,
        "polynomial.rational_function.identity.verify": RationalFunctionIdentityAdapter,
        "polynomial.map.inverse.candidate_synthesize": (
            PolynomialMapInverseSynthesizeAdapter
        ),
        "polynomial.map.inverse.verify": PolynomialMapInverseVerifyAdapter,
    }
    if operation_id == "polynomial.map.collision_evidence.verify":
        checker_id = _catalog_checker_id(
            catalog,
            checkers,
            "polynomial.map.collision_evidence.verify",
        )
        return witness_verification_adapter(
            operation_id=operation_id,
            title="Verify stored polynomial-map collision evidence",
            description=(
                "Independently replay one exact stored collision witness against "
                "its bound map and injectivity claim."
            ),
            checker_id=checker_id,
            tags=("polynomial", "map", "collision"),
            verification=verification,
        )
    adapter_type = adapter_types.get(operation_id)
    if adapter_type is None:
        return None
    checker_fields: dict[str, str | None] = {}
    if operation_id in {
        "polynomial.map.collision_witness",
        "polynomial.map.collision.search",
        "polynomial.map.collision.verify",
    }:
        checker_fields["collision_checker_id"] = _catalog_checker_id(
            catalog, checkers, "polynomial.map.collision.verify"
        )
    if operation_id == "polynomial.map.compute_jacobian":
        checker_fields["jacobian_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
    if operation_id == "polynomial.map.keller_condition.verify":
        checker_fields["keller_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
    if operation_id == "polynomial.identity.verify":
        checker_fields["identity_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
    if operation_id == "polynomial.rational_function.identity.verify":
        checker_fields["rational_function_identity_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
    if operation_id == "polynomial.map.inverse.refute_by_collision":
        checker_fields["inverse_collision_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
    if operation_id == "polynomial.map.inverse.verify":
        checker_fields["inverse_checker_id"] = _catalog_checker_id(
            catalog, checkers, operation_id
        )
        checker_fields["identity_checker_id"] = _catalog_checker_id(
            catalog, checkers, "polynomial.identity.verify"
        )
    contracts = replace(
        register_polynomial_resources(store, schemas),
        **cast(dict[str, Any], checker_fields),
    )
    resources = PolynomialResources(store, artifacts, verification, contracts)
    return adapter_type(resources)


def _catalog_checker_id(
    catalog: OperationCatalog,
    checkers: CheckerRegistry,
    operation_id: str,
) -> str:
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is missing; run `jacobian update`: {operation_id}"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    return binding.checker_id


def build_polynomial_operations(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[tuple[OperationAdapter[Any], ...], PolynomialContracts]:
    """Register exact polynomial-map schemas, adapters, and optional checker."""
    contracts = register_polynomial_resources(store, schemas)
    collision_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact rational polynomial-map collision checker",
            entrypoint="jacobian_checkers.polynomial_maps:check_collision",
            evidence_kind=EvidenceKind.WITNESS,
            format_id="polynomial.map_collision",
            format_version="1",
            claim_schema_uris=(contracts.claim_schema_uri,),
            semantics_uris=(contracts.semantics_uri,),
            candidate_schema_uris=(contracts.map_schema_uri,),
            reason="bundled polynomial-map reference checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    jacobian_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact sparse polynomial Jacobian replay checker",
            entrypoint="jacobian_checkers.polynomial_maps:check_jacobian",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.jacobian_replay",
            format_version="1",
            claim_schema_uris=(contracts.jacobian_claim_schema_uri,),
            semantics_uris=(contracts.semantics_uri,),
            candidate_schema_uris=(contracts.jacobian_schema_uri,),
            reason="bundled independent sparse-polynomial Jacobian checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    keller_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact polynomial-map Keller-condition checker",
            entrypoint="jacobian_checkers.polynomial_maps:check_keller_condition",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.map.keller_condition.replay",
            format_version="1",
            claim_schema_uris=(contracts.keller_claim_schema_uri,),
            semantics_uris=(contracts.semantics_uri,),
            candidate_schema_uris=(contracts.jacobian_schema_uri,),
            reason=(
                "bundled independent exact checker for a nonzero constant "
                "polynomial-map Jacobian determinant"
            ),
        ),
        authorize=authorize_checker,
    ).checker_id
    identity_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact sparse rational polynomial identity checker",
            entrypoint="jacobian_checkers.polynomial_maps:check_identity",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.identity_replay",
            format_version="1",
            claim_schema_uris=(contracts.identity_claim_schema_uri,),
            semantics_uris=(contracts.identity_semantics_uri,),
            candidate_schema_uris=(contracts.right_polynomial_schema_uri,),
            reason="bundled independent sparse-polynomial identity checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    rational_function_identity_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact sparse rational-function identity checker",
            entrypoint=(
                "jacobian_checkers.rational_functions:check_rational_function_identity"
            ),
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.rational_function.identity_replay",
            format_version="1",
            claim_schema_uris=(contracts.rational_function_identity_claim_schema_uri,),
            semantics_uris=(contracts.rational_function_identity_semantics_uri,),
            candidate_schema_uris=(contracts.rational_function_right_schema_uri,),
            reason="bundled independent sparse cross-multiplication checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    inverse_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact two-sided polynomial-map inverse checker",
            entrypoint="jacobian_checkers.polynomial_maps:check_map_inverse",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.map.inverse.two_sided_replay",
            format_version="1",
            claim_schema_uris=(contracts.inverse_claim_schema_uri,),
            semantics_uris=(contracts.inverse_semantics_uri,),
            candidate_schema_uris=(contracts.inverse_residual_schema_uri,),
            reason=("bundled independent two-sided sparse-polynomial map checker"),
        ),
        authorize=authorize_checker,
    ).checker_id
    inverse_collision_checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact polynomial-map inverse-obstruction checker",
            entrypoint=(
                "jacobian_checkers.polynomial_maps:check_collision_refutes_inverse"
            ),
            evidence_kind=EvidenceKind.WITNESS,
            format_id="polynomial.map_collision_refutes_inverse",
            format_version="1",
            claim_schema_uris=(contracts.inverse_collision_claim_schema_uri,),
            semantics_uris=(contracts.semantics_uri,),
            candidate_schema_uris=(contracts.map_schema_uri,),
            reason=(
                "bundled independent exact collision replay whose logical "
                "consequence is absence of a two-sided polynomial inverse"
            ),
        ),
        authorize=authorize_checker,
    ).checker_id
    contracts = replace(
        contracts,
        collision_checker_id=collision_checker_id,
        jacobian_checker_id=jacobian_checker_id,
        keller_checker_id=keller_checker_id,
        identity_checker_id=identity_checker_id,
        rational_function_identity_checker_id=rational_function_identity_checker_id,
        inverse_checker_id=inverse_checker_id,
        inverse_collision_checker_id=inverse_collision_checker_id,
    )
    resources = PolynomialResources(
        store=store,
        artifacts=artifacts,
        verification=verification,
        contracts=contracts,
    )
    collision_evidence_verify = witness_verification_adapter(
        operation_id="polynomial.map.collision_evidence.verify",
        title="Verify stored polynomial-map collision evidence",
        description=(
            "Independently replay one exact stored collision witness against its "
            "bound map and injectivity claim."
        ),
        checker_id=collision_checker_id,
        tags=("polynomial", "map", "collision"),
        verification=verification,
    )
    return (
        (
            PolynomialMapEvaluationAdapter(resources),
            PolynomialJacobianAdapter(resources),
            *(
                (PolynomialKellerConditionVerifyAdapter(resources),)
                if keller_checker_id is not None
                else ()
            ),
            PolynomialCollisionAdapter(resources),
            *((collision_evidence_verify,) if collision_evidence_verify else ()),
            PolynomialIdentityAdapter(resources),
            *(
                (RationalFunctionIdentityAdapter(resources),)
                if rational_function_identity_checker_id is not None
                else ()
            ),
            PolynomialCollisionSearchAdapter(resources),
            *(
                (PolynomialCollisionVerifyAdapter(resources),)
                if collision_checker_id is not None
                else ()
            ),
            *(
                (PolynomialMapInverseCollisionVerifyAdapter(resources),)
                if inverse_collision_checker_id is not None
                else ()
            ),
            PolynomialMapInverseSynthesizeAdapter(resources),
            *(
                (PolynomialMapInverseVerifyAdapter(resources),)
                if inverse_checker_id is not None and identity_checker_id is not None
                else ()
            ),
        ),
        contracts,
    )
