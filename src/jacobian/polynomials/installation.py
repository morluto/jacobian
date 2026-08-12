"""Register polynomial-map schemas, adapters, and optional checkers."""

from __future__ import annotations

from jacobian.artifacts import ArtifactService
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, WitnessEnvelope
from jacobian.contracts.polynomials import (
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
from jacobian.polynomials.resources import PolynomialInstallation, PolynomialResources
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_capabilities import witness_verification_adapter


def install_polynomial_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[tuple[CapabilityAdapter, ...], PolynomialInstallation]:
    """Register exact polynomial-map schemas, adapters, and optional checker."""

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
            "maximum_dimension": 4,
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
    map_schema_uri = schemas.register(
        name="jacobian.rational-polynomial-map",
        version="1",
        schema=model_schema(RationalPolynomialMap),
    )
    evaluation_schema_uri = schemas.register(
        name="jacobian.polynomial-map-evaluation",
        version="1",
        schema=model_schema(PolynomialMapEvaluation),
    )
    jacobian_schema_uri = schemas.register(
        name="jacobian.polynomial-jacobian",
        version="1",
        schema=model_schema(PolynomialJacobian),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-injectivity-claim",
        version="1",
        schema=model_schema(PolynomialInjectivityClaim),
    )
    jacobian_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-jacobian-claim",
        version="1",
        schema=model_schema(PolynomialJacobianClaim),
    )
    right_polynomial_schema_uri = schemas.register(
        name="jacobian.sparse-rational-polynomial-right",
        version="1",
        schema=model_schema(RationalPolynomial),
    )
    left_polynomial_schema_uri = schemas.register(
        name="jacobian.sparse-rational-polynomial-left",
        version="1",
        schema=model_schema(RationalPolynomial),
    )
    identity_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-identity-claim",
        version="1",
        schema=model_schema(PolynomialIdentityClaim),
    )
    rational_function_left_schema_uri = schemas.register(
        name="jacobian.sparse-rational-function-left",
        version="1",
        schema=model_schema(RationalFunctionArtifact),
    )
    rational_function_right_schema_uri = schemas.register(
        name="jacobian.sparse-rational-function-right",
        version="1",
        schema=model_schema(RationalFunctionArtifact),
    )
    rational_function_identity_claim_schema_uri = schemas.register(
        name="jacobian.rational-function-identity-claim",
        version="1",
        schema=model_schema(RationalFunctionIdentityClaim),
    )
    keller_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-keller-condition-claim",
        version="1",
        schema=model_schema(PolynomialKellerConditionClaim),
    )
    inverse_collision_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-no-two-sided-inverse-claim",
        version="1",
        schema=model_schema(PolynomialNoTwoSidedInverseClaim),
    )
    inverse_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-inverse-claim",
        version="1",
        schema=model_schema(PolynomialMapInverseClaim),
    )
    inverse_residual_schema_uri = schemas.register(
        name="jacobian.polynomial-map-composition-residuals",
        version="1",
        schema=model_schema(PolynomialMapCompositionResiduals),
    )
    inverse_synthesis_schema_uri = schemas.register(
        name="jacobian.polynomial-map-inverse-synthesis",
        version="1",
        schema=model_schema(PolynomialMapInverseSynthesisArtifact),
    )
    witness_schema_uri = schemas.register(
        name="jacobian.witness-envelope",
        version="1",
        schema=model_schema(WitnessEnvelope),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    collision_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact rational polynomial-map collision checker",
                entrypoint="jacobian_checkers.polynomial_maps:check_collision",
                evidence_kind=EvidenceKind.WITNESS,
                format_id="polynomial.map_collision",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(map_schema_uri,),
                reason="bundled polynomial-map reference checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    jacobian_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact sparse polynomial Jacobian replay checker",
                entrypoint="jacobian_checkers.polynomial_maps:check_jacobian",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.jacobian_replay",
                format_version="1",
                claim_schema_uris=(jacobian_claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(jacobian_schema_uri,),
                reason="bundled independent sparse-polynomial Jacobian checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    keller_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact polynomial-map Keller-condition checker",
                entrypoint="jacobian_checkers.polynomial_maps:check_keller_condition",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.map.keller_condition.replay",
                format_version="1",
                claim_schema_uris=(keller_claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(jacobian_schema_uri,),
                reason=(
                    "bundled independent exact checker for a nonzero constant "
                    "polynomial-map Jacobian determinant"
                ),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    identity_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact sparse rational polynomial identity checker",
                entrypoint="jacobian_checkers.polynomial_maps:check_identity",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.identity_replay",
                format_version="1",
                claim_schema_uris=(identity_claim_schema_uri,),
                semantics_uris=(identity_semantics_uri,),
                candidate_schema_uris=(right_polynomial_schema_uri,),
                reason="bundled independent sparse-polynomial identity checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    rational_function_identity_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact sparse rational-function identity checker",
                entrypoint=(
                    "jacobian_checkers.rational_functions:"
                    "check_rational_function_identity"
                ),
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.rational_function.identity_replay",
                format_version="1",
                claim_schema_uris=(rational_function_identity_claim_schema_uri,),
                semantics_uris=(rational_function_identity_semantics_uri,),
                candidate_schema_uris=(rational_function_right_schema_uri,),
                reason="bundled independent sparse cross-multiplication checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    inverse_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact two-sided polynomial-map inverse checker",
                entrypoint="jacobian_checkers.polynomial_maps:check_map_inverse",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.map.inverse.two_sided_replay",
                format_version="1",
                claim_schema_uris=(inverse_claim_schema_uri,),
                semantics_uris=(inverse_semantics_uri,),
                candidate_schema_uris=(inverse_residual_schema_uri,),
                reason=("bundled independent two-sided sparse-polynomial map checker"),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    inverse_collision_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact polynomial-map inverse-obstruction checker",
                entrypoint=(
                    "jacobian_checkers.polynomial_maps:check_collision_refutes_inverse"
                ),
                evidence_kind=EvidenceKind.WITNESS,
                format_id="polynomial.map_collision_refutes_inverse",
                format_version="1",
                claim_schema_uris=(inverse_collision_claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(map_schema_uri,),
                reason=(
                    "bundled independent exact collision replay whose logical "
                    "consequence is absence of a two-sided polynomial inverse"
                ),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = PolynomialInstallation(
        semantics_uri=semantics_uri,
        identity_semantics_uri=identity_semantics_uri,
        rational_function_identity_semantics_uri=(
            rational_function_identity_semantics_uri
        ),
        inverse_semantics_uri=inverse_semantics_uri,
        map_schema_uri=map_schema_uri,
        evaluation_schema_uri=evaluation_schema_uri,
        jacobian_schema_uri=jacobian_schema_uri,
        claim_schema_uri=claim_schema_uri,
        jacobian_claim_schema_uri=jacobian_claim_schema_uri,
        right_polynomial_schema_uri=right_polynomial_schema_uri,
        left_polynomial_schema_uri=left_polynomial_schema_uri,
        identity_claim_schema_uri=identity_claim_schema_uri,
        rational_function_left_schema_uri=rational_function_left_schema_uri,
        rational_function_right_schema_uri=rational_function_right_schema_uri,
        rational_function_identity_claim_schema_uri=(
            rational_function_identity_claim_schema_uri
        ),
        keller_claim_schema_uri=keller_claim_schema_uri,
        inverse_collision_claim_schema_uri=inverse_collision_claim_schema_uri,
        inverse_claim_schema_uri=inverse_claim_schema_uri,
        inverse_residual_schema_uri=inverse_residual_schema_uri,
        inverse_synthesis_schema_uri=inverse_synthesis_schema_uri,
        witness_schema_uri=witness_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
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
        installation=installation,
    )
    collision_evidence_verify = witness_verification_adapter(
        capability_id="polynomial.map.collision_evidence.verify",
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
        installation,
    )
