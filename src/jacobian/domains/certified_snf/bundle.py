"""Explicit bundle for transformation-certified Smith normal forms."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_CAPABILITIES
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import known_provider_runtime

CERTIFIED_SNF_BUNDLE = DomainBundle(
    domain_id="certified_snf",
    schema_namespace="jacobian.certified-snf",
    semantics=DomainSemantics(
        name="jacobian.transformation-certified-smith-normal-form",
        version="1",
        definition={
            "domain": "ZZ",
            "maximum_rows": 16,
            "maximum_columns": 16,
            "maximum_input_digits": 32,
            "relation": "D = U A V",
            "left_transformation": "square unimodular row-basis change",
            "right_transformation": "square unimodular column-basis change",
            "normalization": "positive nonzero diagonal with divisibility chain",
            "excluded": (
                "Hermite normal form, rational canonical form, modular normal "
                "forms, and homology conclusions"
            ),
        },
    ),
    provider_runtime=known_provider_runtime(
        "jacobian.certified-snf",
        features=(
            "exact-integer",
            "elementary-row-operations",
            "elementary-column-operations",
            "left-unimodular-transformation",
            "right-unimodular-transformation",
            "smith-divisibility-chain",
        ),
    ),
    backend_version="jacobian.certified-snf/1",
    capabilities=CERTIFIED_SNF_CAPABILITIES,
    diagnostics=DomainDiagnostics(
        invalid_request=CapabilityDiagnostic(
            code="INVALID_CERTIFIED_SMITH_REQUEST",
            stage="certified_smith_input_validation",
            message="Input does not satisfy the bounded certified-Smith contract.",
            hint=(
                "Supply a nonempty matrix of at most 16 by 16 canonical integer "
                "strings, each containing at most 32 decimal digits."
            ),
        )
    ),
    scope_description="the complete supplied bounded integer matrix",
    completeness_basis=(
        "elementary row and column operations produced both full basis changes "
        "and the complete canonical Smith diagonal"
    ),
    assurance_basis=(
        "exact integer computation capped at COMPUTED; independent certificate "
        "replay is available through matrix.normal_form.smith.certified.verify"
    ),
)

__all__ = ["CERTIFIED_SNF_BUNDLE"]
