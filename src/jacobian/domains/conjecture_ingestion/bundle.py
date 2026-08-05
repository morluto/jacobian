"""Explicit bundle for privacy-sensitive external conjecture ingestion."""

from collections.abc import Mapping

from jacobian.conjecture_ingestion import install_conjecture_ingestion_capability
from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import jacobian_provider_runtime

_CAPABILITY_ID = "dataset.conjecture.ingest"


def _install(
    context: InstallationContext,
    dependencies: Mapping[str, InstalledDomainBundle],
) -> InstalledDomainBundle:
    if dependencies:
        raise ValueError("conjecture ingestion has no bundle dependencies")
    adapter, installation = install_conjecture_ingestion_capability(
        context.store,
        context.schemas,
        context.artifacts,
    )
    return InstalledDomainBundle(
        adapters=(adapter,),
        semantics_uri=installation.semantics_uri,
        input_schema_uris={},
        result_schema_uris={_CAPABILITY_ID: installation.artifact_schema_uri},
        obligation_schema_uris={},
    )


def build_conjecture_ingestion_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="conjecture_ingestion",
        schema_namespace="jacobian.external-conjecture",
        semantics=DomainSemantics(
            name="jacobian.external-conjecture-ingestion",
            version="1",
            definition={
                "description": (
                    "license-aware external conjecture indexing with text withholding"
                ),
                "policy_id": "jacobian.external-conjecture-publication/v1",
                "verification": (
                    "none; ingestion is heuristic provenance and never verifies a claim"
                ),
            },
        ),
        provider_runtime=jacobian_provider_runtime(
            "jacobian.conjecture-ingestion",
            features=("license-policy", "metadata-withholding", "provenance"),
        ),
        backend_version="jacobian",
        capabilities=(),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_EXTERNAL_CONJECTURE_RECORD",
                stage="request_validation",
                message="The external conjecture record is invalid.",
                hint="Provide pinned corpus, item, license, and provenance fields.",
            )
        ),
        scope_description="one external conjecture record under one policy version",
        completeness_basis="the registered license policy was applied to the complete record",
        assurance_basis=(
            "ingestion preserves a sourced conjecture for research only; "
            "it does not establish truth, formal correspondence, or proof"
        ),
        managed_capability_ids=(_CAPABILITY_ID,),
        managed_installer=_install,
    )


__all__ = ["build_conjecture_ingestion_bundle"]
