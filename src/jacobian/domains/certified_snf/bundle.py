"""Explicit bundle for transformation-certified Smith normal forms."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.certified_snf.checkers import CERTIFIED_SNF_EXACT_REPLAY_CHECKERS
from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_CAPABILITIES
from jacobian.operations import DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_certified_snf_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
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
            "jacobian.sympy",
            features=(
                "exact-integer",
                "sympy-smith-normal-decomposition",
                "left-unimodular-transformation",
                "right-unimodular-transformation",
                "smith-divisibility-chain",
            ),
        ),
        backend_version=SYMPY_VERSION,
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
        checker_declarations=CERTIFIED_SNF_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_certified_snf_bundle"]
