"""Validated real-analysis domain bundle."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.analysis.operations import POINT_ENCLOSURE_OPERATIONS
from jacobian.operations import DomainDiagnostics, DomainSemantics


def build_real_analysis_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="analysis",
        schema_namespace="jacobian.validated-analysis",
        semantics=DomainSemantics(
            name="jacobian.analysis",
            version="1",
            definition={
                "description": "rigorous real-function enclosures",
                "scope": (
                    "one EXP, LOG, SQRT, SIN, or COS evaluation at an exact "
                    "rational point and declared precision"
                ),
                "budget": "wall_seconds is enforced in an isolated Arb worker",
                "failure": "non-finite balls and worker failures are non-conclusions",
            },
        ),
        operations=POINT_ENCLOSURE_OPERATIONS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_REAL_ANALYSIS_REQUEST",
                stage="real_analysis_input_validation",
                message="Input does not satisfy the bounded real-analysis contract.",
                hint="Use a supported function, bounded rational, and declared precision.",
            )
        ),
    )


__all__ = ["build_real_analysis_bundle"]
