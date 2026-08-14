"""Rational optimization domain bundle."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.optimization.checkers import (
    RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.optimization.operations import RATIONAL_LINEAR_OPERATIONS
from jacobian.operations import DomainDiagnostics, DomainSemantics


def build_rational_optimization_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="optimization",
        schema_namespace="jacobian.validated-analysis",
        semantics=DomainSemantics(
            name="jacobian.optimization",
            version="1",
            definition={
                "description": "exact bounded rational optimization",
                "scope": "minimize c^T x subject to A x = b and x >= 0",
                "budget": "wall_seconds is enforced in an isolated SymPy worker",
                "failure": (
                    "timeouts, backend failures, and missing candidates are non-conclusions"
                ),
            },
        ),
        operations=RATIONAL_LINEAR_OPERATIONS,
        checker_declarations=RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_RATIONAL_OPTIMIZATION_REQUEST",
                stage="rational_optimization_input_validation",
                message="Input does not satisfy the rational optimization contract.",
                hint="Use the declared bounded standard-form rational LP model.",
            )
        ),
    )


__all__ = ["build_rational_optimization_bundle"]
