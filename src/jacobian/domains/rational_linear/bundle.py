"""Installation bundle for domain-owned rational-linear operations."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)
from jacobian.domains._examples import example
from jacobian.domains.rational_linear.checkers import (
    RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.rational_linear.operations import (
    RUNTIME,
    compute_rational_inconsistency,
    compute_rational_solution,
)
from jacobian.operations import (
    ComputedOperation,
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import PYTHON_FLINT_VERSION, known_provider_runtime


def build_rational_linear_bundle() -> DomainBundle:
    producer_runtime = RUNTIME
    capabilities = (
        ComputedOperation(
            capability_id="linear.rational_solution.compute",
            title="Compute an exact rational solution",
            description="Return one total bounded rational solution candidate inline.",
            request_model=LinearRationalSolutionFindRequest,
            result_model=LinearRationalSolutionResult,
            implementation=compute_rational_solution,
            relation_id="linear.rational_solution.relation",
            tags=("linear-algebra", "rational", "solution", "exact"),
            provider_runtime=producer_runtime,
            invocation_examples=(
                example(
                    "identity_solution",
                    "Solve a one-variable identity system.",
                    {
                        "system": {
                            "variables": ["x"],
                            "coefficients": {"entries": [[{"num": "1", "den": "1"}]]},
                            "rhs": [{"num": "2", "den": "1"}],
                        }
                    },
                ),
            ),
        ),
        ComputedOperation(
            capability_id="linear.rational_inconsistency.compute",
            title="Compute an exact rational inconsistency witness",
            description="Return one normalized left witness inline when the system is inconsistent.",
            request_model=LinearRationalInconsistencyFindRequest,
            result_model=LinearRationalInconsistencyResult,
            implementation=compute_rational_inconsistency,
            relation_id="linear.rational_inconsistency.relation",
            tags=("linear-algebra", "rational", "inconsistency", "exact"),
            provider_runtime=producer_runtime,
        ),
    )
    return DomainBundle(
        domain_id="rational_linear",
        schema_namespace="jacobian.rational-linear",
        semantics=DomainSemantics(
            name="jacobian.exact-rational-linear",
            version="1",
            definition={
                "domain": "bounded rational linear systems",
                "producer": f"Python-FLINT {PYTHON_FLINT_VERSION}",
                "results": "ordinary solution and inconsistency candidates are inline",
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.rational-linear",
            features=("rational-linear-contracts", "inline-exact-candidates"),
        ),
        backend_version=f"python-flint-{PYTHON_FLINT_VERSION}",
        capabilities=capabilities,
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_RATIONAL_LINEAR_REQUEST",
                stage="rational_linear_input_validation",
                message="Input does not satisfy the exact rational-linear contract.",
                hint="Use canonical rational components and bounded dimensions.",
            )
        ),
        scope_description="the complete supplied rational linear system",
        completeness_basis="the pinned FLINT producer completed its bounded computation",
        assurance_basis="computed candidate only; independent verification is separate",
        checker_declarations=RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_rational_linear_bundle"]
