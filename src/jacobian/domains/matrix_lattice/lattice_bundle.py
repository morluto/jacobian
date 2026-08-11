"""Installation bundle for bounded lattice reduction."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.matrix_lattice.lattice import (
    LATTICE_CAPABILITIES,
    LATTICE_RUNTIME,
)
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import (
    PYTHON_FLINT_HNF_FLINT_VERSION,
    PYTHON_FLINT_VERSION,
)


def build_lattice_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="lattice",
        schema_namespace="jacobian.lattice",
        semantics=DomainSemantics(
            name="jacobian.exact-integer-lattice-reduction",
            version="1",
            definition={
                "representation": "integer row basis",
                "maximum_rows": 32,
                "maximum_columns": 32,
                "maximum_decimal_digits_per_entry": 256,
                "budget": "wall_seconds from 1 through 60",
                "algorithm": "FLINT LLL",
                "gram": "exact",
                "delta": "FLINT double 0.99",
                "eta": "FLINT double 0.51",
                "relation": "reduced_basis = transformation * source_basis",
                "timeout": "operational TIMEOUT with no retained result artifacts",
                "assurance": "computed; no independent checker",
            },
        ),
        provider_runtime=LATTICE_RUNTIME,
        backend_version=(
            f"python-flint {PYTHON_FLINT_VERSION} / FLINT {PYTHON_FLINT_HNF_FLINT_VERSION}"
        ),
        capabilities=LATTICE_CAPABILITIES,
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_LATTICE_REDUCTION_REQUEST",
                stage="lattice_input_validation",
                message="Input does not satisfy the bounded exact lattice contract.",
                hint=(
                    "Use a 1..32 by 1..32 canonical integer row basis, entries of "
                    "at most 256 digits, and wall_seconds from 1 through 60."
                ),
            )
        ),
    )
