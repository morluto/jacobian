"""Installation bundle for bounded lattice reduction."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.matrix_lattice.lattice import (
    LATTICE_OPERATIONS,
)
from jacobian.operations import DomainDiagnostics, DomainSemantics


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
            },
        ),
        operations=LATTICE_OPERATIONS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
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
