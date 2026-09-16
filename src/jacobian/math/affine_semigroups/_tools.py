"""Public declaration for exact integer relation lattices."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.affine_semigroups._models import (
    RelationLatticeRequest,
    RelationLatticeResult,
)
from jacobian.math.affine_semigroups.operations import relation_lattice


def _run_relation_lattice(request: RelationLatticeRequest) -> RelationLatticeResult:
    return relation_lattice(request.configuration)


TOOLS = (
    MathTool(
        operation_id="integer_configuration.relation_lattice.compute",
        title="Compute the exact integer relation lattice of a configuration",
        description=(
            "Return the complete integer kernel lattice ker_Z(A) of one bounded "
            "integer configuration A in ZZ^(d x n): the canonical row-Hermite "
            "basis in ZZ^n, its unimodular transformation, rank and nullity, "
            "the configuration Smith invariant factors, and the saturation "
            "profile of the embedding. The returned basis satisfies "
            "A B^T = 0 exactly. Admission limits the configuration to 12 rows "
            "and columns and 8 digits per scalar; no solution search, Hilbert "
            "basis, or toric-ideal claim is made."
        ),
        request_type=RelationLatticeRequest,
        result_type=RelationLatticeResult,
        run=_run_relation_lattice,
        discovery_terms=(
            "integer kernel of a generator matrix",
            "integer relation lattice of a configuration",
            "canonical HNF basis of integer linear relations",
            "rank and nullity of an integer matrix over ZZ",
            "Smith normal form kernel generators",
        ),
        tags=(
            "lattice",
            "integer",
            "kernel",
            "smith-normal-form",
            "hermite-normal-form",
            "exact",
        ),
        examples=(
            OperationExample(
                name="one_by_three_configuration",
                description=(
                    "Compute ker_Z([1 2 3]), a rank-two sublattice of ZZ^3 whose "
                    "basis replays [1 2 3] B^T = 0."
                ),
                input={"configuration": {"entries": [["1", "2", "3"]]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
