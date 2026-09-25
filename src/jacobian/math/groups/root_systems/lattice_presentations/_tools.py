"""Owner-local declarations for root/weight lattice presentations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.root_systems._models import FiniteCartanDatum
from jacobian.math.groups.root_systems.lattice_presentations._models import (
    RootWeightLatticePresentation,
)
from jacobian.math.groups.root_systems.lattice_presentations.operations import (
    root_weight_lattice_presentation,
)

_A2_DATUM = {
    "cartan_matrix": {
        "matrix": {
            "domain": "ZZ",
            "row_count": 2,
            "column_count": 2,
            "entries": [[2, -1], [-1, 2]],
        },
        "simple_root_axis": [0, 1],
    },
    "symmetrizer": [{"num": 1, "den": 1}, {"num": 1, "den": 1}],
    "root_to_weight": {
        "domain": "ZZ",
        "row_count": 2,
        "column_count": 2,
        "entries": [[2, -1], [-1, 2]],
    },
    "coroot_to_coweight": {
        "domain": "ZZ",
        "row_count": 2,
        "column_count": 2,
        "entries": [[2, -1], [-1, 2]],
    },
}

TOOLS = (
    MathTool(
        operation_id="root_system.root_weight_lattice_embedding.compute",
        title="Present the root lattice inside the weight lattice",
        description=(
            "Materialize the simple-root lattice Q as a source-bound integer "
            "sublattice of the fundamental-weight lattice P. The result retains "
            "both `IntegerLattice` values and the exact root-to-weight inclusion, "
            "so it composes directly with the generic lattice quotient operation. "
            "Finite Cartan rank is at most 8."
        ),
        request_type=FiniteCartanDatum,
        result_type=RootWeightLatticePresentation,
        run=root_weight_lattice_presentation,
        tags=("algebra", "root-system", "lattice", "exact"),
        discovery_terms=("root lattice in weight lattice", "root weight inclusion"),
        examples=(
            OperationExample(
                name="a2_root_lattice_in_weight_lattice",
                description=(
                    "Present Q(A2) as a sublattice of P(A2), with Cartan inclusion "
                    "of index 3."
                ),
                input=_A2_DATUM,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
