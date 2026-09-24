"""Public declaration for finite simplicial-set degeneracy profiles."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.degeneracy import (
    DegeneracyProfileResult,
    degeneracy_profile,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_DELTA_ONE = standard_simplex(1, 2).model_dump(mode="json")

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.degeneracy_profile.compute",
        title="Classify degeneracies in a finite simplicial set",
        description=(
            "For each simplex in a checked finite prefix, return its source-axis "
            "index as nondegenerate or a deterministic first immediate witness "
            "s_i(y)=x. Witnesses are one-step presentations, not the full "
            "epi-mono normal form. Tables are checked before profiling; work and "
            "output are bounded by the finite carrier limits."
        ),
        request_type=FiniteTruncatedSimplicialSet,
        result_type=DegeneracyProfileResult,
        run=degeneracy_profile,
        tags=("topology", "simplicial-set", "degeneracy", "exact"),
        discovery_terms=(
            "nondegenerate simplices",
            "degenerate simplex",
            "degeneracy profile",
        ),
        examples=(
            OperationExample(
                name="delta_one_degree_two_profile",
                description=(
                    "Classify the degree-0..2 prefix of Delta[1]; its unique "
                    "nondegenerate positive-degree simplex is the edge in "
                    "degree one, and every degree-two simplex is degenerate."
                ),
                input=_DELTA_ONE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
