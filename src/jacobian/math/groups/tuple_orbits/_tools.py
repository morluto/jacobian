"""Tuple-family diagonal-action orbit operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.groups.tuple_orbits._models import (
    TupleFamilyOrbitRequest,
    TupleFamilyOrbitResult,
)
from jacobian.math.groups.tuple_orbits.operations import tuple_family_orbit_profile

TOOLS: MathTools = (
    MathTool(
        operation_id="group.action.tuple_family.diagonal_orbit_profile.compute",
        title="Compute diagonal-action tuple-family orbit profiles",
        description="Partition an indexed tuple family by a finite permutation group's diagonal action and return canonical representatives, source indices, orbit-stabilizer sizes, and least transporters.",
        request_type=TupleFamilyOrbitRequest,
        result_type=TupleFamilyOrbitResult,
        run=tuple_family_orbit_profile,
        tags=("group-action", "tuple", "orbit", "stabilizer", "exact"),
        examples=(
            OperationExample(
                name="pairs_under_swap",
                description="Profile repeated and distinct binary pairs under coordinate-value swapping.",
                input={
                    "group": {"degree": 2, "generators": [[1, 0]]},
                    "arity": 2,
                    "family": [[0, 0], [1, 1], [0, 1]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
