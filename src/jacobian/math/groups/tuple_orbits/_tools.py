"""Tuple-family diagonal-action orbit operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.groups.tuple_orbits._models import (
    TupleFamilyOrbitResult,
    TupleFamilyOrbitSource,
)
from jacobian.math.groups.tuple_orbits.operations import tuple_family_orbit_profile

TOOLS: MathTools = (
    MathTool(
        operation_id="group_action.tuple_family.orbit_profile.compute",
        title="Compute diagonal-action tuple-family orbit profiles",
        description=(
            "Partition an indexed family of fixed-arity ordered tuples by the "
            "diagonal action of one finite permutation action. Return the "
            "lexicographically least ambient representative, all source "
            "indices, full orbit and stabilizer sizes, a least transporter, "
            "and whether the supplied family is a union of complete ambient "
            "orbits; tuple order and repeated coordinates are preserved and "
            "the action domain is retained in the source value."
        ),
        request_type=TupleFamilyOrbitSource,
        result_type=TupleFamilyOrbitResult,
        run=tuple_family_orbit_profile,
        tags=("group-action", "tuple", "orbit", "stabilizer", "exact"),
        examples=(
            OperationExample(
                name="pairs_under_swap",
                description=(
                    "Profile repeated and distinct binary pairs under the "
                    "diagonal swap; tuple rows must have one common arity and "
                    "coordinates must use the declared action domain."
                ),
                input={
                    "action": {
                        "domain": ["a", "b"],
                        "generators": [[1, 0]],
                    },
                    "arity": 2,
                    "family": [[0, 0], [1, 1], [0, 1]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
