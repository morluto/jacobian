"""Public occupied affine-coset intersection profile."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.finite.cosets._models import (
    CosetIntersectionProfile,
    CosetIntersectionSource,
)
from jacobian.math.geometry.finite.cosets.operations import coset_intersection_profile


def _compute(request: CosetIntersectionSource) -> CosetIntersectionProfile:
    return coset_intersection_profile(request.space, request.subspace, request.subset)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="finite_geometry.subspace.coset_intersection_profile.compute",
        title="Partition a finite subset by subspace cosets",
        description=(
            "Return every occupied affine coset of a supplied prime-field RREF "
            "subspace, its canonical quotient representative and complete sorted "
            "intersection with the supplied finite subset. Retain the field, "
            "ordered axes, subspace and subset. Representatives have zero pivot "
            "coordinates; empty subsets give empty partitions."
        ),
        request_type=CosetIntersectionSource,
        result_type=CosetIntersectionProfile,
        run=_compute,
        tags=("finite-geometry", "cosets", "subspace", "quotient", "exact"),
        examples=(
            OperationExample(
                name="unequal_fibres",
                description="Three points of F_2^3 meet two cosets of a line.",
                input={
                    "space": {"field_order": 2, "axis": ["x", "y", "z"]},
                    "subspace": {
                        "space": {"field_order": 2, "axis": ["x", "y", "z"]},
                        "basis": [[1, 0, 0]],
                    },
                    "subset": [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                },
            ),
        ),
    ),
)
