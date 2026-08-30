"""Exact rational coordinate-tensor operations."""

from __future__ import annotations

from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.differential._bounds import (
    build_lie_derivative_plan,
)
from jacobian.math.geometry.differential._models import RationalLieDerivativeProfile
from jacobian.math.geometry.differential._sympy import (
    compute_lie_derivative_components,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)


def lie_derivative(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
) -> RationalLieDerivativeProfile:
    """Return the exact coordinate Lie derivative ``L_X T``.

    The result remains authoritative only on the explicit intersection of the
    retained source loci, even when rational-function normalization cancels a
    denominator from every result component.
    """

    plan = build_lie_derivative_plan(vector_field, tensor)
    components = compute_lie_derivative_components(vector_field, tensor, plan)
    locus_guards = canonical_locus_guards(
        plan.inherited_locus_guards,
        component_denominators=tuple(component.denominator for component in components),
        variable_count=len(tensor.coordinate_axis),
    )
    result_tensor = RationalCoordinateTensor(
        coordinate_axis=tensor.coordinate_axis,
        variance=tensor.variance,
        components=components,
        retained_nonzero_denominators=locus_guards,
    )
    result = RationalLieDerivativeProfile._from_kernel(
        vector_field=vector_field,
        source=tensor,
        lie_derivative=result_tensor,
    )
    actual_result_bytes = len(encode_strict_json(result.model_dump(mode="json")))
    if actual_result_bytes > plan.result_bytes_upper_bound:
        raise AssertionError(
            "Lie-derivative serialized result exceeded its admitted upper bound"
        )
    return result


__all__ = ["lie_derivative"]
