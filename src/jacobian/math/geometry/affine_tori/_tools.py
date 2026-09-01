"""Public operation declaration for affine-torus fixed loci."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.affine_tori._models import (
    AffineTorusFixedLocusRequest,
    AffineTorusFixedLocusResult,
)
from jacobian.math.geometry.affine_tori.operations import affine_torus_fixed_locus


def _compute_fixed_locus(
    request: AffineTorusFixedLocusRequest,
) -> AffineTorusFixedLocusResult:
    return affine_torus_fixed_locus(request.affine_map)


TOOLS: MathTools = (
    MathTool(
        operation_id="affine_torus.fixed_locus.compute",
        title="Compute an affine-torus fixed locus",
        description=(
            "Compute the exact fixed locus of x -> A*x+b on the standard real "
            "torus R^n/Z^n. Return either a primitive invariant-character "
            "obstruction or a base point, primitive connected subtorus, and "
            "finite component-group presentation."
        ),
        request_type=AffineTorusFixedLocusRequest,
        result_type=AffineTorusFixedLocusResult,
        run=_compute_fixed_locus,
        tags=(
            "affine-torus",
            "fixed-locus",
            "integer-lattice",
            "hermite-normal-form",
            "exact",
            "bounded",
        ),
        examples=(
            example(
                "two_components_on_a_circle",
                "Compute the two fixed points of multiplication by three on T^1; "
                "the linear part is square and the translation is represented on "
                "that same standard torus.",
                {
                    "affine_map": {
                        "torus": {"dimension": 1},
                        "linear_part": {
                            "row_count": 1,
                            "column_count": 1,
                            "entries": [["3"]],
                        },
                        "translation": {
                            "torus": {"dimension": 1},
                            "coordinates": [{"num": "0", "den": "1"}],
                        },
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
