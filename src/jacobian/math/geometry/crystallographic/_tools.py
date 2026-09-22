"""Public declaration for crystallographic mapping-torus complexes."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.crystallographic._models import (
    CrystallographicMappingTorusRequest,
)
from jacobian.math.geometry.crystallographic.operations import (
    mapping_torus_chain_complex,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue


def _compute_mapping_torus(
    request: CrystallographicMappingTorusRequest,
) -> ChainComplexValue:
    return mapping_torus_chain_complex(
        request.linear_part,
        request.finite_order_exponent,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.mapping_torus.chain_complex.compute",
        title="Construct a flat mapping-torus cellular complex",
        description=(
            "For a bounded finite-order integral automorphism A, verify a supplied "
            "positive exponent m with A^m = I and return the exact integral based "
            "cellular chain complex of the compact flat mapping torus. The result "
            "is the mapping cone of Λ* A - I and composes directly with integral "
            "chain-complex homology. This covers cyclic-holonomy Bieberbach mapping "
            "tori; it does not classify crystallographic groups or search for a "
            "fundamental domain."
        ),
        request_type=CrystallographicMappingTorusRequest,
        result_type=ChainComplexValue,
        run=_compute_mapping_torus,
        tags=(
            "crystallographic-group",
            "Bieberbach-group",
            "flat-manifold",
            "mapping-torus",
            "chain-complex",
            "exact",
        ),
        discovery_terms=(
            "crystallographic mapping torus",
            "Bieberbach cyclic holonomy",
            "flat manifold cellular complex",
            "torus automorphism mapping cone",
        ),
        examples=(
            OperationExample(
                name="klein_bottle_cellular_complex",
                description=(
                    "Construct the exact integral cellular complex of the Klein "
                    "bottle as the mapping torus of x -> -x on the circle; the "
                    "supplied exponent must satisfy (-1)^2 = 1."
                ),
                input={
                    "linear_part": {
                        "domain": "ZZ",
                        "row_count": 1,
                        "column_count": 1,
                        "entries": [["-1"]],
                    },
                    "finite_order_exponent": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
