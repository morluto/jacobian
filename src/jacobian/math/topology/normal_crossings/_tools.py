"""Normal crossings operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.topology.normal_crossings._models import (
    DualComplexResult,
    NearbyCycleLatticesResult,
    NormalCrossingsPresentationRequest,
)
from jacobian.math.topology.normal_crossings.operations import (
    dual_complex,
    nearby_cycle_lattices,
)

_SINGLE_BRANCH = {
    "components": ["D0"],
    "strata": [{"components": ["D0"], "dimension": 2}],
}

_NODE = {
    "components": ["D0", "D1"],
    "strata": [
        {"components": ["D0"], "dimension": 1},
        {"components": ["D1"], "dimension": 1},
        {"components": ["D0", "D1"], "dimension": 0},
    ],
}

_TRIPLE_POINT = {
    "components": ["D0", "D1", "D2"],
    "strata": [
        {"components": ["D0"], "dimension": 2},
        {"components": ["D1"], "dimension": 2},
        {"components": ["D2"], "dimension": 2},
        {"components": ["D0", "D1"], "dimension": 1},
        {"components": ["D0", "D2"], "dimension": 1},
        {"components": ["D1", "D2"], "dimension": 1},
        {"components": ["D0", "D1", "D2"], "dimension": 0},
    ],
}


def _dual_complex(
    request: NormalCrossingsPresentationRequest,
) -> DualComplexResult:
    return dual_complex(request)


def _nearby_cycle_lattices(
    request: NormalCrossingsPresentationRequest,
) -> NearbyCycleLatticesResult:
    return nearby_cycle_lattices(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.normal_crossings.dual_complex.compute",
        title="Compute the dual complex and Cech incidence complex of an SNC presentation",
        description="Validate a finite strict simple-normal-crossings incidence "
        "presentation (strata keyed by exact component sets, downward-closed "
        "incidence, transverse dimension identity, branch multiplicity at "
        "most eight), then return the dual complex as a canonical finite "
        "simplicial complex with one vertex per component and one simplex per "
        "inclusion-maximal stratum, per-stratum branch-multiplicity transport, "
        "and the normalization/Cech incidence complex over ZZ with strata of "
        "cardinality k+1 in degree k and the alternating signed-incidence "
        "differential d(S) = sum_j (-1)^j (S without i_j), whose square is "
        "replayed to vanish by the shared chain-complex kernel.",
        request_type=NormalCrossingsPresentationRequest,
        result_type=DualComplexResult,
        run=_dual_complex,
        tags=(
            "topology",
            "normal-crossings",
            "dual-complex",
            "chain-complex",
            "exact",
        ),
        discovery_terms=(
            "simple normal crossings dual complex",
            "semistable reduction dual complex",
            "normalization Cech incidence complex",
            "SNC stratum incidence presentation",
        ),
        examples=(
            OperationExample(
                name="triple_point_dual_complex",
                description="Compute the dual complex of the local model "
                "t = z0*z1*z2 (one stratum per nonempty component subset); "
                "the result is the full two-simplex on three components.",
                input=_TRIPLE_POINT,
            ),
            OperationExample(
                name="node_dual_complex",
                description="Compute the dual complex of the local model "
                "t = z0*z1 (two components meeting in one stratum); the "
                "result is a single interval with two vertices.",
                input=_NODE,
            ),
        ),
    ),
    MathTool(
        operation_id="topology.normal_crossings.nearby_cycle_lattices.compute",
        title="Compute integral nearby-cycle stalk lattices and specialization maps",
        description="Validate a finite strict simple-normal-crossings incidence "
        "presentation with the shared admission, then return the exact "
        "integral nearby-cycle stalk data of the local monomial models: for "
        "each stratum with r branches the Milnor-fibre phase lattice "
        "K = ker(sum: Z^r -> Z) with its Smith-certified saturated "
        "successive-difference basis, lattice rank r-1, and Milnor-fibre "
        "cohomology ranks H^q = exterior^q Hom(K, Z) of C(r-1, q) for q in "
        "0..r-1; plus, for each cover inclusion I of J in the component sets, "
        "the specialization K_J -> K_I as the fiber-sum fold that is the "
        "identity on I and folds the added branch onto the largest component "
        "of I in sorted order, expressed in the saturated bases.",
        request_type=NormalCrossingsPresentationRequest,
        result_type=NearbyCycleLatticesResult,
        run=_nearby_cycle_lattices,
        tags=(
            "topology",
            "normal-crossings",
            "nearby-cycles",
            "lattice",
            "exact",
        ),
        discovery_terms=(
            "nearby cycle stalk lattice",
            "Milnor fiber cohomology ranks",
            "nearby cycle specialization map",
            "monodromy phase lattice",
        ),
        examples=(
            OperationExample(
                name="triple_point_stalk_lattices",
                description="Compute the nearby-cycle stalks of t = z0*z1*z2; "
                "the triple point carries the rank-two sum-zero lattice with "
                "cohomology ranks (1, 2, 1).",
                input=_TRIPLE_POINT,
            ),
            OperationExample(
                name="smooth_branch_stalk_lattice",
                description="Compute the nearby-cycle stalk of the smooth local "
                "model t = z0; the single branch has the rank-zero lattice "
                "with cohomology ranks (1,).",
                input=_SINGLE_BRANCH,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
