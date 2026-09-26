"""Cellular sheaf operation declarations."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.cellular_sheaves._models import (
    FromCoverMapsRequest,
    FromCoverMapsResult,
    SheafCohomologyRequest,
    SheafCohomologyResult,
    SheafSubcomplexRequest,
    SheafSubcomplexResult,
)
from jacobian.math.topology.cellular_sheaves.cohomology_map_tools import (
    TOOLS as COHOMOLOGY_MAP_TOOLS,
)
from jacobian.math.topology.cellular_sheaves.extensions_tools import (
    TOOLS as EXTENSION_TOOLS,
)
from jacobian.math.topology.cellular_sheaves.operations import (
    from_cover_maps,
    restrict_to_subcomplex,
    sheaf_cohomology,
)
from jacobian.math.topology.cellular_sheaves.morphism_kernel_tools import (
    TOOLS as MORPHISM_KERNEL_TOOLS,
)

__all__ = ["TOOLS"]


def _run_from_cover_maps(
    request: FromCoverMapsRequest,
) -> FromCoverMapsResult:
    return from_cover_maps(
        request.complex,
        request.coefficient_field,
        request.prime,
        request.stalks,
        request.cover_maps,
    )


def _run_sheaf_cohomology(
    request: SheafCohomologyRequest,
) -> SheafCohomologyResult:
    return sheaf_cohomology(request.sheaf)


def _run_restrict_to_subcomplex(
    request: SheafSubcomplexRequest,
) -> SheafSubcomplexResult:
    return restrict_to_subcomplex(request.sheaf, request.subcomplex)


_INTERVAL_CANONICAL = {
    "vertices": ["a", "b"],
    "maximal_simplices": [["a", "b"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"]]},
        {"dimension": 1, "faces": [["a", "b"]]},
    ],
    "dimension": 1,
    "f_vector": [2, 1],
    "closure_size": 3,
}

_TRIANGLE_CANONICAL = {
    "vertices": ["a", "b", "c"],
    "maximal_simplices": [["a", "b", "c"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
        {"dimension": 1, "faces": [["a", "b"], ["a", "c"], ["b", "c"]]},
        {"dimension": 2, "faces": [["a", "b", "c"]]},
    ],
    "dimension": 2,
    "f_vector": [3, 3, 1],
    "closure_size": 7,
}


def _rank_one_stalks() -> list[dict[str, object]]:
    return [
        {"simplex": simplex, "basis": ["x"]}
        for simplex in (
            ["a"],
            ["b"],
            ["c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
            ["a", "b", "c"],
        )
    ]


def _identity_cover_maps() -> list[dict[str, object]]:
    pairs = (
        (["a"], ["a", "b"]),
        (["b"], ["a", "b"]),
        (["a"], ["a", "c"]),
        (["c"], ["a", "c"]),
        (["b"], ["b", "c"]),
        (["c"], ["b", "c"]),
        (["a", "b"], ["a", "b", "c"]),
        (["a", "c"], ["a", "b", "c"]),
        (["b", "c"], ["a", "b", "c"]),
    )
    return [
        {"source": source, "target": target, "entries": [[{"num": "1", "den": "1"}]]}
        for source, target in pairs
    ]


TOOLS: MathTools = (
    *MORPHISM_KERNEL_TOOLS,
    *COHOMOLOGY_MAP_TOOLS,
    *EXTENSION_TOOLS,
    MathTool(
        operation_id="cellular_sheaf.subcomplex.restrict",
        title="Restrict a cellular sheaf to an included subcomplex",
        description=(
            "Filter a checked sheaf diagram to the cells of an included "
            "simplicial subcomplex, retaining its exact stalk bases, coefficient "
            "field, and all comparable-cell restriction maps."
        ),
        request_type=SheafSubcomplexRequest,
        result_type=SheafSubcomplexResult,
        run=_run_restrict_to_subcomplex,
        tags=("topology", "cellular-sheaf", "restriction", "subcomplex", "exact"),
        examples=(
            OperationExample(
                name="interval_vertex_subcomplex",
                description="Restrict a constant sheaf on an interval to its vertex a.",
                input={
                    "sheaf": {
                        "complex": _INTERVAL_CANONICAL,
                        "coefficient_field": "QQ",
                        "stalks": [
                            {"simplex": ["a"], "basis": ["x"]},
                            {"simplex": ["b"], "basis": ["x"]},
                            {"simplex": ["a", "b"], "basis": ["x"]},
                        ],
                        "cover_restrictions": [
                            {
                                "source": ["a"],
                                "target": ["a", "b"],
                                "row_basis": ["x"],
                                "column_basis": ["x"],
                                "entries": [[{"num": "1", "den": "1"}]],
                                "cover_path": [["a"], ["a", "b"]],
                            },
                            {
                                "source": ["b"],
                                "target": ["a", "b"],
                                "row_basis": ["x"],
                                "column_basis": ["x"],
                                "entries": [[{"num": "1", "den": "1"}]],
                                "cover_path": [["b"], ["a", "b"]],
                            },
                        ],
                        "derived_restrictions": [],
                        "diamonds": 0,
                        "comparable_pairs": 2,
                    },
                    "subcomplex": {
                        "vertices": ["a"],
                        "maximal_simplices": [["a"]],
                        "faces_by_dimension": [{"dimension": 0, "faces": [["a"]]}],
                        "dimension": 0,
                        "f_vector": [1],
                        "closure_size": 1,
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.cohomology.compute",
        title="Compute cellular sheaf cohomology with representative cocycles",
        description=(
            "Compute the cellular cohomology of a checked finite cellular "
            "sheaf over QQ or a bounded prime field: assemble the "
            "signed-incidence cochain complex from the complete restriction "
            "diagram, replay delta^2 = 0 and the Euler-characteristic "
            "identity inside the kernel, and return per-degree Betti "
            "numbers with representative cocycles in cochain coordinates."
        ),
        request_type=SheafCohomologyRequest,
        result_type=SheafCohomologyResult,
        run=_run_sheaf_cohomology,
        tags=(
            "topology",
            "cellular-sheaf",
            "sheaf-cohomology",
            "betti-number",
            "exact",
        ),
        discovery_terms=(
            "cellular sheaf cohomology",
            "sheaf Betti numbers",
            "sheaf cocycles",
            "cellular cochain complex",
        ),
        examples=(
            OperationExample(
                name="interval_constant_sheaf_cohomology",
                description=(
                    "Cohomology of the constant rank-one QQ sheaf on an "
                    "interval: H^0 is one-dimensional and H^1 vanishes."
                ),
                input={
                    "sheaf": {
                        "complex": _INTERVAL_CANONICAL,
                        "coefficient_field": "QQ",
                        "prime": None,
                        "stalks": [
                            {"simplex": ["a"], "basis": ["x"]},
                            {"simplex": ["b"], "basis": ["x"]},
                            {"simplex": ["a", "b"], "basis": ["x"]},
                        ],
                        "cover_restrictions": [
                            {
                                "source": ["a"],
                                "target": ["a", "b"],
                                "row_basis": ["x"],
                                "column_basis": ["x"],
                                "entries": [[{"num": "1", "den": "1"}]],
                                "cover_path": [["a"], ["a", "b"]],
                            },
                            {
                                "source": ["b"],
                                "target": ["a", "b"],
                                "row_basis": ["x"],
                                "column_basis": ["x"],
                                "entries": [[{"num": "1", "den": "1"}]],
                                "cover_path": [["b"], ["a", "b"]],
                            },
                        ],
                        "derived_restrictions": [],
                        "diamonds": 0,
                        "comparable_pairs": 2,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.from_cover_maps.compute",
        title="Construct a cellular sheaf from its cover restriction maps",
        description=(
            "Check one candidate cellular sheaf diagram on a bounded finite "
            "simplicial complex over QQ or a bounded prime field: one based "
            "stalk per nonempty simplex and one exact matrix per "
            "codimension-one face inclusion directed from face to coface. An "
            "accepted diagram returns the canonical sheaf with every derived "
            "comparable-face restriction composed along a canonical saturated "
            "chain after every length-two diamond was verified to commute; "
            "otherwise the result carries the first missing map, wrong-axis "
            "map, or one concrete noncommuting diamond with both paths and "
            "both composites."
        ),
        request_type=FromCoverMapsRequest,
        result_type=FromCoverMapsResult,
        run=_run_from_cover_maps,
        tags=(
            "topology",
            "cellular-sheaf",
            "stalk",
            "restriction-map",
            "commutative-diagram",
            "exact",
        ),
        discovery_terms=(
            "cellular sheaf",
            "cellular sheaf restrictions",
            "stalk diagram",
            "restriction maps",
            "sheaf from cover maps",
            "noncommuting diamond",
        ),
        examples=(
            OperationExample(
                name="interval_constant_rank_one_sheaf",
                description=(
                    "Build the constant rank-one QQ sheaf on an interval with "
                    "identity restrictions; every nonempty simplex needs exactly "
                    "one stalk and every cover inclusion exactly one matrix."
                ),
                input={
                    "complex": _INTERVAL_CANONICAL,
                    "coefficient_field": "QQ",
                    "stalks": [
                        {"simplex": ["a"], "basis": ["x"]},
                        {"simplex": ["b"], "basis": ["x"]},
                        {"simplex": ["a", "b"], "basis": ["x"]},
                    ],
                    "cover_maps": [
                        {
                            "source": ["a"],
                            "target": ["a", "b"],
                            "entries": [[{"num": "1", "den": "1"}]],
                        },
                        {
                            "source": ["b"],
                            "target": ["a", "b"],
                            "entries": [[{"num": "1", "den": "1"}]],
                        },
                    ],
                },
            ),
            OperationExample(
                name="triangle_noncommuting_diamond",
                description=(
                    "Reject a filled-triangle rank-one diagram whose corrupted "
                    "edge-to-face map breaks one diamond; the result names both "
                    "saturated paths and both composite matrices."
                ),
                input={
                    "complex": _TRIANGLE_CANONICAL,
                    "coefficient_field": "QQ",
                    "stalks": _rank_one_stalks(),
                    "cover_maps": [
                        *(_identity_cover_maps()[:6]),
                        {
                            "source": ["a", "b"],
                            "target": ["a", "b", "c"],
                            "entries": [[{"num": "2", "den": "1"}]],
                        },
                        *(_identity_cover_maps()[7:]),
                    ],
                },
            ),
        ),
    ),
)
