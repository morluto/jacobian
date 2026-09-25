"""Public operation for checked parallelepiped translation tori."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicFundamentalDomainResult,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori._models import (
    BieberbachTranslationTorusChains,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori.operations import (
    translation_torus_quotient_chains,
)

_UNIT_CUBE_EXAMPLE = {
    "source": {
        "affine_realization": {
            "source": {
                "multiplication_table": [[0]],
                "action_matrices": [
                    [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]
                ],
                "factor_set": [[["0", "0", "0"]]],
            },
            "section_maps": [
                {
                    "holonomy_element": 0,
                    "linear_part": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                    "section_shift": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                }
            ],
        },
        "polytope": {
            "space": {"axes": ["x", "y", "z"]},
            "vertices": [
                {
                    "vertex_id": "v0",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v1",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v2",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v3",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v4",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v5",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v6",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "v7",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ],
        },
        "lattice_axes": ["x", "y", "z"],
        "facet_profile": {
            "vertices": [
                {
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ]
                },
                {
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ]
                },
            ],
            "dimension": 3,
            "facets": [
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "-1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        "offset": {"num": "0", "den": "1"},
                    },
                    "source_vertex_indices": [0, 1, 2, 3],
                },
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "-1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        "offset": {"num": "0", "den": "1"},
                    },
                    "source_vertex_indices": [0, 1, 4, 5],
                },
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                        "offset": {"num": "0", "den": "1"},
                    },
                    "source_vertex_indices": [0, 2, 4, 6],
                },
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "offset": {"num": "1", "den": "1"},
                    },
                    "source_vertex_indices": [1, 3, 5, 7],
                },
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        "offset": {"num": "1", "den": "1"},
                    },
                    "source_vertex_indices": [2, 3, 6, 7],
                },
                {
                    "halfspace": {
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        "offset": {"num": "1", "den": "1"},
                    },
                    "source_vertex_indices": [4, 5, 6, 7],
                },
            ],
        },
        "pairings": [
            {
                "source_facet_index": 0,
                "target_facet_index": 5,
                "lattice_translation": ["1", "0", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 1,
                "target_facet_index": 4,
                "lattice_translation": ["0", "1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 2,
                "target_facet_index": 3,
                "lattice_translation": ["0", "0", "1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 3,
                "target_facet_index": 2,
                "lattice_translation": ["0", "0", "-1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 4,
                "target_facet_index": 1,
                "lattice_translation": ["0", "-1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 5,
                "target_facet_index": 0,
                "lattice_translation": ["-1", "0", "0"],
                "holonomy_element": 0,
            },
        ],
    },
    "is_fundamental_domain": True,
    "polytope_volume": {"num": "1", "den": "1"},
    "quotient_covolume": {"num": "1", "den": "1"},
    "overlap_translation": None,
    "overlap_holonomy_element": None,
}

TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.translation_torus.quotient_chains.compute",
        title="Construct quotient chains of a translation 3-torus",
        description=(
            "Return the integral product cellular chain complex for a verified "
            "rank-three pure translation group with a parallelepiped fundamental "
            "domain. This bounded slice has eight vertices, six facets, and "
            "opposite facet translations; it does not construct general "
            "three-dimensional Bieberbach face orbits."
        ),
        request_type=CrystallographicFundamentalDomainResult,
        result_type=BieberbachTranslationTorusChains,
        run=translation_torus_quotient_chains,
        tags=("Bieberbach-group", "quotient-chains", "integral-homology", "exact"),
        discovery_terms=(
            "integral cellular chains of a three dimensional torus",
            "quotient homology of a crystallographic translation lattice",
            "Bieberbach translation group quotient chain complex",
        ),
        examples=(
            OperationExample(
                name="unit_cube_three_torus",
                description="Compute the integral quotient chains for the unit cube translation lattice.",
                input=_UNIT_CUBE_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
