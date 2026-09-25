"""Public tool manifest for Bieberbach polygon free resolutions."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.crystallographic.extensions.resolutions._models import (
    BieberbachPolygonFreeResolution,
    BieberbachPolygonFreeResolutionRequest,
)
from jacobian.math.geometry.crystallographic.extensions.resolutions.operations import (
    polygon_free_resolution,
)


def _compute(
    request: BieberbachPolygonFreeResolutionRequest,
) -> BieberbachPolygonFreeResolution:
    return polygon_free_resolution(request.source)


_UNIT_SQUARE_EXAMPLE = {
    "source": {
        "source": {
            "source": {
                "affine_realization": {
                    "source": {
                        "multiplication_table": [[0]],
                        "action_matrices": [[["1", "0"], ["0", "1"]]],
                        "factor_set": [[["0", "0"]]],
                    },
                    "section_maps": [
                        {
                            "holonomy_element": 0,
                            "linear_part": [["1", "0"], ["0", "1"]],
                            "section_shift": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        }
                    ],
                },
                "polytope": {
                    "space": {"axes": ["x", "y"]},
                    "vertices": [
                        {
                            "vertex_id": "v0",
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        },
                        {
                            "vertex_id": "v1",
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        },
                        {
                            "vertex_id": "v2",
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        },
                        {
                            "vertex_id": "v3",
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        },
                    ],
                },
                "lattice_axes": ["x", "y"],
                "facet_profile": {
                    "vertices": [
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                    ],
                    "dimension": 2,
                    "facets": [
                        {
                            "halfspace": {
                                "coefficients": [
                                    {"num": "-1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                                "offset": {"num": "0", "den": "1"},
                            },
                            "source_vertex_indices": [0, 1],
                        },
                        {
                            "halfspace": {
                                "coefficients": [
                                    {"num": "0", "den": "1"},
                                    {"num": "-1", "den": "1"},
                                ],
                                "offset": {"num": "0", "den": "1"},
                            },
                            "source_vertex_indices": [0, 2],
                        },
                        {
                            "halfspace": {
                                "coefficients": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                                "offset": {"num": "1", "den": "1"},
                            },
                            "source_vertex_indices": [1, 3],
                        },
                        {
                            "halfspace": {
                                "coefficients": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                                "offset": {"num": "1", "den": "1"},
                            },
                            "source_vertex_indices": [2, 3],
                        },
                    ],
                },
                "pairings": [
                    {
                        "source_facet_index": 0,
                        "target_facet_index": 3,
                        "lattice_translation": ["1", "0"],
                        "holonomy_element": 0,
                    },
                    {
                        "source_facet_index": 1,
                        "target_facet_index": 2,
                        "lattice_translation": ["0", "1"],
                        "holonomy_element": 0,
                    },
                    {
                        "source_facet_index": 2,
                        "target_facet_index": 1,
                        "lattice_translation": ["0", "-1"],
                        "holonomy_element": 0,
                    },
                    {
                        "source_facet_index": 3,
                        "target_facet_index": 0,
                        "lattice_translation": ["-1", "0"],
                        "holonomy_element": 0,
                    },
                ],
            },
            "is_fundamental_domain": True,
            "polytope_volume": {"num": "1", "den": "1"},
            "quotient_covolume": {"num": "1", "den": "1"},
            "overlap_translation": None,
            "overlap_holonomy_element": None,
        },
        "vertex_orbits": [[0, 1, 2, 3]],
        "edge_orbit_representatives": [0, 1],
        "orbit_maps": [
            {
                "source_facet_index": 0,
                "source_vertex_index": 0,
                "target_facet_index": 3,
                "target_vertex_index": 2,
                "lattice_translation": ["1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 0,
                "source_vertex_index": 1,
                "target_facet_index": 3,
                "target_vertex_index": 3,
                "lattice_translation": ["1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 1,
                "source_vertex_index": 0,
                "target_facet_index": 2,
                "target_vertex_index": 1,
                "lattice_translation": ["0", "1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 1,
                "source_vertex_index": 2,
                "target_facet_index": 2,
                "target_vertex_index": 3,
                "lattice_translation": ["0", "1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 2,
                "source_vertex_index": 1,
                "target_facet_index": 1,
                "target_vertex_index": 0,
                "lattice_translation": ["0", "-1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 2,
                "source_vertex_index": 3,
                "target_facet_index": 1,
                "target_vertex_index": 2,
                "lattice_translation": ["0", "-1"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 3,
                "source_vertex_index": 2,
                "target_facet_index": 0,
                "target_vertex_index": 0,
                "lattice_translation": ["-1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_facet_index": 3,
                "source_vertex_index": 3,
                "target_facet_index": 0,
                "target_vertex_index": 1,
                "lattice_translation": ["-1", "0"],
                "holonomy_element": 0,
            },
        ],
        "boundary_1_to_0": [
            {
                "source_cell_index": 0,
                "target_cell_index": 0,
                "coefficient": -1,
                "incidence_index": 1,
                "lattice_translation": ["0", "-1"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 0,
                "target_cell_index": 0,
                "coefficient": 1,
                "incidence_index": 0,
                "lattice_translation": ["0", "0"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 1,
                "target_cell_index": 0,
                "coefficient": -1,
                "incidence_index": 0,
                "lattice_translation": ["0", "0"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 1,
                "target_cell_index": 0,
                "coefficient": 1,
                "incidence_index": 2,
                "lattice_translation": ["-1", "0"],
                "holonomy_element": 0,
            },
        ],
        "boundary_2_to_1": [
            {
                "source_cell_index": 0,
                "target_cell_index": 1,
                "coefficient": 1,
                "incidence_index": 1,
                "lattice_translation": ["0", "0"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 0,
                "target_cell_index": 0,
                "coefficient": -1,
                "incidence_index": 3,
                "lattice_translation": ["-1", "0"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 0,
                "target_cell_index": 1,
                "coefficient": -1,
                "incidence_index": 2,
                "lattice_translation": ["0", "-1"],
                "holonomy_element": 0,
            },
            {
                "source_cell_index": 0,
                "target_cell_index": 0,
                "coefficient": 1,
                "incidence_index": 0,
                "lattice_translation": ["0", "0"],
                "holonomy_element": 0,
            },
        ],
        "quotient_chain_complex": {
            "coefficient_ring": "ZZ",
            "prime": None,
            "degree_min": 0,
            "degree_max": 2,
            "basis_sizes": [1, 2, 1],
            "differential_matrices": [[["0", "0"]], [["0"], ["0"]]],
        },
    }
}

TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.bieberbach.polygon_free_resolution.compute",
        title="Construct a Bieberbach polygon free resolution",
        description=(
            "Construct the finite cellular free ZGamma resolution through degree "
            "two from a verified torsion-free paired polygon. The exact sparse "
            "group-ring boundaries use the extension's lattice translations and "
            "holonomy indices; augmentation gives the integral quotient chains. "
            "The admitted scope is dimension two, at most 32 polygon vertices "
            "and facets, and the source-bound face-orbit contract."
        ),
        request_type=BieberbachPolygonFreeResolutionRequest,
        result_type=BieberbachPolygonFreeResolution,
        run=_compute,
        tags=("Bieberbach-group", "free-resolution", "group-homology", "exact"),
        discovery_terms=(
            "free integral group-ring resolution of a Bieberbach group",
            "cellular resolution from a flat polygon side pairing",
            "group homology resolution of a Klein bottle group",
        ),
        examples=(
            OperationExample(
                name="unit_square_torus",
                description="Construct the integral free resolution for the translation action on a unit square.",
                input=_UNIT_SQUARE_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
