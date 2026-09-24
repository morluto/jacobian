"""Public finite crystallographic-extension tools."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    CrystallographicAffineRealization,
    CrystallographicExtensionTorsionResult,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingRequest,
    CrystallographicPolytopePairingResult,
    FiniteLatticeExtension,
)
from jacobian.math.geometry.crystallographic.extensions.face_orbits import (
    quotient_face_orbit_complex,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    decide_extension_torsion,
    pair_crystallographic_polytope_facets,
)


def _decide(request: FiniteLatticeExtension) -> CrystallographicExtensionTorsionResult:
    return decide_extension_torsion(request)


def _pair_polytope(
    request: CrystallographicPolytopePairingRequest,
) -> CrystallographicPolytopePairingResult:
    return pair_crystallographic_polytope_facets(request)


def _check_fundamental_domain(
    request: CrystallographicPolytopePairingResult,
) -> CrystallographicFundamentalDomainResult:
    return check_crystallographic_fundamental_domain(request)


_TOY_KLEIN_EXTENSION = {
    "multiplication_table": [[0, 1], [1, 0]],
    "action_matrices": [
        [["1", "0"], ["0", "1"]],
        [["1", "0"], ["0", "-1"]],
    ],
    "factor_set": [
        [["0", "0"], ["0", "0"]],
        [["0", "0"], ["1", "0"]],
    ],
}

_S3_SECTION_EXTENSION = {
    "multiplication_table": [
        [0, 1, 2, 3, 4, 5],
        [1, 0, 4, 5, 2, 3],
        [2, 5, 0, 4, 3, 1],
        [3, 4, 5, 0, 1, 2],
        [4, 3, 1, 2, 5, 0],
        [5, 2, 3, 1, 0, 4],
    ],
    "action_matrices": [
        [["1", "0"], ["0", "1"]],
        [["-1", "1"], ["0", "1"]],
        [["1", "0"], ["1", "-1"]],
        [["0", "-1"], ["-1", "0"]],
        [["0", "-1"], ["1", "-1"]],
        [["-1", "1"], ["-1", "0"]],
    ],
    # This is the integral coboundary of u_(23)=(1,0), with all other u_g=0.
    "factor_set": [
        [["0", "0"] for _ in range(6)],
        [
            ["0", "0"],
            ["0", "0"],
            ["-1", "0"],
            ["0", "0"],
            ["-1", "0"],
            ["0", "0"],
        ],
        [
            ["0", "0"],
            ["1", "0"],
            ["2", "1"],
            ["1", "0"],
            ["1", "0"],
            ["1", "0"],
        ],
        [
            ["0", "0"],
            ["0", "0"],
            ["0", "-1"],
            ["0", "0"],
            ["0", "0"],
            ["-1", "0"],
        ],
        [
            ["0", "0"],
            ["0", "0"],
            ["0", "1"],
            ["-1", "0"],
            ["0", "0"],
            ["0", "0"],
        ],
        [
            ["0", "0"],
            ["-1", "0"],
            ["-1", "-1"],
            ["0", "0"],
            ["0", "0"],
            ["0", "0"],
        ],
    ],
}

_TRANSLATION_EXTENSION = {
    "multiplication_table": [[0]],
    "action_matrices": [[["1", "0"], ["0", "1"]]],
    "factor_set": [[["0", "0"]]],
}

_UNIT_SQUARE_PAIRING = {
    "affine_realization": {
        "source": _TRANSLATION_EXTENSION,
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
                "vertex_id": "v00",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "vertex_id": "v01",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            },
            {
                "vertex_id": "v10",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "vertex_id": "v11",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
            },
        ],
    },
    "lattice_axes": ["x", "y"],
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
}

_UNIT_SQUARE_PAIRING_RESULT = {
    "affine_realization": _UNIT_SQUARE_PAIRING["affine_realization"],
    "polytope": _UNIT_SQUARE_PAIRING["polytope"],
    "lattice_axes": ["x", "y"],
    "facet_profile": {
        "vertices": [
            {"coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}]},
            {"coordinates": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}]},
            {"coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}]},
            {"coordinates": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}]},
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
    "pairings": _UNIT_SQUARE_PAIRING["pairings"],
}

TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.quotient_face_orbits.compute",
        title="Construct a bounded Bieberbach polygon quotient complex",
        description=(
            "Recheck a rank-two crystallographic fundamental polygon and use "
            "its exact directed side pairings to compute vertex and edge orbits, "
            "group-labelled face incidences, and the augmented integral cellular "
            "chain complex of the quotient. This operation supports at most 32 "
            "vertices and facets. It does not return or claim a free ZGamma "
            "resolution."
        ),
        request_type=CrystallographicFundamentalDomainResult,
        result_type=BieberbachFaceOrbitComplex,
        run=quotient_face_orbit_complex,
        tags=("crystallographic-group", "Bieberbach", "quotient-cell-complex", "exact"),
        discovery_terms=(
            "Bieberbach flat surface integral quotient cellular chains",
            "compute Klein bottle or torus chain complex from exact side pairings",
            "crystallographic polygon face orbits with group-labelled boundaries",
        ),
    ),
    MathTool(
        operation_id="crystallographic.extension.fundamental_domain.check",
        title="Check a crystallographic polytope fundamental domain",
        description=(
            "Recompute the affine realization and complete facet ledger, then "
            "enumerate every potentially intersecting lattice translate of each "
            "holonomy image. Return an exact nonidentity full-dimensional overlap "
            "witness when present and compare exact polytope volume with 1/|G|. "
            "Facet pairings alone do not establish tiling. The bounded theorem "
            "applies to faithful cocompact actions in rank at most four and "
            "holonomy order at most eight."
        ),
        request_type=CrystallographicPolytopePairingResult,
        result_type=CrystallographicFundamentalDomainResult,
        run=_check_fundamental_domain,
        tags=("crystallographic-group", "fundamental-domain", "exact"),
        discovery_terms=(
            "check crystallographic fundamental polytope",
            "exact tiling by crystallographic group translates",
            "Bieberbach polytope volume covolume overlap check",
        ),
        examples=(
            OperationExample(
                name="unit_square_fundamental_domain",
                description=(
                    "Recheck the source-bound side-pairing result and prove that "
                    "the unit square has no full-dimensional translate overlap "
                    "and has exact covolume one. Boundary contacts are permitted."
                ),
                input=_UNIT_SQUARE_PAIRING_RESULT,
            ),
        ),
    ),
    MathTool(
        operation_id="crystallographic.extension.polytope_facet_pairings.compute",
        title="Validate exact side pairings of a crystallographic polytope",
        description=(
            "Compute the complete exact facet profile of a bounded rational "
            "V-polytope and validate a complete directed ledger of extension "
            "elements mapping each facet's full vertex set bijectively to its "
            "target. Every reverse entry must be the exact group inverse. "
            "The result binds the profile, polytope, axes, and affine realization; "
            "it does not establish a tiling, quotient cell structure, "
            "torsion-freeness, or a resolution."
        ),
        request_type=CrystallographicPolytopePairingRequest,
        result_type=CrystallographicPolytopePairingResult,
        run=_pair_polytope,
        tags=("crystallographic-group", "polytope", "facet-pairing", "exact"),
        discovery_terms=(
            "crystallographic polytope facet side pairing",
            "exact group element maps facets of rational polytope",
            "fundamental polytope candidate facet pairing ledger",
        ),
        examples=(
            OperationExample(
                name="unit_square_translation_pairings",
                description=(
                    "Pair opposite sides of the unit square by the positive and "
                    "negative coordinate translations in Z^2. The value records "
                    "exact side maps and inverse pairings only; it does not assert "
                    "that translates tile the plane."
                ),
                input=_UNIT_SQUARE_PAIRING,
            ),
        ),
    ),
    MathTool(
        operation_id="crystallographic.extension.affine_realization.compute",
        title="Realize a finite lattice extension by exact affine section maps",
        description=(
            "Return the canonical rational shifts q_g=(1/|G|) sum_h f(g,h) "
            "and source-bound affine section maps A_g(x)=rho(g)x+q_g. They "
            "satisfy q_g+rho(g)q_h-q_(gh)=f(g,h), so A_g composed with A_h "
            "equals the lattice translation T_f(g,h) composed with A_(gh); "
            "for nonzero cocycle, section maps are not a representation of G. "
            "The full extension acts by (v,g) -> T_v composed with A_g. This "
            "bounded operation realizes affine algebra only; it makes no claim "
            "about torsion-freeness, fundamental domains, or resolutions."
        ),
        request_type=FiniteLatticeExtension,
        result_type=CrystallographicAffineRealization,
        run=affine_section_realization,
        tags=("crystallographic-group", "affine-action", "group-extension", "exact"),
        discovery_terms=(
            "affine realization of a crystallographic extension",
            "rational section shift from a group cocycle",
            "affine action of a finite lattice extension",
        ),
        examples=(
            OperationExample(
                name="s3_nonzero_cocycle_section",
                description=(
                    "Realize a two-dimensional faithful S3 lattice action with a "
                    "nonzero integral coboundary. The (23) section map squared "
                    "is translation by (2,1), followed by the identity section map."
                ),
                input=_S3_SECTION_EXTENSION,
            ),
            OperationExample(
                name="rank_two_translation_lattice",
                description=(
                    "Realize the trivial-holonomy extension; its section map is "
                    "the identity and arbitrary lattice translations complete "
                    "the full affine action."
                ),
                input=_TRANSLATION_EXTENSION,
            ),
        ),
    ),
    MathTool(
        operation_id="crystallographic.extension.torsion_freeness.decide",
        title="Decide torsion-freeness of a finite lattice extension",
        description=(
            "Validate a bounded finite group table, its faithful integral action "
            "on Z^n, and a normalized integral 2-cocycle defining an extension. "
            "For every nonidentity holonomy element, solve the exact norm-map "
            "integer equation for a finite-order lift, returning either a lift or "
            "Smith divisibility obstructions. Torsion-free outputs are "
            "Bieberbach groups with the supplied lattice as their full translation "
            "subgroup; this operation does not construct a fundamental domain or "
            "cellular resolution."
        ),
        request_type=FiniteLatticeExtension,
        result_type=CrystallographicExtensionTorsionResult,
        run=_decide,
        tags=(
            "crystallographic-group",
            "Bieberbach-group",
            "group-extension",
            "2-cocycle",
            "torsion-freeness",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "finite holonomy lattice extension",
            "Bieberbach group extension cocycle",
            "torsion-free crystallographic group",
            "integral action and group 2-cocycle",
            "finite-order lift norm equation",
        ),
        examples=(
            OperationExample(
                name="klein_bottle_extension_is_torsion_free",
                description=(
                    "The order-two reflection fixes the first lattice direction; "
                    "the cocycle value (1,0) makes every lift square to an odd "
                    "translation in that direction, so no lift has finite order."
                ),
                input=_TOY_KLEIN_EXTENSION,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
