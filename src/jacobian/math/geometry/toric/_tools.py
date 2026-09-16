"""Public declarations for exact rational toric-geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CHART_BOX,
    MAX_TORIC_CHART_CANDIDATES,
    MAX_TORIC_CHART_DUAL_RAYS,
    MAX_TORIC_CHART_GENERATORS,
    MAX_TORIC_CHART_RELATIONS,
    MAX_TORIC_CONE_COUNT,
    MAX_TORIC_CONE_GENERATORS,
    MAX_TORIC_COORDINATE_DIGITS,
    MAX_TORIC_LATTICE_RANK,
    MAX_TORIC_MORPHISM_WORK,
    MAX_TORIC_RAY_COUNT,
    CharacterDivisorRequest,
    CharacterDivisorResult,
    FanValidationRequest,
    FanValidationResult,
    OrbitConeProfileRequest,
    OrbitConeProfileResult,
    ToricAffineChartRequest,
    ToricAffineChartResult,
    ToricMorphismRequest,
    ToricMorphismResult,
)
from jacobian.math.geometry.toric.operations import (
    check_toric_morphism,
    compute_affine_chart,
    compute_character_divisor,
    compute_orbit_cone_profile,
    validate_fan,
)

_ENVELOPE_SENTENCE = (
    f"Envelope: lattice rank at most {MAX_TORIC_LATTICE_RANK}, at most "
    f"{MAX_TORIC_RAY_COUNT} rays, {MAX_TORIC_CONE_COUNT} cones, "
    f"{MAX_TORIC_CONE_GENERATORS} generators per cone, and "
    f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits per coordinate."
)


def _run_validate(request: FanValidationRequest) -> FanValidationResult:
    return validate_fan(request.fan)


def _run_orbit_profile(request: OrbitConeProfileRequest) -> OrbitConeProfileResult:
    return compute_orbit_cone_profile(request.fan)


def _run_character_divisor(
    request: CharacterDivisorRequest,
) -> CharacterDivisorResult:
    return compute_character_divisor(request.fan, request.character)


def _run_affine_chart(request: ToricAffineChartRequest) -> ToricAffineChartResult:
    return compute_affine_chart(request.fan, request.cone)


def _run_morphism(request: ToricMorphismRequest) -> ToricMorphismResult:
    return check_toric_morphism(request.source, request.target, request.matrix)


FAN_VALIDATE_OPERATION = MathTool(
    operation_id="toric.fan.validate",
    title="Recognize an exact rational fan presentation",
    description=(
        "Decide whether a proposed finite fan in Z^n is a valid rational fan: "
        "primitive distinct rays, strongly convex cones, every declared "
        "generator an extreme ray, complete face closure, and pairwise "
        "intersections equal to common faces. Return VALID or INVALID with the "
        "first exact obstruction code and the retained source fan. Recognition "
        "is exact integer and rational Fourier-Motzkin arithmetic; no floats. "
        + _ENVELOPE_SENTENCE
    ),
    request_type=FanValidationRequest,
    result_type=FanValidationResult,
    run=_run_validate,
    tags=("geometry", "toric", "fan", "exact"),
    discovery_terms=(
        "rational fan validation",
        "strongly convex rational polyhedral cone",
        "fan face closure",
        "toric variety fan check",
    ),
    examples=(
        OperationExample(
            name="projective_plane_fan",
            description=(
                "Recognize the complete fan of P^2 from its three primitive "
                "rays and seven cones; precondition: strictly increasing "
                "unique ray-index tuples inside the published envelope."
            ),
            input={
                "fan": {
                    "lattice_rank": 2,
                    "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
                    "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
                }
            },
        ),
    ),
)

ORBIT_CONE_PROFILE_OPERATION = MathTool(
    operation_id="toric.orbit_cone_profile.compute",
    title="Compute the orbit-cone profile of a rational fan",
    description=(
        "For a validated rational fan in Z^n, return the orbit-cone "
        "correspondence: one row per cone with its generator-span dimension, "
        "orbit dimension n - dim, simpliciality, and smoothness (Smith "
        "invariant factors all 1); all face relations tau <= sigma; and "
        "ray-cone incidence. Admission re-recognizes the caller-authored fan "
        "exactly once and reuses those facts. " + _ENVELOPE_SENTENCE
    ),
    request_type=OrbitConeProfileRequest,
    result_type=OrbitConeProfileResult,
    run=_run_orbit_profile,
    tags=("geometry", "toric", "orbit-cone", "smoothness", "exact"),
    discovery_terms=(
        "orbit-cone correspondence",
        "torus orbit dimensions",
        "smooth toric variety check",
        "fan face lattice",
    ),
    examples=(
        OperationExample(
            name="degree_six_del_pezzo_profile",
            description=(
                "Compute all thirteen orbit rows, face relations, and per-cone "
                "smoothness of the degree-6 del Pezzo fan; precondition: a fan "
                "passing exact recognition inside the published envelope."
            ),
            input={
                "fan": {
                    "lattice_rank": 2,
                    "rays": [
                        ["1", "0"],
                        ["0", "1"],
                        ["-1", "0"],
                        ["0", "-1"],
                        ["1", "1"],
                        ["-1", "-1"],
                    ],
                    "cones": [
                        [],
                        [0],
                        [1],
                        [2],
                        [3],
                        [4],
                        [5],
                        [0, 3],
                        [0, 4],
                        [1, 2],
                        [1, 4],
                        [2, 5],
                        [3, 5],
                    ],
                }
            },
        ),
    ),
)

CHARACTER_DIVISOR_OPERATION = MathTool(
    operation_id="toric.character_divisor.compute",
    title="Compute the principal toric character divisor div(chi^m)",
    description=(
        "For a validated rational fan and an integer character m in the dual "
        "lattice, return the principal T-invariant divisor div(chi^m) = "
        "sum_rho <m, v_rho> D_rho: exact integer coefficients in declared ray "
        "order with zero coefficients retained, is_principal, and the nonzero "
        "support. Admission re-recognizes the fan exactly once before pairing. "
        + _ENVELOPE_SENTENCE
    ),
    request_type=CharacterDivisorRequest,
    result_type=CharacterDivisorResult,
    run=_run_character_divisor,
    tags=("geometry", "toric", "divisor", "character", "exact"),
    discovery_terms=(
        "principal torus-invariant divisor",
        "character divisor of a toric variety",
        "divisor of a monomial",
        "toric Weil divisor pairing",
    ),
    examples=(
        OperationExample(
            name="affine_plane_character_divisor",
            description=(
                "Compute div(chi^m) for m = (1, -2) on the affine plane fan "
                "with rays e1, e2; precondition: an exactly recognized fan and "
                "an integer character of the dual lattice."
            ),
            input={
                "fan": {
                    "lattice_rank": 2,
                    "rays": [["1", "0"], ["0", "1"]],
                    "cones": [[], [0], [1], [0, 1]],
                },
                "character": {"entries": ["1", "-2"]},
            },
        ),
    ),
)

AFFINE_CHART_OPERATION = MathTool(
    operation_id="toric.affine_chart.compute",
    title="Compute an exact affine monomial toric chart",
    description=(
        "For one cone of a validated rational fan, return the affine chart "
        "Spec k[sigma^vee cap M]: the primitive extreme rays of the exact dual "
        "cone, the complete Hilbert basis of the affine semigroup (reduced from "
        "the fundamental-parallelepiped candidates), the canonical integer "
        "relation lattice with every relation replayed to zero on the "
        "generators, and the face localizations inverting the canonical "
        "supporting character of each proper face. A lower-dimensional cone is "
        "presented by the free (torus) lineality factor sigma^perp cap M in "
        "``torus_basis`` plus the pointed quotient Hilbert basis, so the total "
        "semigroup is generated by ``hilbert_basis`` over the free part. Only "
        "cones whose derived dual cone, parallelepiped, Hilbert basis, and "
        "relation lattice fit the "
        f"published envelope (at most {MAX_TORIC_CHART_DUAL_RAYS} dual rays, "
        f"{MAX_TORIC_CHART_BOX} parallelepiped box points, "
        f"{MAX_TORIC_CHART_CANDIDATES} candidates, "
        f"{MAX_TORIC_CHART_GENERATORS} generators, "
        f"{MAX_TORIC_CHART_RELATIONS} relations) are admitted; an unbounded "
        "request is refused rather than returned with an incomplete basis. "
        + _ENVELOPE_SENTENCE
    ),
    request_type=ToricAffineChartRequest,
    result_type=ToricAffineChartResult,
    run=_run_affine_chart,
    tags=("geometry", "toric", "affine-chart", "hilbert-basis", "exact"),
    discovery_terms=(
        "affine toric chart",
        "monoid algebra of a cone",
        "Hilbert basis of a rational cone",
        "dual cone semigroup presentation",
    ),
    examples=(
        OperationExample(
            name="affine_plane_chart",
            description=(
                "Compute the affine chart of the two-dimensional cone spanned by "
                "e1 and e2 in the affine-plane fan: the polynomial chart "
                "Spec k[x, y] with two Hilbert generators, no relations, and the "
                "three face localizations."
            ),
            input={
                "fan": {
                    "lattice_rank": 2,
                    "rays": [["1", "0"], ["0", "1"]],
                    "cones": [[], [0], [1], [0, 1]],
                },
                "cone": [0, 1],
            },
        ),
    ),
)

MORPHISM_CHECK_OPERATION = MathTool(
    operation_id="toric.morphism.check",
    title="Check an exact lattice-equivariant toric morphism",
    description=(
        "For two validated rational fans and an integer lattice matrix "
        "phi: N_1 -> N_2, decide the toric-morphism condition: every source cone "
        "must map into a single target cone. Return the induced fan-compatible "
        "source-cone to target-cone assignment with the source ray images, or the "
        "first obstructing source cone with its image vectors and, when a single "
        "ray image already leaves every target cone, that ray and image as the "
        "witness. Every assignment is replayed by exact cone-membership of the "
        "ray images. Admission bounds the fan recognition for both fans and the "
        f"exact cone-membership work (at most {MAX_TORIC_MORPHISM_WORK} "
        "source-cone/ray/target-cone tests). " + _ENVELOPE_SENTENCE
    ),
    request_type=ToricMorphismRequest,
    result_type=ToricMorphismResult,
    run=_run_morphism,
    tags=("geometry", "toric", "morphism", "fan-map", "exact"),
    discovery_terms=(
        "toric morphism",
        "fan-compatible lattice map",
        "equivariant map of toric varieties",
        "lattice homomorphism of fans",
    ),
    examples=(
        OperationExample(
            name="projective_plane_identity",
            description=(
                "Check that the identity lattice map is a toric morphism from the "
                "projective-plane fan to itself, returning the induced cone "
                "assignments and ray images."
            ),
            input={
                "source": {
                    "lattice_rank": 2,
                    "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
                    "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
                },
                "target": {
                    "lattice_rank": 2,
                    "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
                    "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
                },
                "matrix": {
                    "row_count": 2,
                    "column_count": 2,
                    "entries": [["1", "0"], ["0", "1"]],
                },
            },
        ),
    ),
)

TOOLS: MathTools = (
    FAN_VALIDATE_OPERATION,
    ORBIT_CONE_PROFILE_OPERATION,
    CHARACTER_DIVISOR_OPERATION,
    AFFINE_CHART_OPERATION,
    MORPHISM_CHECK_OPERATION,
)

__all__ = ["TOOLS"]
