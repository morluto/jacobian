"""Public declarations for exact rational toric-geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CONE_COUNT,
    MAX_TORIC_CONE_GENERATORS,
    MAX_TORIC_COORDINATE_DIGITS,
    MAX_TORIC_LATTICE_RANK,
    MAX_TORIC_RAY_COUNT,
    CharacterDivisorRequest,
    CharacterDivisorResult,
    FanValidationRequest,
    FanValidationResult,
    OrbitConeProfileRequest,
    OrbitConeProfileResult,
)
from jacobian.math.geometry.toric.operations import (
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

TOOLS: MathTools = (
    FAN_VALIDATE_OPERATION,
    ORBIT_CONE_PROFILE_OPERATION,
    CHARACTER_DIVISOR_OPERATION,
)

__all__ = ["TOOLS"]
