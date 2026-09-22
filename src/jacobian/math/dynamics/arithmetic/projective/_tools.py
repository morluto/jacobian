from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.dynamics.arithmetic.projective._kernel import (
    _preflight_critical_orbits,
    apply_projective_map,
    compose_projective_maps,
    is_critical_point,
    projective_orbit,
)
from jacobian.math.dynamics.arithmetic.projective_models import (
    CriticalOrbitRequest,
    CriticalOrbitResult,
    CriticalOrbitRow,
    ExactPeriodResult,
    ProjectiveApplyRequest,
    ProjectiveComposeRequest,
    ProjectiveMapResult,
    ProjectiveOrbitRequest,
    ProjectivePointResult,
)


def compute_projective_apply(request: ProjectiveApplyRequest) -> ProjectivePointResult:
    return ProjectivePointResult(
        map=request.map,
        point=request.point,
        image=apply_projective_map(request.map, request.point),
    )


def compute_projective_compose(
    request: ProjectiveComposeRequest,
) -> ProjectiveMapResult:
    return ProjectiveMapResult(
        outer=request.outer,
        inner=request.inner,
        composition=compose_projective_maps(request.outer, request.inner),
    )


def compute_critical_orbit(request: CriticalOrbitRequest) -> CriticalOrbitResult:
    # Admit the complete retained batch before derivative checks or orbit work.
    _preflight_critical_orbits(request.map, request.critical_points, request.max_steps)
    rows = []
    for point in request.critical_points:
        if not is_critical_point(request.map, point):
            raise OperationDomainValidationError(
                location=("critical_points",),
                code="arithmetic_dynamics.projective_not_critical",
                message="every supplied point must be critical for the homogeneous map",
            )
        orbit, repeat = projective_orbit(request.map, point, request.max_steps)
        rows.append(
            CriticalOrbitRow(
                point=point,
                orbit=orbit,
                termination="REPEAT_FOUND" if repeat else "STEP_BOUND_REACHED",
                period=repeat[2] if repeat else None,
            )
        )
    return CriticalOrbitResult(map=request.map, rows=tuple(rows))


def compute_projective_orbit(request: ProjectiveOrbitRequest) -> ExactPeriodResult:
    orbit, repeat = projective_orbit(request.map, request.start, request.max_steps)
    found = repeat is not None
    first = repeat[0] if repeat else None
    period = repeat[2] if repeat else None
    return ExactPeriodResult.from_kernel(
        map=request.map,
        start=request.start,
        orbit=orbit,
        max_steps=request.max_steps,
        termination="REPEAT_FOUND" if found else "STEP_BOUND_REACHED",
        first_seen_index=first,
        period=period,
        exact_period=bool(found and first == 0),
    )


TOOLS = (
    MathTool(
        operation_id="arithmetic_dynamics.projective_map.apply.compute",
        title="Apply a homogeneous projective map",
        description="Apply a homogeneous rational map to a normalized point of P1(Q).",
        request_type=ProjectiveApplyRequest,
        result_type=ProjectivePointResult,
        run=compute_projective_apply,
        tags=("arithmetic-dynamics", "projective", "exact"),
        examples=(
            OperationExample(
                name="apply_infinity",
                description="Apply the homogeneous map [X:Y] -> [X^2:Y^2] to infinity; the point must be normalized.",
                input={
                    "map": {
                        "degree": 2,
                        "numerator": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "denominator": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "point": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.projective_map.compose.compute",
        title="Compose homogeneous projective maps",
        description="Compose two homogeneous maps with exact coefficient reconstruction.",
        request_type=ProjectiveComposeRequest,
        result_type=ProjectiveMapResult,
        run=compute_projective_compose,
        tags=("arithmetic-dynamics", "projective", "exact"),
        examples=(
            OperationExample(
                name="compose_identity",
                description="Compose homogeneous maps exactly; composition degree must remain in the bounded envelope.",
                input={
                    "outer": {
                        "degree": 1,
                        "numerator": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "denominator": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "inner": {
                        "degree": 1,
                        "numerator": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "denominator": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.projective_map.critical_orbit.compute",
        title="Compute bounded critical orbits",
        description="Compute exact critical-orbit prefixes; a finite prefix never proves global postcritical finiteness.",
        request_type=CriticalOrbitRequest,
        result_type=CriticalOrbitResult,
        run=compute_critical_orbit,
        tags=("arithmetic-dynamics", "projective", "critical", "exact"),
        examples=(
            OperationExample(
                name="critical_fixed",
                description="Compute the critical orbit prefix of z squared at zero; every supplied point must be a critical point.",
                input={
                    "map": {
                        "degree": 2,
                        "numerator": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "denominator": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "critical_points": [
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}}
                    ],
                    "max_steps": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic_dynamics.projective_point.exact_period.compute",
        title="Compute a bounded exact-period profile",
        description="Compute a projective orbit prefix and distinguish an exact period from a later repeat; a step bound never proves nonperiodicity.",
        request_type=ProjectiveOrbitRequest,
        result_type=ExactPeriodResult,
        run=compute_projective_orbit,
        tags=("arithmetic-dynamics", "projective", "period", "exact"),
        examples=(
            OperationExample(
                name="fixed_point",
                description="Compute the period profile of the fixed point 0; a repeat at index zero establishes exact period one.",
                input={
                    "map": {
                        "degree": 1,
                        "numerator": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "denominator": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "start": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "max_steps": 4,
                },
            ),
        ),
    ),
)
__all__ = [
    "TOOLS",
    "compute_critical_orbit",
    "compute_projective_apply",
    "compute_projective_compose",
    "compute_projective_orbit",
]
