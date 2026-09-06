"""Real algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingProfile,
    CommonInterlacingRequest,
)
from jacobian.math.polynomials.real_algebra._models import (
    RootCountRequest,
    RootCountResult,
    SturmChainRequest,
    SturmChainResult,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    PlaneComponentProfileRequest,
    PlaneComponentProfileResult,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    StrictSublevelMeasureRequest,
    StrictSublevelMeasureResult,
)
from jacobian.math.polynomials.real_algebra.operations import (
    common_interlacing_profile as _common_interlacing_profile_native,
)
from jacobian.math.polynomials.real_algebra.operations import (
    compute_plane_component_profile as _compute_plane_component_profile_native,
)
from jacobian.math.polynomials.real_algebra.operations import (
    compute_root_count as _compute_root_count_native,
)
from jacobian.math.polynomials.real_algebra.operations import (
    compute_strict_sublevel_measure as _compute_strict_sublevel_measure_native,
)
from jacobian.math.polynomials.real_algebra.operations import (
    compute_sturm_chain as _compute_sturm_chain_native,
)


def compute_sturm_chain(request: SturmChainRequest) -> SturmChainResult:
    return _compute_sturm_chain_native(request.polynomial)


def compute_root_count(request: RootCountRequest) -> RootCountResult:
    return _compute_root_count_native(request.polynomial, request.lower, request.upper)


def compute_strict_sublevel_measure(
    request: StrictSublevelMeasureRequest,
) -> StrictSublevelMeasureResult:
    return _compute_strict_sublevel_measure_native(
        request.polynomial,
        request.threshold,
        request.lower,
        request.upper,
    )


def compute_plane_component_profile(
    request: PlaneComponentProfileRequest,
) -> PlaneComponentProfileResult:
    return _compute_plane_component_profile_native(
        request.semialgebraic_set,
        request.samples,
    )


def compute_common_interlacing_profile(
    request: CommonInterlacingRequest,
) -> CommonInterlacingProfile:
    return _common_interlacing_profile_native(request.family)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.real.common_interlacing_profile.compute",
        title="Compute an exact common polynomial-interlacing profile",
        description="Decide common weak interlacing for a bounded labelled family of monic, "
        "same-positive-degree univariate polynomials over QQ. Return every "
        "distinct exact real root with source multiplicity and either all "
        "attained closed gap endpoints or the first deterministic non-real-root "
        "or empty-gap obstruction. The envelope admits 2-8 sources, source "
        "degree at most 32, total degree at most 128, 64-digit rational "
        "components, 256-digit primitive height, and real-rooted irreducible "
        "factors of degree at most 8; higher-degree root-free factors are "
        "retained only to report the exact non-real-root obstruction.",
        request_type=CommonInterlacingRequest,
        result_type=CommonInterlacingProfile,
        run=compute_common_interlacing_profile,
        tags=("polynomial", "real-algebra", "common-interlacing", "exact"),
        discovery_terms=(
            "common interlacer",
            "common interlacing",
            "weak polynomial interlacing",
        ),
        examples=(
            OperationExample(
                name="quadratic_family",
                description="Compute the common gap [-1, 1] for x^2 - 1 and x^2 - 4.",
                input={
                    "family": [
                        {
                            "label": "inner",
                            "polynomial": {
                                "variables": ["x"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                        },
                        {
                            "label": "outer",
                            "polynomial": {
                                "variables": ["x"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2],
                                        },
                                        {
                                            "coefficient": {"num": "-4", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                        },
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.sturm_chain.compute",
        title="Compute an ordinary exact Sturm sequence",
        description="Compute SymPy's ordinary Euclidean-remainder Sturm sequence for a "
        "non-constant univariate polynomial with rational coefficients encoded "
        "canonically. Its primitive integer coefficients and input scalar components have at most 16 decimal digits. The current envelope is "
        "degree at most 32 and coefficients of at most 16 decimal digits.",
        request_type=SturmChainRequest,
        result_type=SturmChainResult,
        run=compute_sturm_chain,
        tags=("polynomial", "sturm-chain", "exact"),
        examples=(
            OperationExample(
                name="cubic",
                description="Sturm chain of x^3 - 2x^2 + x - 3.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "-3", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.root_count.compute",
        title="Count real roots in an interval via Sturm's theorem",
        description="Count distinct real roots of a bounded univariate polynomial with "
        "rational coefficients in the closed interval [lower, upper] using SymPy's "
        "ordinary exact Sturm sequence. The current envelope is degree at most "
        "32; primitive integer coefficients and input scalar components have at most 16 decimal digits.",
        request_type=RootCountRequest,
        result_type=RootCountResult,
        run=compute_root_count,
        tags=("polynomial", "root-count", "exact"),
        examples=(
            OperationExample(
                name="cubic",
                description="Count roots of x^3 - 2x^2 + x - 3 in [-10, 10].",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "-3", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "lower": {"num": "-10", "den": "1"},
                    "upper": {"num": "10", "den": "1"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.real.strict_sublevel_measure.compute",
        title="Compute an exact strict polynomial sublevel measure",
        description="Return the complete component decomposition and source-bound exact "
        "real-algebraic measure of {x in [lower, upper] : |f(x)| < threshold} "
        "for a canonical univariate polynomial over QQ.",
        request_type=StrictSublevelMeasureRequest,
        result_type=StrictSublevelMeasureResult,
        run=compute_strict_sublevel_measure,
        tags=("polynomial", "real-algebra", "sublevel-set", "measure", "exact"),
        examples=(
            OperationExample(
                name="quadratic_irrational_length",
                description="Measure |x^2| < 2 on [-2, 2], with endpoints at ±sqrt(2).",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                }
                            ]
                        },
                    },
                    "threshold": {"num": "2", "den": "1"},
                    "lower": {"num": "-2", "den": "1"},
                    "upper": {"num": "2", "den": "1"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="real_algebraic.plane_semialgebraic.component_profile.compute",
        runtime_requirements=("qepcad",),
        title="Compute exact components of a plane semialgebraic set",
        description="Return the complete connected-component partition of a bounded-size "
        "normalized sign table in R^2, one exact algebraic representative per "
        "component, and component IDs for supplied exact points. The pinned "
        "QEPCAD 1.74 backend computes sign-invariant CAD cell closures. Inputs "
        "have at most four degree-four QQ[x,y] polynomials, 48 total terms, "
        "32-digit rational coefficients, 81 sign rows, and eight degree-sixteen "
        "algebraic samples; projection work is preflighted. Backend non-completion "
        "uses the execution-error path and establishes no topological conclusion.",
        request_type=PlaneComponentProfileRequest,
        result_type=PlaneComponentProfileResult,
        run=compute_plane_component_profile,
        tags=(
            "real-algebraic-geometry",
            "semialgebraic-set",
            "connected-components",
            "exact",
        ),
        examples=(
            OperationExample(
                name="empty_sign_table",
                description="Compute the empty semialgebraic set represented by no accepted sign rows.",
                input={
                    "semialgebraic_set": {
                        "axis": ["x", "y"],
                        "polynomials": [
                            {
                                "variables": ["x", "y"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2, 0],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 2],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0, 0],
                                        },
                                    ]
                                },
                            },
                            {
                                "variables": ["x", "y"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2, 0],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 2],
                                        },
                                        {
                                            "coefficient": {"num": "-4", "den": "1"},
                                            "exponents": [0, 0],
                                        },
                                    ]
                                },
                            },
                        ],
                        "sign_conditions": [],
                    },
                    "samples": [],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
