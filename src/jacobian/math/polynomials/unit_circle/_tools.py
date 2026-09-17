"""Public declarations for exact unit-circle polynomial operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.unit_circle._models import (
    FejerRieszFactorResult,
    HermitianLaurentPolynomial,
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
)
from jacobian.math.polynomials.unit_circle._root_profile import (
    UnitDiskProfile,
    UnitDiskProfileRequest,
    unit_disk_profile,
)
from jacobian.math.polynomials.unit_circle._sup_norm import (
    unit_circle_sup_norm_squared,
)
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    UnitCircleSupNormSquaredRequest,
    UnitCircleSupNormSquaredResult,
)
from jacobian.math.polynomials.unit_circle.operations import (
    real_symmetric_degree_one_fejer_riesz_factor,
    unit_circle_arc_energy,
)


def _run_sup_norm(
    request: UnitCircleSupNormSquaredRequest,
) -> UnitCircleSupNormSquaredResult:
    return unit_circle_sup_norm_squared(request.polynomial)


def _run_arc_energy(request: UnitCircleArcEnergyRequest) -> UnitCircleArcEnergyResult:
    return unit_circle_arc_energy(
        request.polynomial, request.start_turn, request.end_turn
    )


def _run_fejer(source: HermitianLaurentPolynomial) -> FejerRieszFactorResult:
    return real_symmetric_degree_one_fejer_riesz_factor(source)


POLYNOMIAL_ONE_PLUS_Z = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    },
    "start_turn": {"num": "-1", "den": "4"},
    "end_turn": {"num": "1", "den": "4"},
}


SUP_NORM_ONE_PLUS_Z = {
    "polynomial": {
        "terms": [
            {
                "coefficient": {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                "exponent": 1,
            },
            {
                "coefficient": {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                "exponent": 0,
            },
        ]
    }
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.root_location.unit_disk_profile.compute",
        title="Count exact polynomial roots relative to the unit disk",
        description=(
            "Count roots of a nonzero univariate rational polynomial strictly "
            "inside, on, and strictly outside the unit circle, with multiplicity. "
            "Includes repeated boundary roots and reciprocal common factors; "
            "nonzero constants have zero counts. Exact arithmetic admission uses "
            "compressed support degree and coefficient growth. Rational radius r "
            "is handled by supplying p(r*z)."
        ),
        request_type=UnitDiskProfileRequest,
        result_type=UnitDiskProfile,
        run=lambda request: unit_disk_profile(request.polynomial),
        tags=("polynomial", "roots", "unit-disk", "schur-stability", "exact"),
        examples=(
            OperationExample(
                name="reciprocal_pair",
                description="Roots 1/2 and 2 lie on opposite sides of the circle.",
                input={
                    "polynomial": {
                        "variables": ["z"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "2", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-5", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.unit_circle.arc_energy.compute",
        title="Compute exact unit-circle arc energy",
        description=(
            "Compute the exact normalized energy integral of a bounded rational "
            "polynomial on an oriented arc with unwrapped rational turns. The "
            "result is A+B/pi, with B in the standard real cyclotomic field "
            "fixed by the rational endpoint conductor."
        ),
        request_type=UnitCircleArcEnergyRequest,
        result_type=UnitCircleArcEnergyResult,
        run=_run_arc_energy,
        tags=("polynomial", "unit-circle", "integral", "exact"),
        examples=(
            OperationExample(
                name="one_plus_z_right_semicircle",
                description="Energy of 1+z on the right semicircle.",
                input=POLYNOMIAL_ONE_PLUS_Z,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.unit_circle.real_symmetric_degree_one_fejer_riesz_factor.compute",
        title="Decide exact degree-one scalar Fejer-Riesz factorization",
        description=(
            "For a real-symmetric rational Laurent polynomial supported on "
            "{-1,0,1}, return its normalized exact outer factor, the zero "
            "conclusion, or an exact cosine witness of negativity."
        ),
        request_type=HermitianLaurentPolynomial,
        result_type=FejerRieszFactorResult,
        run=_run_fejer,
        tags=(
            "polynomial",
            "unit-circle",
            "fejer-riesz",
            "degree-one",
            "exact",
        ),
        examples=(
            OperationExample(
                name="boundary_zero",
                description="The factor of 2-z-z^-1 is 1-z.",
                input={
                    "terms": [
                        {"exponent": -1, "coefficient": {"num": "-1", "den": "1"}},
                        {"exponent": 0, "coefficient": {"num": "2", "den": "1"}},
                        {"exponent": 1, "coefficient": {"num": "-1", "den": "1"}},
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.unit_circle.sup_norm_squared.compute",
        title="Compute a certified unit-circle supremum norm squared",
        description=(
            "For a bounded Gaussian-rational polynomial P(z), return a certified "
            "rational enclosure of max_{|z|=1} |P(z)|^2 together with the exact "
            "transformed numerator Q(t) and denominator exponent d, the exact "
            "derivative numerator whose real roots are the finite critical "
            "points, the complete critical-value comparison ledger, and the "
            "separate z=-1 endpoint. Also returns the exact maximum as an "
            "indexed real-algebraic root whenever its irreducible resultant "
            "factor fits the shared carrier (else None), a rational enclosure "
            "of max |P|, and reports a full-circle maximizing set when the "
            "modulus is constant. Exact arithmetic; no sampled grid, float "
            "maximum, or unverified upper bound is ever returned."
        ),
        request_type=UnitCircleSupNormSquaredRequest,
        result_type=UnitCircleSupNormSquaredResult,
        run=_run_sup_norm,
        tags=(
            "polynomial",
            "unit-circle",
            "sup-norm",
            "maximum-modulus",
            "littlewood",
            "exact",
        ),
        discovery_terms=(
            "unit circle sup norm",
            "maximum modulus on the unit circle",
            "Littlewood polynomial extremum",
        ),
        examples=(
            OperationExample(
                name="one_plus_z",
                description="max |1+z|^2 on the circle is 4, attained at z=1.",
                input=SUP_NORM_ONE_PLUS_Z,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
