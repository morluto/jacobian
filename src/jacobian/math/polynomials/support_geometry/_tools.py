"""Polynomial support geometry operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.polynomials.support_geometry._models import (
    InitialFormRequest,
    NewtonPolytopeRequest,
    SupportRequest,
    WeightProfileRequest,
)
from jacobian.math.polynomials.support_geometry.operations import (
    compute_initial_form,
    compute_newton_polytope,
    compute_support,
    compute_weight_profile,
)
from jacobian.math.polynomials.support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)

# Canonical x^2 + xy + y^2 over the ordered ring QQ[x, y].
_TOY_POLYNOMIAL = {
    "variables": ["x", "y"],
    "polynomial": {
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [2, 0]},
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [1, 1]},
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 2]},
        ]
    },
}

_TOY_VARS = ("x", "y")


TOOLS: MathTools = (
    MathTool(
        operation_id="polynomial.support.compute",
        title="Compute polynomial exponent support",
        description=(
            "Extract the exponent support of an exact sparse polynomial: "
            "exponent set, term count, coordinatewise min/max, and total degree profile."
        ),
        request_type=SupportRequest,
        result_type=PolynomialSupport,
        run=compute_support,
        tags=("polynomial", "support", "exact"),
        examples=(
            example(
                "xy_squared_support",
                (
                    "Compute support of x^2 + xy + y^2; the polynomial must be a "
                    "canonical RationalPolynomial with ordered variables and unique "
                    "nonnegative exponents (the zero polynomial is allowed and yields an empty support)."
                ),
                {"polynomial": dict(_TOY_POLYNOMIAL)},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.newton_polytope.compute",
        title="Compute Newton polytope",
        description=(
            "Compute the Newton polytope (convex hull of support exponents) of "
            "a polynomial with at most 96 terms, classifying every support "
            "point exactly as a vertex or contained in the hull of the rest."
        ),
        request_type=NewtonPolytopeRequest,
        result_type=NewtonPolytope,
        run=compute_newton_polytope,
        tags=("polynomial", "newton-polytope", "exact"),
        examples=(
            example(
                "xy_squared_newton",
                (
                    "Compute Newton polytope of x^2 + xy + y^2; the polynomial must be a "
                    "canonical RationalPolynomial with at most 96 terms so the per-point "
                    "exact extremality test stays bounded."
                ),
                {"polynomial": dict(_TOY_POLYNOMIAL)},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.weight_profile.compute",
        title="Compute weight profile",
        description=(
            "Compute the weight profile of a polynomial's support under an integer "
            "weight vector: minimum weight, minimizing exponents, and weight layers."
        ),
        request_type=WeightProfileRequest,
        result_type=PolynomialWeightProfile,
        run=compute_weight_profile,
        tags=("polynomial", "weight-profile", "exact"),
        examples=(
            example(
                "weight_profile_xy",
                (
                    "Weight profile of x^2 + xy + y^2 under w=(1,1); the polynomial must be "
                    "nonzero with at most 1024 terms and coefficient components at most "
                    "512 digits, and the weight length must match the variable count with "
                    "each component bounded by 2^31."
                ),
                {
                    "polynomial": dict(_TOY_POLYNOMIAL),
                    "weight": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.initial_form.compute",
        title="Compute initial form",
        description=(
            "Compute the initial form of a polynomial under a weight vector: "
            "the sum of all terms with minimum weight."
        ),
        request_type=InitialFormRequest,
        result_type=PolynomialFaceData,
        run=compute_initial_form,
        tags=("polynomial", "initial-form", "exact"),
        examples=(
            example(
                "initial_form_xy",
                (
                    "Initial form of x^2 + xy + y^2 under w=(1,2): weights are "
                    "2,3,4 so initial form is x^2; source must be nonzero with "
                    "at most 1024 terms and 512-digit coefficients, weight length "
                    "must match variable count (each bounded by 2^31)."
                ),
                {
                    "polynomial": dict(_TOY_POLYNOMIAL),
                    "weight": [1, 2],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
