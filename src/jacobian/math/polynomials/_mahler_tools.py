"""Integer-polynomial profile and Mahler-measure operation declarations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials import _mahler_kernel as native
from jacobian.math.polynomials._mahler_models import (
    ContentPrimitiveProfileRequest,
    ContentPrimitiveProfileResult,
    MahlerMeasureRequest,
    MahlerMeasureResult,
    RealQuadraticRootProfileRequest,
    RealQuadraticRootProfileResult,
)

INTEGER_POLYNOMIAL_PROFILE_OPERATIONS = (
    MathTool(
        operation_id="polynomial.integer.content_primitive_profile.compute",
        title="Compute the sign, content, and primitive part of an integer polynomial",
        description=(
            "Return the sign, the nonnegative coefficient gcd, the "
            "positive-leading primitive part, the degree, and the exact "
            "reconstruction of one canonical integer polynomial. This makes the "
            "content/leading-coefficient boundary explicit before factor-level "
            "normalization."
        ),
        request_type=ContentPrimitiveProfileRequest,
        result_type=ContentPrimitiveProfileResult,
        run=native.content_primitive_profile,
        tags=("polynomial", "integer", "content", "primitive", "exact"),
        examples=(
            OperationExample(
                name="scaled_quadratic",
                description="6x^2-6 has content 6 and primitive part x^2-1.",
                input={"polynomial": {"coefficients": ["6", "0", "-6"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.quadratic.real_root_profile.compute",
        title="Compute the exact real-root profile of a quadratic",
        description=(
            "For one quadratic a x^2 + b x + c with nonzero leading coefficient, "
            "return the exact discriminant, root kind, sums and products, the "
            "distinct real roots as quadratic surds or the squared modulus of a "
            "complex conjugate pair, and exact unit-disk locations. Coefficients "
            "have at most 256 digits. Nonsquare discriminants are retained after "
            "content normalization; the published roots are primitive "
            "RealAlgebraicValue polynomials and do not require square-free "
            "factorization."
        ),
        request_type=RealQuadraticRootProfileRequest,
        result_type=RealQuadraticRootProfileResult,
        run=native.quadratic_root_profile,
        tags=("polynomial", "quadratic", "roots", "exact"),
        examples=(
            OperationExample(
                name="golden_quadratic",
                description="x^2-x-1 has one root inside and one outside the unit disk.",
                input={"polynomial": {"coefficients": ["1", "-1", "-1"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.mahler_measure.compute",
        title="Compute the exact Mahler measure of a bounded polynomial",
        description=(
            "Return the exact Mahler measure |a_d| * prod_i max(1, |alpha_i|) of "
            "one canonical integer polynomial of degree zero through two, together "
            "with the complete root-location ledger. The leading coefficient is "
            "never dropped, and an unresolved root location is a typed refusal "
            "rather than a silently ignored root. Coefficients have at most 256 "
            "digits. Nonsquare discriminants are retained after content "
            "normalization without a separate square-free factorization bound."
        ),
        request_type=MahlerMeasureRequest,
        result_type=MahlerMeasureResult,
        run=native.mahler_measure,
        tags=("polynomial", "mahler", "measure", "exact"),
        examples=(
            OperationExample(
                name="golden_measure",
                description="The Mahler measure of x^2-x-1 is the golden ratio.",
                input={"polynomial": {"coefficients": ["1", "-1", "-1"]}},
            ),
        ),
    ),
)
