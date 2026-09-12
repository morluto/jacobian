"""Published bounded graded-quotient operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.graded._models import (
    HilbertDimensionResult,
    HilbertFunctionRequest,
    HilbertFunctionResult,
    HilbertMultiplicityResult,
    HilbertPolynomialResult,
    HilbertSeriesRequest,
    HilbertSeriesResult,
    HVectorResult,
    InitialMonomialIdealRequest,
    InitialMonomialIdealResult,
    StandardMonomialsRequest,
    StandardMonomialsResult,
)
from jacobian.math.polynomials.graded.operations import (
    h_vector,
    hilbert_dimension,
    hilbert_function,
    hilbert_multiplicity,
    hilbert_polynomial,
    hilbert_series,
    initial_monomial_ideal,
    standard_monomials,
)

_EXAMPLE_IDEAL = {
    "variables": ["x", "y"],
    "generators": [
        {
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2, 0],
                    }
                ]
            },
        }
    ],
}


TOOLS = (
    MathTool(
        operation_id="graded_quotient.hilbert_series.compute",
        title="Compute a bounded exact Hilbert series",
        description="Return the ambient, canonical reduced, and (1-t)^d h-numerator forms, denominator dimension, and a finite coefficient prefix.",
        request_type=HilbertSeriesRequest,
        result_type=HilbertSeriesResult,
        run=lambda request: hilbert_series(
            request.ideal,
            request.monomial_order,
            request.prefix_degree,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "hilbert-series", "exact"),
        examples=(
            OperationExample(
                name="hypersurface",
                description="Compute the series of QQ[x,y]/(x^2).",
                input={"ideal": _EXAMPLE_IDEAL, "prefix_degree": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="graded_quotient.hilbert_polynomial.compute",
        title="Compute a bounded Hilbert polynomial",
        description="Return the exact eventual Hilbert polynomial and derived stabilization degree from a standard-graded quotient series.",
        request_type=HilbertSeriesRequest,
        result_type=HilbertPolynomialResult,
        run=lambda request: hilbert_polynomial(
            request.ideal,
            request.monomial_order,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "hilbert-polynomial", "exact"),
        examples=(
            OperationExample(
                name="line",
                description="Compute the eventual Hilbert polynomial of QQ[x,y]/(x^2).",
                input={"ideal": _EXAMPLE_IDEAL},
            ),
        ),
    ),
    MathTool(
        operation_id="graded_quotient.dimension.compute",
        title="Compute graded quotient dimension",
        description="Return the Krull dimension as the reduced Hilbert-series denominator exponent.",
        request_type=HilbertSeriesRequest,
        result_type=HilbertDimensionResult,
        run=lambda request: hilbert_dimension(
            request.ideal,
            request.monomial_order,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "dimension", "exact"),
        examples=(
            OperationExample(
                name="line",
                description="Compute the dimension of QQ[x,y]/(x^2).",
                input={"ideal": _EXAMPLE_IDEAL},
            ),
        ),
    ),
    MathTool(
        operation_id="graded_quotient.multiplicity.compute",
        title="Compute graded quotient multiplicity",
        description="Return the exact multiplicity from the reduced Hilbert numerator, including zero-dimensional length.",
        request_type=HilbertSeriesRequest,
        result_type=HilbertMultiplicityResult,
        run=lambda request: hilbert_multiplicity(
            request.ideal,
            request.monomial_order,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "multiplicity", "exact"),
        examples=(
            OperationExample(
                name="line",
                description="Compute multiplicity of QQ[x,y]/(x^2).",
                input={"ideal": _EXAMPLE_IDEAL},
            ),
        ),
    ),
    MathTool(
        operation_id="graded_quotient.h_vector.compute",
        title="Compute the reduced Hilbert h-vector",
        description="Return the exact integer coefficients of the reduced numerator under the (1-t)^d convention.",
        request_type=HilbertSeriesRequest,
        result_type=HVectorResult,
        run=lambda request: h_vector(
            request.ideal,
            request.monomial_order,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "h-vector", "exact"),
        examples=(
            OperationExample(
                name="line",
                description="Compute the h-vector of QQ[x,y]/(x^2).",
                input={"ideal": _EXAMPLE_IDEAL},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.ideal.initial_monomial_ideal.compute",
        title="Compute a bounded initial monomial ideal",
        description="Compute the reduced Gröbner basis of a homogeneous QQ ideal and project its leading monomials under the selected order.",
        request_type=InitialMonomialIdealRequest,
        result_type=InitialMonomialIdealResult,
        run=lambda request: initial_monomial_ideal(
            request.ideal,
            request.monomial_order,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "ideal", "initial-ideal", "groebner", "exact"),
        examples=(
            OperationExample(
                name="principal_square",
                description="Compute the initial ideal of (x^2).",
                input={"ideal": _EXAMPLE_IDEAL},
            ),
        ),
    ),
    MathTool(
        operation_id="monomial_ideal.standard_monomials.degree.compute",
        title="Enumerate standard monomials in one degree",
        description="Return every degree-m monomial not divisible by a generator of a bounded unit monomial ideal.",
        request_type=StandardMonomialsRequest,
        result_type=StandardMonomialsResult,
        run=lambda request: standard_monomials(request.initial_ideal, request.degree),
        tags=("polynomial", "monomial-ideal", "standard-monomial", "exact"),
        examples=(
            OperationExample(
                name="degree_two",
                description="Enumerate degree-two standard monomials modulo (x^2).",
                input={"initial_ideal": _EXAMPLE_IDEAL, "degree": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="graded_quotient.hilbert_function.compute",
        title="Compute a bounded Hilbert-function prefix",
        description="Compute exact graded quotient dimensions through a finite degree prefix using the initial monomial ideal.",
        request_type=HilbertFunctionRequest,
        result_type=HilbertFunctionResult,
        run=lambda request: hilbert_function(
            request.ideal,
            request.monomial_order,
            request.max_degree,
            resource_budget=request.resource_budget,
        ),
        tags=("polynomial", "graded", "hilbert-function", "exact"),
        examples=(
            OperationExample(
                name="prefix",
                description="Compute the Hilbert-function prefix through degree two for (x^2).",
                input={"ideal": _EXAMPLE_IDEAL, "max_degree": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
