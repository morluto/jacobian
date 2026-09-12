"""Published bounded graded-quotient operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.graded._models import (
    HilbertFunctionRequest,
    HilbertFunctionResult,
    InitialMonomialIdealRequest,
    InitialMonomialIdealResult,
    StandardMonomialsRequest,
    StandardMonomialsResult,
)
from jacobian.math.polynomials.graded.operations import (
    hilbert_function,
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
        operation_id="polynomial.ideal.initial_monomial_ideal.compute",
        title="Compute a bounded initial monomial ideal",
        description="Compute the reduced Gröbner basis of a homogeneous QQ ideal and project its leading monomials under the selected order.",
        request_type=InitialMonomialIdealRequest,
        result_type=InitialMonomialIdealResult,
        run=lambda request: initial_monomial_ideal(
            request.ideal, request.monomial_order, resource_budget=request.resource_budget
        ),
        tags=("polynomial", "ideal", "initial-ideal", "groebner", "exact"),
        examples=(OperationExample(name="principal_square", description="Compute the initial ideal of (x^2).", input={"ideal": _EXAMPLE_IDEAL}),),
    ),
    MathTool(
        operation_id="monomial_ideal.standard_monomials.degree.compute",
        title="Enumerate standard monomials in one degree",
        description="Return every degree-m monomial not divisible by a generator of a bounded unit monomial ideal.",
        request_type=StandardMonomialsRequest,
        result_type=StandardMonomialsResult,
        run=lambda request: standard_monomials(request.initial_ideal, request.degree),
        tags=("polynomial", "monomial-ideal", "standard-monomial", "exact"),
        examples=(OperationExample(name="degree_two", description="Enumerate degree-two standard monomials modulo (x^2).", input={"initial_ideal": _EXAMPLE_IDEAL, "degree": 2}),),
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
        examples=(OperationExample(name="prefix", description="Compute the Hilbert-function prefix through degree two for (x^2).", input={"ideal": _EXAMPLE_IDEAL, "max_degree": 2}),),
    ),
)

__all__ = ["TOOLS"]
