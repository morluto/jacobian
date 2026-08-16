"""Multivariate polynomial operation declarations."""
from collections.abc import Callable
from jacobian.contracts.base import ContractModel
from jacobian.contracts.multivariate_polynomial import (
    MultivariateGCDRequest, MultivariateGCDResult,
    MultivariateResultantRequest, MultivariateResultantResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.multivariate_polynomial.operations import (
    compute_multivariate_gcd, compute_multivariate_resultant,
)
from jacobian.math_tools import MathTool

def mp_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id, title, description, request_model, result_model, operation, *tags,
    examples=(), version="1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(operation_id=operation_id, version=version, title=title,
        description=description, request_type=request_model, result_type=result_model,
        run=operation, tags=tags, examples=examples)

MULTIVARIATE_POLYNOMIAL_OPERATIONS = (
    mp_operation(
        "polynomial.multivariate.gcd.compute", "Compute multivariate polynomial GCD",
        "Compute the GCD of two multivariate polynomials over QQ using SymPy.",
        MultivariateGCDRequest, MultivariateGCDResult, compute_multivariate_gcd,
        "polynomial", "multivariate", "gcd", "exact",
        examples=(example("simple_gcd", "GCD of x**2-y**2 and x-y over x,y.",
            {"left": {"variables": ["x", "y"], "expression": "x**2 - y**2"},
             "right": {"variables": ["x", "y"], "expression": "x - y"}}),),
    ),
    mp_operation(
        "polynomial.multivariate.resultant.compute", "Compute multivariate resultant",
        "Compute the resultant of two multivariate polynomials w.r.t. one variable using SymPy.",
        MultivariateResultantRequest, MultivariateResultantResult, compute_multivariate_resultant,
        "polynomial", "multivariate", "resultant", "exact",
        examples=(example("simple_resultant", "Resultant of x**2 and x+1 w.r.t. x",
            {"left": {"variables": ["x"], "expression": "x**2"},
             "right": {"variables": ["x"], "expression": "x + 1"},
             "eliminate_variable": "x"}),),
    ),
)
