"""Number field operation declarations."""
from collections.abc import Callable
from jacobian.contracts.base import ContractModel
from jacobian.contracts.number_field import (
    NumberFieldRequest, NumberFieldDiscriminantResult, NumberFieldRingOfIntegersResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.number_field.operations import (
    compute_nf_discriminant, compute_nf_ring_of_integers,
)
from jacobian.math_tools import MathTool

def nf_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id, title, description, request_model, result_model, operation, *tags,
    examples=(), version="1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(operation_id=operation_id, version=version, title=title,
        description=description, request_type=request_model, result_type=result_model,
        run=operation, tags=tags, examples=examples)

NUMBER_FIELD_OPERATIONS = (
    nf_operation(
        "number_field.discriminant.compute", "Compute the discriminant of a number field",
        "Compute the discriminant of a number field defined by one irreducible polynomial using SymPy.",
        NumberFieldRequest, NumberFieldDiscriminantResult, compute_nf_discriminant,
        "number-field", "discriminant", "exact",
        examples=(example("quadratic_disc", "Discriminant of x^2-2.",
            {"coefficients_descending": ["1", "0", "-2"], "variable": "x"}),),
    ),
)
