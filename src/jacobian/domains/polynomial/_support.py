"""Polynomial operation declarations."""

from collections.abc import Callable
from dataclasses import replace

from sympy.polys.polyerrors import PolynomialError

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains.polynomial.operations import PolynomialOutputBudgetError
from jacobian.operations import (
    ComputedOperation,
    ComputedOperationFactory,
    ComputedOutcome,
    OperationExecutionFailure,
    OperationFailure,
)

_polynomial_operation_factory = ComputedOperationFactory(
    OperationFailure(
        code="POLYNOMIAL_OPERATION_NOT_APPLICABLE",
        stage="polynomial_computation",
        hint="Check the declared ring, variable, and operation budgets.",
        exceptions=(PolynomialError, TypeError, ValueError),
    )
)


def polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    capability_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[ContractModel], ContractModel],
    *tags: str,
    invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
    relation_id: str | None = None,
) -> ComputedOperation[RequestT, ResultT]:
    """Declare an exact polynomial operation with bounded-output failure semantics."""

    declared = _polynomial_operation_factory(
        capability_id,
        title,
        description,
        request_model,
        result_model,
        operation,
        *tags,
        invocation_examples=invocation_examples,
        relation_id=relation_id,
    )
    implementation = declared.implementation

    def execute(request: RequestT) -> ComputedOutcome[ResultT]:
        try:
            return implementation(request)
        except PolynomialOutputBudgetError as error:
            return OperationExecutionFailure(
                status=ExecutionStatus.ERROR,
                diagnostic=CapabilityDiagnostic(
                    code="POLYNOMIAL_OUTPUT_LIMIT_EXCEEDED",
                    stage="polynomial_output_validation",
                    message=str(error),
                    hint=(
                        "Use a smaller polynomial input or an operation with an "
                        "explicitly larger output budget."
                    ),
                ),
            )

    return replace(declared, implementation=execute)
