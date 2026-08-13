"""Polynomial operation declarations."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains.polynomial.operations import PolynomialOutputBudgetError
from jacobian.operation_declarations import (
    DurableOperationFactory,
    InlineOperationFactory,
    OperationAbortError,
    OperationDeclaration,
    OperationExample,
    OperationFailure,
    OperationRefusalError,
)

_POLYNOMIAL_FAILURE = OperationFailure(
    code="POLYNOMIAL_OPERATION_NOT_APPLICABLE",
    stage="polynomial_computation",
    hint="Check the declared ring, variable, and operation budgets.",
    exceptions=(TypeError, ValueError),
)
_polynomial_operation_factory = InlineOperationFactory(_POLYNOMIAL_FAILURE)
_materialized_polynomial_operation_factory = DurableOperationFactory(
    _POLYNOMIAL_FAILURE
)


def _polynomial_error() -> type[Exception]:
    """Load SymPy's polynomial error class only while handling an invocation."""

    from sympy.polys.polyerrors import PolynomialError

    return cast(type[Exception], PolynomialError)


def polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "2",
) -> OperationDeclaration[RequestT, ResultT]:
    """Declare an exact polynomial operation with bounded-output failure semantics."""

    declared = _polynomial_operation_factory(
        operation_id,
        title,
        description,
        request_model,
        result_model,
        operation,
        *tags,
        examples=examples,
        version=version,
    )
    return replace(
        declared,
        execute=_with_polynomial_output_budget(declared.execute),
    )


def materialized_polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    resource_reason: str = "",
    version: str = "2",
) -> OperationDeclaration[RequestT, ResultT]:
    """Declare an exact polynomial operation with durable result lineage."""

    declared = _materialized_polynomial_operation_factory(
        operation_id,
        title,
        description,
        request_model,
        result_model,
        operation,
        *tags,
        examples=examples,
        resource_reason=resource_reason,
        version=version,
    )
    return replace(
        declared,
        execute=_with_polynomial_output_budget(declared.execute),
    )


def _with_polynomial_output_budget[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    implementation: Callable[[RequestT], ResultT],
) -> Callable[[RequestT], ResultT]:
    """Add polynomial-specific bounded-output handling to an operation."""

    def execute(request: RequestT) -> ResultT:
        try:
            return implementation(request)
        except PolynomialOutputBudgetError as error:
            raise OperationAbortError(
                ExecutionStatus.ERROR,
                OperationDiagnostic(
                    code="POLYNOMIAL_OUTPUT_LIMIT_EXCEEDED",
                    stage="polynomial_output_validation",
                    message=str(error),
                    hint=(
                        "Use a smaller polynomial input or an operation with an "
                        "explicitly larger output budget."
                    ),
                ),
            ) from error
        except Exception as error:
            if isinstance(error, _polynomial_error()):
                raise OperationRefusalError(
                    _polynomial_operation_factory.failure.diagnostic(error)
                ) from error
            raise

    return execute
