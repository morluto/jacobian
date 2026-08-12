"""Polynomial operation declarations."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains.polynomial.operations import PolynomialOutputBudgetError
from jacobian.operation_bindings import (
    DurableOperationFactory,
    InlineOperationFactory,
    InstalledOperation,
)
from jacobian.operations import (
    OperationAbortError,
    OperationFailure,
    OperationRefusalError,
)

_polynomial_operation_factory = InlineOperationFactory(
    OperationFailure(
        code="POLYNOMIAL_OPERATION_NOT_APPLICABLE",
        stage="polynomial_computation",
        hint="Check the declared ring, variable, and operation budgets.",
        exceptions=(TypeError, ValueError),
    )
)
_materialized_polynomial_operation_factory = DurableOperationFactory(
    _polynomial_operation_factory.failure
)


def _polynomial_error() -> type[Exception]:
    """Load SymPy's polynomial error class only while handling an invocation."""

    from sympy.polys.polyerrors import PolynomialError

    return cast(type[Exception], PolynomialError)


def polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    capability_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
    version: str = "2",
) -> InstalledOperation[RequestT, ResultT]:
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
        version=version,
    )
    return replace(
        declared,
        spec=replace(
            declared.spec,
            execute=_with_polynomial_output_budget(declared.spec.execute),
        ),
    )


def materialized_polynomial_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    capability_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
    resource_reason: str = "",
    version: str = "2",
) -> InstalledOperation[RequestT, ResultT]:
    """Declare an exact polynomial operation with durable result lineage."""

    declared = _materialized_polynomial_operation_factory(
        capability_id,
        title,
        description,
        request_model,
        result_model,
        operation,
        *tags,
        invocation_examples=invocation_examples,
        resource_reason=resource_reason,
        version=version,
    )
    return replace(
        declared,
        spec=replace(
            declared.spec,
            execute=_with_polynomial_output_budget(declared.spec.execute),
        ),
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
                CapabilityDiagnostic(
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
