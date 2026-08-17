"""Extended coding theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.coding_theory_extended import (
    DualCodeRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ShortenRequest,
    ShortenResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.coding_theory_extended.operations import (
    compute_dual_code,
    compute_puncture,
    compute_shorten,
)
from jacobian.math_tools import MathTool


def ct_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


CODING_THEORY_EXTENDED_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    ct_operation(
        "code.dual_code.compute",
        "Compute the dual code parity-check matrix",
        "Compute the exact parity-check matrix H of a linear code from "
        "its generator matrix G over a prime field, such that GH^T = 0.",
        DualCodeRequest,
        ParityCheckResult,
        compute_dual_code,
        "code",
        "dual-code",
        "exact",
        examples=(
            example(
                "repetition_code",
                "Dual of the binary repetition code.",
                {
                    "code": {
                        "field_order": 2,
                        "generator_matrix": [[1, 1, 1]],
                    },
                },
            ),
        ),
    ),
    ct_operation(
        "code.puncture.compute",
        "Puncture a linear code by deleting one coordinate",
        "Compute the punctured code by deleting one coordinate position "
        "from the generator matrix.",
        PunctureRequest,
        PunctureResult,
        compute_puncture,
        "code",
        "puncture",
        "exact",
        examples=(
            example(
                "repetition_code",
                "Puncture the binary repetition code at position 1.",
                {
                    "code": {
                        "field_order": 2,
                        "generator_matrix": [[1, 1, 1]],
                    },
                    "position": 1,
                },
            ),
        ),
    ),
    ct_operation(
        "code.shorten.compute",
        "Shorten a linear code at one coordinate",
        "Shorten a linear code by fixing one coordinate to a value and "
        "then deleting that coordinate from the generator matrix.",
        ShortenRequest,
        ShortenResult,
        compute_shorten,
        "code",
        "shorten",
        "exact",
        examples=(
            example(
                "repetition_code",
                "Shorten the binary repetition code at position 0, value 0.",
                {
                    "code": {
                        "field_order": 2,
                        "generator_matrix": [[1, 1, 1]],
                    },
                    "position": 0,
                    "value": 0,
                },
            ),
        ),
    ),
)
