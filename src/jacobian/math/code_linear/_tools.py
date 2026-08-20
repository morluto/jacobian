"""Code linear operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_linear._models import (
    CodeEqualRequest,
    CodeEqualResult,
    CodewordCheckRequest,
    CodewordCheckResult,
    DualCodeResult,
    FromGeneratorResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    ParityCheckRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ShortenRequest,
    ShortenResult,
    SyndromeRequest,
    SyndromeResult,
)
from jacobian.math.code_linear._operations import (
    compute_code_equal,
    compute_codeword_check,
    compute_dual_code,
    compute_from_generator,
    compute_macwilliams_transform,
    compute_parity_check,
    compute_puncture,
    compute_shorten,
    compute_syndrome,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "code.linear.from_generator.compute",
        "Canonicalize a linear code from a generator matrix",
        "Reduce a generator matrix over a bounded prime field to canonical "
        "full-row-rank form and report dimension, length, and cardinality.",
        GeneratorMatrixRequest,
        FromGeneratorResult,
        compute_from_generator,
        "code",
        "linear",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Canonicalize the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    _op(
        "code.linear.dual.compute",
        "Compute the dual code of a linear code",
        "Compute the exact dual code C^perp as a generator matrix, returning "
        "dual dimension and length.",
        GeneratorMatrixRequest,
        DualCodeResult,
        compute_dual_code,
        "code",
        "dual",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Dual of the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    _op(
        "code.linear.parity_check.compute",
        "Compute a parity-check matrix for a linear code",
        "Return one canonical parity-check matrix for C, with dimension and "
        "rank relation k = n - rank(H).",
        ParityCheckRequest,
        ParityCheckResult,
        compute_parity_check,
        "code",
        "parity-check",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Parity-check of the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    _op(
        "code.linear.codeword.check",
        "Check whether a word is a codeword",
        "Check whether a word lies in the row space of the generator and "
        "return membership, Hamming weight, and syndrome.",
        CodewordCheckRequest,
        CodewordCheckResult,
        compute_codeword_check,
        "code",
        "codeword",
        "exact",
        examples=(
            example(
                "member_word",
                "Check [1,1] against generator [1,1] over F_2.",
                {
                    "field_order": 2,
                    "generator_matrix": [[1, 1]],
                    "word": [1, 1],
                },
            ),
        ),
    ),
    _op(
        "code.linear.syndrome.compute",
        "Compute the syndrome of a word under a parity-check",
        "Return the exact syndrome Hw^T over the prime field, and whether "
        "the word is a member of the code.",
        SyndromeRequest,
        SyndromeResult,
        compute_syndrome,
        "code",
        "syndrome",
        "exact",
        examples=(
            example(
                "binary_repetition_syndrome",
                "Syndrome of [1,0] under parity-check [1,1] over F_2.",
                {
                    "field_order": 2,
                    "parity_check_matrix": [[1, 1]],
                    "word": [1, 0],
                },
            ),
        ),
    ),
    _op(
        "code.linear.equal.decide",
        "Decide whether two generator matrices define the same code",
        "Check exact mutual row-space containment; return equality or a "
        "concrete codeword witnessing the difference.",
        CodeEqualRequest,
        CodeEqualResult,
        compute_code_equal,
        "code",
        "equality",
        "exact",
        examples=(
            example(
                "equal_row_equivalent",
                "Two row-equivalent matrices define the same code.",
                {
                    "field_order": 2,
                    "generator_matrix_a": [[1, 1]],
                    "generator_matrix_b": [[1, 1]],
                },
            ),
        ),
    ),
    _op(
        "code.linear.macwilliams_transform.compute",
        "MacWilliams transform of a weight distribution",
        "Apply the q-ary MacWilliams identity to compute the dual code weight "
        "distribution from the primal weight enumerator.",
        MacWilliamsRequest,
        MacWilliamsResult,
        compute_macwilliams_transform,
        "code",
        "macwilliams",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "MacWilliams transform of the binary length-2 repetition code.",
                {
                    "field_order": 2,
                    "code_cardinality": 2,
                    "length": 2,
                    "weights": [1, 0, 1],
                },
            ),
        ),
    ),
    _op(
        "code.linear.puncture.compute",
        "Puncture a linear code at one coordinate",
        "Delete one coordinate from the generator matrix and return the "
        "punctured code with canonical generator basis.",
        PunctureRequest,
        PunctureResult,
        compute_puncture,
        "code",
        "puncture",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Puncture the binary length-2 repetition code at coordinate 0.",
                {
                    "field_order": 2,
                    "generator_matrix": [[1, 1]],
                    "coordinate": 0,
                },
            ),
        ),
    ),
    _op(
        "code.linear.shorten.compute",
        "Shorten a linear code at one coordinate",
        "Shorten a code by fixing one coordinate to zero and then puncturing "
        "it, returning the shortened code with canonical generator basis.",
        ShortenRequest,
        ShortenResult,
        compute_shorten,
        "code",
        "shorten",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Shorten the binary length-3 repetition code at coordinate 0.",
                {
                    "field_order": 2,
                    "generator_matrix": [[1, 1, 1]],
                    "coordinate": 0,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
