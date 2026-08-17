"""Combinatorics on words operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.words._models import (
    ConjugatesRequest,
    ConjugatesResult,
    FactorOccurrencesRequest,
    FactorOccurrencesResult,
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    MorphismApplyRequest,
    MorphismApplyResult,
    MorphismComposeRequest,
    MorphismComposeResult,
    ParikhRequest,
    ParikhResult,
    PeriodsRequest,
    PeriodsResult,
    PrefixFunctionRequest,
    PrefixFunctionResult,
    PrimitiveRootRequest,
    PrimitiveRootResult,
)
from jacobian.math.words._operations import (
    apply_morphism,
    compose_morphisms,
    compute_conjugates,
    compute_factor_occurrences,
    compute_factors_length,
    compute_incidence_matrix,
    compute_parikh_vector,
    compute_periods,
    compute_prefix_function,
    compute_primitive_root,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


WORDS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "word.factors.length.compute",
        "Compute distinct factors of a given length",
        "Given a finite word and 0 <= n <= |w|, return the complete distinct "
        "factor set of length n with occurrence positions and multiplicities.",
        FactorsLengthRequest,
        FactorsLengthResult,
        compute_factors_length,
        "combinatorics",
        "words",
        "factors",
        "exact",
        examples=(
            example(
                "abaab_factors_2",
                "Distinct length-2 factors of abaab; "
                "the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["a", "b", "a", "a", "b"], "factor_length": 2},
            ),
        ),
    ),
    _op(
        "word.factor_occurrences.compute",
        "Find all occurrences of a pattern",
        "Find every start index where a pattern occurs in a word, "
        "including overlapping occurrences.",
        FactorOccurrencesRequest,
        FactorOccurrencesResult,
        compute_factor_occurrences,
        "combinatorics",
        "words",
        "pattern-matching",
        "exact",
        examples=(
            example(
                "aba_occurrences",
                "All occurrences of 'ab' in 'abaabab'; "
                "both words must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["a", "b", "a", "a", "b", "a", "b"], "pattern": ["a", "b"]},
            ),
        ),
    ),
    _op(
        "word.periods.compute",
        "Compute all periods of a word",
        "Return every period p of a finite word, the least period, and "
        "whether the word is primitive (least period equals length).",
        PeriodsRequest,
        PeriodsResult,
        compute_periods,
        "combinatorics",
        "words",
        "periods",
        "exact",
        examples=(
            example(
                "ababab_periods",
                "Periods of 'ababab'; the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["a", "b", "a", "b", "a", "b"]},
            ),
        ),
    ),
    _op(
        "word.primitive_root.compute",
        "Compute primitive root of a word",
        "Find the unique primitive word u and exponent k with w = u^k, "
        "using the empty-word convention.",
        PrimitiveRootRequest,
        PrimitiveRootResult,
        compute_primitive_root,
        "combinatorics",
        "words",
        "primitive-root",
        "exact",
        examples=(
            example(
                "abcabc_root",
                "Primitive root of 'abcabc'; the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b", "c"], "word": ["a", "b", "c", "a", "b", "c"]},
            ),
        ),
    ),
    _op(
        "word.conjugates.compute",
        "Compute all cyclic conjugates",
        "Return all cyclic rotations of a finite word and the least "
        "lexicographic conjugate under alphabet order.",
        ConjugatesRequest,
        ConjugatesResult,
        compute_conjugates,
        "combinatorics",
        "words",
        "conjugates",
        "exact",
        examples=(
            example(
                "baab_conjugates",
                "All conjugates of 'baab'; the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["b", "a", "a", "b"]},
            ),
        ),
    ),
    _op(
        "word.parikh_vector.compute",
        "Compute Parikh vector of a word",
        "Return the exact count per alphabet symbol (Parikh vector), "
        "total length, and support set.",
        ParikhRequest,
        ParikhResult,
        compute_parikh_vector,
        "combinatorics",
        "words",
        "parikh",
        "exact",
        examples=(
            example(
                "abaab_parikh",
                "Parikh vector of 'abaab'; the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["a", "b", "a", "a", "b"]},
            ),
        ),
    ),
    _op(
        "word.prefix_function.compute",
        "Compute KMP prefix function",
        "Return the Knuth-Morris-Pratt prefix function (border table) of "
        "a finite word.",
        PrefixFunctionRequest,
        PrefixFunctionResult,
        compute_prefix_function,
        "combinatorics",
        "words",
        "prefix-function",
        "exact",
        examples=(
            example(
                "aabaab_prefix",
                "KMP prefix function of 'aabaab'; the word must use only declared alphabet symbols.",
                {"alphabet": ["a", "b"], "word": ["a", "a", "b", "a", "a", "b"]},
            ),
        ),
    ),
    _op(
        "word_morphism.apply.compute",
        "Apply a word morphism to a word",
        "Apply a finite word morphism (monoid homomorphism) to a source "
        "word, returning the concatenated image.",
        MorphismApplyRequest,
        MorphismApplyResult,
        apply_morphism,
        "combinatorics",
        "words",
        "morphism",
        "exact",
        examples=(
            example(
                "fibonacci_apply",
                "Apply Fibonacci morphism a->ab, b->a to the word 'ab'; "
                "images must use only target alphabet symbols.",
                {
                    "source_alphabet": ["a", "b"],
                    "target_alphabet": ["a", "b"],
                    "images": [["a", "b"], ["a"]],
                    "word": ["a", "b"],
                },
            ),
        ),
    ),
    _op(
        "word_morphism.compose.compute",
        "Compose two word morphisms",
        "Compose morphisms sigma: A* -> B* and tau: B* -> C*, returning "
        "tau ∘ sigma.",
        MorphismComposeRequest,
        MorphismComposeResult,
        compose_morphisms,
        "combinatorics",
        "words",
        "morphism",
        "exact",
        examples=(
            example(
                "compose_identity",
                "Compose two morphisms over {a,b}; images must use only "
                "declared alphabet symbols.",
                {
                    "source_alphabet": ["a", "b"],
                    "middle_alphabet": ["a", "b"],
                    "target_alphabet": ["a", "b"],
                    "sigma_images": [["a", "b"], ["b"]],
                    "tau_images": [["b"], ["a", "a"]],
                },
            ),
        ),
    ),
    _op(
        "word_morphism.incidence_matrix.compute",
        "Compute incidence matrix of a morphism",
        "Compute the exact nonnegative integer matrix M where M[b,a] counts "
        "occurrences of target letter b in the image of source letter a.",
        IncidenceMatrixRequest,
        IncidenceMatrixResult,
        compute_incidence_matrix,
        "combinatorics",
        "words",
        "morphism",
        "matrix",
        "exact",
        examples=(
            example(
                "fibonacci_matrix",
                "Incidence matrix of Fibonacci morphism a->ab, b->a; "
                "images must use only target alphabet symbols.",
                {
                    "source_alphabet": ["a", "b"],
                    "target_alphabet": ["a", "b"],
                    "images": [["a", "b"], ["a"]],
                },
            ),
        ),
    ),
)


TOOLS = WORDS_OPERATIONS

__all__ = ["TOOLS"]
