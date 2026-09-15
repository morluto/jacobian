"""Public combinatorics-on-words operation declarations."""

from collections.abc import Callable
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.logic.languages.words._models import (
    FactorComplexityRequest,
    FactorComplexityResult,
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    MorphismApplyRequest,
    MorphismApplyResult,
    MorphismComposeRequest,
    MorphismComposeResult,
    MorphismImageLengthsRequest,
    MorphismImageLengthsResult,
    MorphismIterateRequest,
    MorphismIterateResult,
    MorphismPowerRequest,
    MorphismPowerResult,
    PeriodsRequest,
    PeriodsResult,
    RauzyGraphEdge,
    RauzyGraphRequest,
    RauzyGraphResult,
    SubstitutionDependencyGraphRequest,
    SubstitutionDependencyGraphResult,
    SubstitutionFactorComplexityRequest,
    SubstitutionFactorComplexityResult,
    SubstitutionFixedPointPrefixRequest,
    SubstitutionFixedPointPrefixResult,
    SubstitutionPrimitivityProfileRequest,
    SubstitutionPrimitivityProfileResult,
)
from jacobian.math.logic.languages.words.operations import (
    apply_morphism,
    compose_morphisms,
    factor_complexity_prefix,
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    iterate_morphism,
    morphism_image_lengths,
    morphism_power,
    periods,
    rauzy_graph,
    substitution_dependency_graph,
    substitution_primitivity_profile,
)


def _admit[T](
    admission: Callable[[], T],
    *,
    location: tuple[str | int, ...],
    code: str,
) -> T:
    try:
        return admission()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(exc),
        ) from exc


def compute_factors_length(request: FactorsLengthRequest) -> FactorsLengthResult:
    analysis = _admit(
        lambda: factors_of_length(request.word, request.factor_length),
        location=("factor_length",),
        code="word.factor_length_out_of_range",
    )
    return FactorsLengthResult._from_kernel(
        request, factors=analysis.factors, occurrences=analysis.occurrences
    )


def compute_periods(request: PeriodsRequest) -> PeriodsResult:
    analysis = periods(request.word)
    return PeriodsResult._from_kernel(
        request,
        periods=analysis.periods,
        least_period=analysis.least_period,
        is_primitive=analysis.primitive,
    )


def compute_incidence_matrix(
    request: IncidenceMatrixRequest,
) -> IncidenceMatrixResult:
    return IncidenceMatrixResult._from_kernel(
        request, incidence_matrix(request.morphism)
    )


def compute_substitution_dependency_graph(
    request: SubstitutionDependencyGraphRequest,
) -> SubstitutionDependencyGraphResult:
    graph = _admit(
        lambda: substitution_dependency_graph(request.substitution),
        location=("substitution",),
        code="words.substitution_dependency_graph_not_admitted",
    )
    return SubstitutionDependencyGraphResult._from_kernel(request, graph)


def compute_substitution_primitivity_profile(
    request: SubstitutionPrimitivityProfileRequest,
) -> SubstitutionPrimitivityProfileResult:
    analysis = _admit(
        lambda: substitution_primitivity_profile(request.dependency_graph),
        location=("dependency_graph",),
        code="words.substitution_primitivity_not_admitted",
    )
    return SubstitutionPrimitivityProfileResult._from_kernel(
        request,
        strongly_connected_components=analysis.strongly_connected_components,
        irreducible=analysis.irreducible,
        aperiodic=analysis.aperiodic,
        primitive=analysis.primitive,
        least_positive_power=analysis.least_positive_power,
        exponent_upper_bound=analysis.exponent_upper_bound,
        obstruction=analysis.obstruction,
    )


def compute_substitution_fixed_point_prefix(
    request: SubstitutionFixedPointPrefixRequest,
) -> SubstitutionFixedPointPrefixResult:
    analysis = _admit(
        lambda: fixed_point_prefix(request.source, request.prefix_length),
        location=("source", "prefix_length"),
        code="words.fixed_point_prefix_not_admitted",
    )
    return SubstitutionFixedPointPrefixResult._from_kernel(
        request,
        prefix=analysis.prefix,
        least_iterate_depth=analysis.least_iterate_depth,
        retained_prefix_lengths=analysis.retained_prefix_lengths,
    )


def compute_morphism_apply(request: MorphismApplyRequest) -> MorphismApplyResult:
    image = _admit(
        lambda: apply_morphism(request.morphism, request.word),
        location=("morphism", "word"),
        code="words.morphism_apply_not_admitted",
    )
    return MorphismApplyResult._from_kernel(request, image)


def compute_morphism_compose(request: MorphismComposeRequest) -> MorphismComposeResult:
    composite = _admit(
        lambda: compose_morphisms(request.first, request.second),
        location=("first", "second"),
        code="words.morphism_compose_not_admitted",
    )
    return MorphismComposeResult._from_kernel(request, composite)


def compute_morphism_power(request: MorphismPowerRequest) -> MorphismPowerResult:
    power = _admit(
        lambda: morphism_power(request.morphism, request.exponent),
        location=("morphism", "exponent"),
        code="words.morphism_power_not_admitted",
    )
    return MorphismPowerResult._from_kernel(request, power)


def compute_morphism_iterate(request: MorphismIterateRequest) -> MorphismIterateResult:
    image = _admit(
        lambda: iterate_morphism(request.morphism, request.word, request.steps),
        location=("morphism", "word", "steps"),
        code="words.morphism_iterate_not_admitted",
    )
    return MorphismIterateResult._from_kernel(request, image)


def compute_morphism_image_lengths(
    request: MorphismImageLengthsRequest,
) -> MorphismImageLengthsResult:
    lengths = morphism_image_lengths(request.morphism)
    return MorphismImageLengthsResult._from_kernel(request, lengths)


def compute_factor_complexity(
    request: FactorComplexityRequest,
) -> FactorComplexityResult:
    analysis = _admit(
        lambda: factor_complexity_prefix(request.word, request.max_order),
        location=("word", "max_order"),
        code="words.factor_complexity_not_admitted",
    )
    return FactorComplexityResult._from_kernel(
        request, complexity=analysis.complexity, families=analysis.families
    )


def compute_rauzy_graph(request: RauzyGraphRequest) -> RauzyGraphResult:
    analysis = _admit(
        lambda: rauzy_graph(request.word, request.order),
        location=("word", "order"),
        code="words.rauzy_graph_not_admitted",
    )
    edges = tuple(
        RauzyGraphEdge(
            source=source, label=label, target=target, occurrences=occurrences
        )
        for (source, label, target), occurrences in zip(
            analysis.edges, analysis.occurrences, strict=True
        )
    )
    return RauzyGraphResult._from_kernel(
        request, vertices=analysis.vertices, edges=edges
    )


def compute_substitution_factor_complexity(
    request: SubstitutionFactorComplexityRequest,
) -> SubstitutionFactorComplexityResult:
    prefix_analysis = _admit(
        lambda: fixed_point_prefix(request.source, request.prefix_length),
        location=("source", "prefix_length"),
        code="words.fixed_point_prefix_not_admitted",
    )
    complexity_analysis = _admit(
        lambda: factor_complexity_prefix(prefix_analysis.prefix, request.max_order),
        location=("source", "max_order"),
        code="words.factor_complexity_not_admitted",
    )
    return SubstitutionFactorComplexityResult._from_kernel(
        request,
        prefix=prefix_analysis.prefix,
        least_iterate_depth=prefix_analysis.least_iterate_depth,
        complexity=complexity_analysis.complexity,
        families=complexity_analysis.families,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="word.factors.length.compute",
        title="Compute all factors of one length",
        description=(
            "Enumerate every distinct contiguous factor of the requested length, "
            "in first-occurrence order, with all zero-based occurrence positions."
        ),
        request_type=FactorsLengthRequest,
        result_type=FactorsLengthResult,
        run=compute_factors_length,
        tags=("combinatorics", "words", "factors", "exact", "complete"),
        examples=(
            OperationExample(
                name="abaab_factors_2",
                description="Enumerate all length-two factors of abaab.",
                input={
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "a", "b"],
                    },
                    "factor_length": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.periods.compute",
        title="Compute all periods of a word",
        description=(
            "Return every positive overlap period and decide whether the word is "
            "a nontrivial integer power. An empty word has no positive periods and "
            "is not primitive."
        ),
        request_type=PeriodsRequest,
        result_type=PeriodsResult,
        run=compute_periods,
        tags=("combinatorics", "words", "periods", "exact", "complete"),
        examples=(
            OperationExample(
                name="ababab_periods",
                description="Compute the complete period profile of ababab.",
                input={
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "b", "a", "b"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.incidence_matrix.compute",
        title="Compute a word-morphism incidence matrix",
        description=(
            "Compute the exact matrix whose target-symbol rows and source-symbol "
            "columns count symbols in each morphism image."
        ),
        request_type=IncidenceMatrixRequest,
        result_type=IncidenceMatrixResult,
        run=compute_incidence_matrix,
        tags=("combinatorics", "words", "morphism", "matrix", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_matrix",
                description="Compute the incidence matrix of a->ab and b->a.",
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="substitution.dependency_graph.compute",
        title="Compute a substitution dependency graph",
        description=(
            "Return every letter dependency a→b, with all zero-based positions "
            "where b occurs in the image of a."
        ),
        request_type=SubstitutionDependencyGraphRequest,
        result_type=SubstitutionDependencyGraphResult,
        run=compute_substitution_dependency_graph,
        tags=("combinatorics", "words", "substitution", "graph", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_dependencies",
                description="Compute the dependency graph of the Fibonacci substitution.",
                input={
                    "substitution": {
                        "morphism": {
                            "source_alphabet": ["0", "1"],
                            "target_alphabet": ["0", "1"],
                            "images": [["0", "1"], ["0"]],
                        }
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="substitution.primitivity_profile.compute",
        title="Compute a substitution primitivity profile",
        description=(
            "Decide whether a substitution dependency graph is primitive, "
            "returning its least positive Boolean power or an exact graph obstruction."
        ),
        request_type=SubstitutionPrimitivityProfileRequest,
        result_type=SubstitutionPrimitivityProfileResult,
        run=compute_substitution_primitivity_profile,
        tags=(
            "combinatorics",
            "words",
            "substitution",
            "matrix",
            "primitivity",
            "exact",
        ),
        examples=(
            OperationExample(
                name="fibonacci_primitivity",
                description="Prove the Fibonacci substitution primitive from its dependency graph.",
                input={
                    "dependency_graph": {
                        "substitution": {
                            "morphism": {
                                "source_alphabet": ["0", "1"],
                                "target_alphabet": ["0", "1"],
                                "images": [["0", "1"], ["0"]],
                            }
                        },
                        "edges": [
                            {
                                "source": "0",
                                "target": "0",
                                "multiplicity": 1,
                                "positions": [0],
                            },
                            {
                                "source": "0",
                                "target": "1",
                                "multiplicity": 1,
                                "positions": [1],
                            },
                            {
                                "source": "1",
                                "target": "0",
                                "multiplicity": 1,
                                "positions": [0],
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="substitution.fixed_point_prefix.compute",
        title="Compute a substitution fixed-point prefix",
        description=(
            "Return the requested finite prefix of a certified prolongable growing "
            "substitution fixed point from the least sufficient iterate, after "
            "bounding source occurrences, prefix length, and generation work."
        ),
        request_type=SubstitutionFixedPointPrefixRequest,
        result_type=SubstitutionFixedPointPrefixResult,
        run=compute_substitution_fixed_point_prefix,
        tags=(
            "combinatorics",
            "words",
            "substitution",
            "fixed-point",
            "exact",
        ),
        examples=(
            OperationExample(
                name="fibonacci_prefix",
                description="Compute eight letters of the Fibonacci fixed point.",
                input={
                    "source": {
                        "substitution": {
                            "morphism": {
                                "source_alphabet": ["0", "1"],
                                "target_alphabet": ["0", "1"],
                                "images": [["0", "1"], ["0"]],
                            }
                        },
                        "seed": "0",
                    },
                    "prefix_length": 8,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.apply.compute",
        title="Apply a word morphism to a finite word",
        description=(
            "Apply one bounded morphism to a finite word over its source alphabet, "
            "returning the exact image over the target alphabet; the word alphabet "
            "must equal the morphism source alphabet and the output must fit the "
            "500-letter bound."
        ),
        request_type=MorphismApplyRequest,
        result_type=MorphismApplyResult,
        run=compute_morphism_apply,
        tags=("combinatorics", "words", "morphism", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_apply_ab",
                description=(
                    "Apply a->ab, b->a to the word ab yielding aba; the word must "
                    "use the morphism source alphabet."
                ),
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    },
                    "word": {"alphabet": ["a", "b"], "letters": ["a", "b"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.compose.compute",
        title="Compose two compatible word morphisms",
        description=(
            "Compose two bounded morphisms with matching middle alphabets, returning "
            "the composite with explicit outer axes; the first target must equal the "
            "second source and composed images must fit the output bound."
        ),
        request_type=MorphismComposeRequest,
        result_type=MorphismComposeResult,
        run=compute_morphism_compose,
        tags=("combinatorics", "words", "morphism", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_then_swap",
                description=(
                    "Compose a->ab, b->a with the a/b to x/y swap; the middle "
                    "alphabets must agree."
                ),
                input={
                    "first": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    },
                    "second": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["x", "y"],
                        "images": [["y"], ["x"]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.power.compute",
        title="Compute a bounded morphism power",
        description=(
            "Iterate one endomorphism a bounded number of times by composition, "
            "returning the power with explicit axes; the morphism must be an "
            "endomorphism, the exponent lies in 0..16, and power images must fit "
            "the image-length bound."
        ),
        request_type=MorphismPowerRequest,
        result_type=MorphismPowerResult,
        run=compute_morphism_power,
        tags=("combinatorics", "words", "morphism", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_square",
                description=(
                    "Square the Fibonacci morphism a->ab, b->a; the morphism must "
                    "be an endomorphism."
                ),
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    },
                    "exponent": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.iterate.compute",
        title="Iterate a morphism on a finite word",
        description=(
            "Apply one endomorphism to a finite word a bounded number of times, "
            "returning the exact stepped image; the word must use the morphism source "
            "alphabet and the expanded output must fit the 500-letter bound."
        ),
        request_type=MorphismIterateRequest,
        result_type=MorphismIterateResult,
        run=compute_morphism_iterate,
        tags=("combinatorics", "words", "morphism", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_iterate_twice",
                description=(
                    "Apply the Fibonacci morphism twice to a; the word must use "
                    "the endomorphism alphabet."
                ),
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    },
                    "word": {"alphabet": ["a", "b"], "letters": ["a"]},
                    "steps": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.image_lengths.compute",
        title="Compute morphism image lengths",
        description=(
            "Return one image length per source symbol in source order with total "
            "and maximum summaries; the lengths are read directly from the retained "
            "morphism images."
        ),
        request_type=MorphismImageLengthsRequest,
        result_type=MorphismImageLengthsResult,
        run=compute_morphism_image_lengths,
        tags=("combinatorics", "words", "morphism", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_lengths",
                description=(
                    "Image lengths of a->ab, b->a are (2, 1); the morphism supplies "
                    "its own source axis."
                ),
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.factor_complexity_prefix.compute",
        title="Compute a factor-complexity prefix",
        description=(
            "Return distinct-factor counts p(0)..p(max) with complete factor families "
            "of the supplied word only; no infinite-language limit is asserted and "
            "max_order must not exceed the word length."
        ),
        request_type=FactorComplexityRequest,
        result_type=FactorComplexityResult,
        run=compute_factor_complexity,
        tags=("combinatorics", "words", "factor-complexity", "exact", "complete"),
        examples=(
            OperationExample(
                name="abaab_complexity_2",
                description=(
                    "Complexity of abaab through order 2; the word supplies its own alphabet."
                ),
                input={
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "a", "b"],
                    },
                    "max_order": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.rauzy_graph.compute",
        title="Compute a Rauzy graph of a finite word",
        description=(
            "Return labeled vertices (order-factors) and edges (order+1 factors with "
            "prefix/suffix endpoints and occurrences) of the supplied word only."
        ),
        request_type=RauzyGraphRequest,
        result_type=RauzyGraphResult,
        run=compute_rauzy_graph,
        tags=("combinatorics", "words", "rauzy", "exact", "complete"),
        examples=(
            OperationExample(
                name="abaab_rauzy_1",
                description=(
                    "Rauzy graph of abaab at order 1; the word supplies its own alphabet."
                ),
                input={
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "a", "b"],
                    },
                    "order": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="substitution.factor_complexity_prefix.compute",
        title="Compute factor complexity of a fixed-point prefix",
        description=(
            "Compose a certified fixed-point prefix with factor-complexity enumeration, "
            "retaining the prefix and iterate depth; claims cover the retained prefix only."
        ),
        request_type=SubstitutionFactorComplexityRequest,
        result_type=SubstitutionFactorComplexityResult,
        run=compute_substitution_factor_complexity,
        tags=("combinatorics", "words", "substitution", "factor-complexity", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_complexity_prefix",
                description=(
                    "Complexity of eight Fibonacci fixed-point letters through order 3; "
                    "the source must be prolongable."
                ),
                input={
                    "source": {
                        "substitution": {
                            "morphism": {
                                "source_alphabet": ["0", "1"],
                                "target_alphabet": ["0", "1"],
                                "images": [["0", "1"], ["0"]],
                            }
                        },
                        "seed": "0",
                    },
                    "prefix_length": 8,
                    "max_order": 3,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
