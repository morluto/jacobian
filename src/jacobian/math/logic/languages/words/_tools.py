"""Public combinatorics-on-words operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.logic.languages.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
    SubstitutionDependencyGraphRequest,
    SubstitutionDependencyGraphResult,
    SubstitutionFixedPointPrefixRequest,
    SubstitutionFixedPointPrefixResult,
    SubstitutionPrimitivityProfileRequest,
    SubstitutionPrimitivityProfileResult,
)
from jacobian.math.logic.languages.words.operations import (
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    periods,
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
    except ValueError as exc:
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
            example(
                "abaab_factors_2",
                "Enumerate all length-two factors of abaab.",
                {
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
            example(
                "ababab_periods",
                "Compute the complete period profile of ababab.",
                {
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
            example(
                "fibonacci_matrix",
                "Compute the incidence matrix of a->ab and b->a.",
                {
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
            example(
                "fibonacci_dependencies",
                "Compute the dependency graph of the Fibonacci substitution.",
                {
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
            example(
                "fibonacci_primitivity",
                "Prove the Fibonacci substitution primitive from its dependency graph.",
                {
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
            "bounding source size, generation work, and serialized output."
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
            example(
                "fibonacci_prefix",
                "Compute eight letters of the Fibonacci fixed point.",
                {
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
)


__all__ = ["TOOLS"]
