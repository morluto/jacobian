"""Numerical semigroup operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._factorization_models import (
    FactorizationComputeRequest,
    FactorizationComputeResult,
    FactorizationDistanceRequest,
    FactorizationDistanceResult,
    FactorizationGraphComputeRequest,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeRequest,
    FactorizationLengthsComputeResult,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_models import (
    BettiElementsRequest,
    BettiElementsResult,
    CatenaryDegreeRequest,
    CatenaryDegreeResult,
    DeltaSetRequest,
    DeltaSetResult,
    ElasticityRequest,
    ElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._models import (
    MAX_GENERATOR,
    _require_minimal_generators,
)
from jacobian.math.number_theory.numerical_semigroups._presentation_models import (
    MinimalPresentationRequest,
    MinimalPresentationResult,
    PresentationBinomialsRequest,
    PresentationBinomialsResult,
)
from jacobian.math.number_theory.numerical_semigroups._summary_models import (
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
    SemigroupMembershipRequest,
    SemigroupMembershipResult,
)
from jacobian.math.number_theory.numerical_semigroups.operations import (
    betti_elements,
    delta_set,
    element_catenary_degree_profile,
    element_delta_set_profile,
    element_elasticity_profile,
    factorization_distance_profile,
    factorization_graph_profile,
    factorization_lengths_profile,
    factorization_profile,
    global_catenary_degree,
    global_elasticity,
    membership,
    minimal_presentation,
    presentation_binomials,
    summary,
)


def _operation[
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
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


def _run_native[T](
    operation: str,
    location: tuple[str | int, ...],
    action: Callable[[], T],
) -> T:
    """Project one native ValueError into the catalog's typed failure."""
    try:
        return action()
    except ValueError as error:
        code = getattr(error, "type", None)
        if not isinstance(code, str):
            code = f"numerical_semigroup.{operation}_admission"
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(error),
        ) from error


def _generators(values: tuple[str, ...]) -> tuple[int, ...]:
    return _require_minimal_generators(values)


def compute_summary(
    request: NumericalSemigroupSummaryRequest,
) -> NumericalSemigroupSummaryResult:
    return _run_native(
        "summary", ("generators",), lambda: summary(_generators(request.generators))
    )


def compute_membership(
    request: SemigroupMembershipRequest,
) -> SemigroupMembershipResult:
    return _run_native(
        "membership",
        ("generators", "value"),
        lambda: membership(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_factorizations(
    request: FactorizationComputeRequest,
) -> FactorizationComputeResult:
    return _run_native(
        "factorizations",
        ("generators", "value"),
        lambda: factorization_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_factorization_lengths(
    request: FactorizationLengthsComputeRequest,
) -> FactorizationLengthsComputeResult:
    return _run_native(
        "factorization_lengths",
        ("generators", "value"),
        lambda: factorization_lengths_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_factorization_distance(
    request: FactorizationDistanceRequest,
) -> FactorizationDistanceResult:
    return _run_native(
        "factorization_distance",
        ("generators", "value", "first", "second"),
        lambda: factorization_distance_profile(
            _generators(request.generators),
            parse_canonical_integer(request.value),
            request.first,
            request.second,
        ),
    )


def compute_factorization_graph(
    request: FactorizationGraphComputeRequest,
) -> FactorizationGraphComputeResult:
    return _run_native(
        "factorization_graph",
        ("generators", "value"),
        lambda: factorization_graph_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_element_delta_set(
    request: ElementDeltaSetRequest,
) -> ElementDeltaSetResult:
    return _run_native(
        "element_delta_set",
        ("generators", "value"),
        lambda: element_delta_set_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_element_elasticity(
    request: ElementElasticityRequest,
) -> ElementElasticityResult:
    return _run_native(
        "element_elasticity",
        ("generators", "value"),
        lambda: element_elasticity_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_element_catenary_degree(
    request: ElementCatenaryDegreeRequest,
) -> ElementCatenaryDegreeResult:
    return _run_native(
        "element_catenary_degree",
        ("generators", "value"),
        lambda: element_catenary_degree_profile(
            _generators(request.generators), parse_canonical_integer(request.value)
        ),
    )


def compute_betti_elements(request: BettiElementsRequest) -> BettiElementsResult:
    return _run_native(
        "betti_elements",
        ("generators",),
        lambda: betti_elements(_generators(request.generators)),
    )


def compute_delta_set(request: DeltaSetRequest) -> DeltaSetResult:
    return _run_native(
        "delta_set", ("generators",), lambda: delta_set(_generators(request.generators))
    )


def compute_elasticity(request: ElasticityRequest) -> ElasticityResult:
    return _run_native(
        "elasticity",
        ("generators",),
        lambda: global_elasticity(_generators(request.generators)),
    )


def compute_catenary_degree(request: CatenaryDegreeRequest) -> CatenaryDegreeResult:
    return _run_native(
        "catenary_degree",
        ("generators",),
        lambda: global_catenary_degree(_generators(request.generators)),
    )


def compute_minimal_presentation(
    request: MinimalPresentationRequest,
) -> MinimalPresentationResult:
    return _run_native(
        "minimal_presentation",
        ("generators",),
        lambda: minimal_presentation(_generators(request.generators)),
    )


def compute_presentation_binomials(
    request: PresentationBinomialsRequest,
) -> PresentationBinomialsResult:
    return _run_native(
        "presentation_binomials",
        ("generators", "relations"),
        lambda: presentation_binomials(
            _generators(request.generators), request.relations
        ),
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _operation(
        "number_theory.numerical_semigroup.summary.compute",
        "Compute numerical semigroup summary",
        "Given a finite set of positive generators with gcd 1, compute "
        "minimal generators, multiplicity, embedding dimension, Frobenius "
        "number, conductor, genus, and gap list.",
        NumericalSemigroupSummaryRequest,
        NumericalSemigroupSummaryResult,
        compute_summary,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "semigroup_3_5",
                "Summary of <3,5>.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.membership.compute",
        "Check numerical semigroup membership",
        "Check whether a given integer is in the numerical semigroup "
        "generated by the given generators.",
        SemigroupMembershipRequest,
        SemigroupMembershipResult,
        compute_membership,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "membership_8_in_3_5",
                "Check if 8 is in <3,5>.",
                {"generators": ["3", "5"], "value": "8"},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.factorizations.compute",
        "Compute complete factorization family Z(s)",
        f"Given positive gcd-one generators (general-path generators each at most {MAX_GENERATOR}; "
        "a presentation containing 1 uses the constant-size free-semigroup path), and "
        "an element, compute the complete bounded factorization family Z(s) on "
        "the increasing minimal generator axis. Redundant or reordered generators "
        "normalize before exact output validation.",
        FactorizationComputeRequest,
        FactorizationComputeResult,
        compute_factorizations,
        "number-theory",
        "numerical-semigroup",
        "factorization",
        "exact",
        examples=(
            example(
                "factorizations_15_in_3_5",
                "Factorizations of 15 in <3,5>.",
                {"generators": ["3", "5"], "value": "15"},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.factorization_graph.compute",
        "Compute factorization graph with connected components",
        "Build the standard factorization graph where two factorizations "
        "are connected if they share a common atom (coordinatewise gcd is "
        "nonzero). Returns edges and connected components.",
        FactorizationGraphComputeRequest,
        FactorizationGraphComputeResult,
        compute_factorization_graph,
        "number-theory",
        "numerical-semigroup",
        "factorization",
        "exact",
        examples=(
            example(
                "graph_15_in_3_5",
                "Factorization graph of 15 in <3,5>.",
                {"generators": ["3", "5"], "value": "15"},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.betti_elements.compute",
        "Compute Betti elements of a numerical semigroup",
        "Compute the Betti elements of a numerical semigroup - elements "
        "whose factorization graph is disconnected.",
        BettiElementsRequest,
        BettiElementsResult,
        compute_betti_elements,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "betti_3_5",
                "Betti elements of <3,5>.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        "Compute a minimal presentation",
        "Compute one minimal presentation of a numerical semigroup, "
        "returning exactly r-1 relations spanning the r factorization "
        "components at each Betti element.",
        MinimalPresentationRequest,
        MinimalPresentationResult,
        compute_minimal_presentation,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "presentation_3_5",
                "Minimal presentation of <3,5>.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.presentation_binomials.compute",
        "Convert presentation to sparse binomials",
        "Normalize a positive gcd-one generator presentation (general-path generators each at most "
        f"{MAX_GENERATOR}; a presentation containing 1 uses the constant-size free-semigroup path) "
        "to its minimal axis, validate relations against that "
        "axis, and convert each relation (u,v) to the toric binomial X^u-X^v "
        "with coefficients 1 and -1.",
        PresentationBinomialsRequest,
        PresentationBinomialsResult,
        compute_presentation_binomials,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "binomials_3_5",
                "Sparse binomials of <3,5>.",
                {
                    "generators": ["3", "5"],
                    "relations": [
                        {"first": [5, 0], "second": [0, 3]},
                    ],
                },
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.delta_set.compute",
        "Compute global delta set",
        "Compute the complete global delta set through the theorem-backed "
        "eventual-periodicity bound, returning the bound and exact checked "
        "range as completeness evidence.",
        DeltaSetRequest,
        DeltaSetResult,
        compute_delta_set,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "global_delta_3_5",
                "Global delta set of <3,5>.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.catenary_degree.compute",
        "Compute global catenary degree",
        "Compute the global catenary degree as the maximum over the complete "
        "Betti set, returning per-Betti degrees and maximizing witnesses.",
        CatenaryDegreeRequest,
        CatenaryDegreeResult,
        compute_catenary_degree,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "global_catenary_3_5",
                "Global catenary degree of <3,5>.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.elasticity.global_compute",
        "Compute global elasticity",
        "Compute the global elasticity of a numerical semigroup as the exact "
        "ratio of its largest and smallest minimal generators.",
        ElasticityRequest,
        ElasticityResult,
        compute_elasticity,
        "number-theory",
        "numerical-semigroup",
        "exact",
        examples=(
            example(
                "global_elasticity_3_5",
                "Compute global elasticity for the semigroup generated by 3 and 5.",
                {"generators": ["3", "5"]},
            ),
        ),
    ),
    _operation(
        "number_theory.numerical_semigroup.elasticity.compute",
        "Compute element elasticity",
        "Compute the elasticity of one element in a numerical semigroup: "
        "the ratio of the maximum to minimum factorization length over all "
        "factorizations of the element. Returns as an exact fraction.",
        ElementElasticityRequest,
        ElementElasticityResult,
        compute_element_elasticity,
        "number-theory",
        "numerical-semigroup",
        "factorization",
        "exact",
        examples=(
            example(
                "elasticity_15_in_3_5",
                "Elasticity of 15 in <3,5>; positive gcd-one general-path generators, each at "
                f"most {MAX_GENERATOR}, normalize to the minimal axis and the "
                "value must be a positive member.",
                {"generators": ["3", "5"], "value": "15"},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
