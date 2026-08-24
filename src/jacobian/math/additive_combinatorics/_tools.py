"""Exact additive combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.additive_combinatorics._models import (
    _MAX_VECTOR_SET_SIZE,
    AdditiveEnergyRequest,
    AdditiveEnergyResult,
    DirectSumPredicateRequest,
    DirectSumPredicateResult,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
    RepresentationProfileRequest,
    RepresentationProfileResult,
    SubsetSumProfileRequest,
    SumsetCardinalityRequest,
    SumsetCardinalityResult,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_additive_energy,
    compute_ordered_difference_profile,
    compute_representation_profile,
    compute_subset_sum_profile,
    compute_sumset_cardinality,
    decide_direct_sum_predicate,
)
from jacobian.math.additive_combinatorics.operations import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
)
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_PROFILE_ENTRIES,
    SubsetSumProfile,
)


def additive_combinatorics_operation[
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


# Reusable invocation payloads for the example blocks below.

_REPRESENTATION_PROFILE_EXAMPLE: dict[str, Any] = {
    "left": {"elements": ["1", "2"]},
    "right": {"elements": ["3", "4"]},
}

_SUBSET_SUM_PROFILE_EXAMPLE: dict[str, Any] = {
    "source": {"items": ["1", "1"]},
}

_ADDITIVE_ENERGY_EXAMPLE: dict[str, Any] = {
    "left": {"elements": ["1", "2"]},
    "right": {"elements": ["3", "4"]},
}

_SUMSET_CARDINALITY_EXAMPLE: dict[str, Any] = {
    "left": {"elements": ["0", "1", "2"]},
    "right": {"elements": ["0", "2"]},
}

_DIRECT_SUM_EXAMPLE: dict[str, Any] = {
    "modulus": 4,
    "left": {"elements": ["0", "1"]},
    "right": {"elements": ["0", "2"]},
}

_DIRECT_SUM_NON_TILING_EXAMPLE: dict[str, Any] = {
    "modulus": 4,
    "left": {"elements": ["0", "1"]},
    "right": {"elements": ["0", "1"]},
}


ADDITIVE_COMBINATORICS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    additive_combinatorics_operation(
        "additive.representation_profile.compute",
        "Compute the representation profile of a sumset",
        "Given two finite integer sets A and B, return r_{A+B}(x) = "
        "|{(a,b) in AxBy : a+b=x}| for every sum x, as the sorted support "
        "with multiplicities.",
        RepresentationProfileRequest,
        RepresentationProfileResult,
        compute_representation_profile,
        "additive-combinatorics",
        "representation-function",
        "sumset",
        "exact",
        examples=(
            example(
                "two_by_two_sumset",
                (
                    "A={1,2}, B={3,4}: r(4)=1, r(5)=2, r(6)=1; "
                    "E(A,B)=6 is derivable from this profile."
                ),
                _REPRESENTATION_PROFILE_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.subset_sum.profile.compute",
        "Compute a complete indexed subset-sum multiplicity profile",
        "Given one finite indexed integer sequence, return the exact number of "
        "index subsets attaining every integer sum. Each position is selectable "
        "at most once, equal values remain distinct positions, and the empty "
        "subset is included. Before execution, result-sensitive admission bounds "
        f"support by {MAX_SUBSET_SUM_PROFILE_ENTRIES:,} rows, two complete sparse-DP "
        f"passes by {MAX_SUBSET_SUM_DP_TRANSITIONS:,} dictionary transitions, and "
        "the conservative serialized result by 4 MiB; "
        "every accepted result is complete.",
        SubsetSumProfileRequest,
        SubsetSumProfile,
        compute_subset_sum_profile,
        "additive-combinatorics",
        "subset-sum",
        "representation-profile",
        "indexed",
        "exact",
        examples=(
            example(
                "repeated_indexed_values",
                (
                    "Compute all subset sums of the two indexed values [1,1], "
                    "giving multiplicities 1,2,1 at sums 0,1,2; input items must "
                    "be canonical integers inside the schema-visible 256-item, "
                    f"256-digit, {MAX_SUBSET_SUM_PROFILE_ENTRIES:,}-row, "
                    f"{MAX_SUBSET_SUM_DP_TRANSITIONS:,}-transition, and "
                    f"{MAX_SUBSET_SUM_PROFILE_RESULT_BYTES // (1024 * 1024)} MiB "
                    "profile bounds."
                ),
                _SUBSET_SUM_PROFILE_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.energy.compute",
        "Compute the additive energy of two integer sets",
        "Given two finite integer sets A and B, compute E(A,B) = "
        "sum_x r_{A+B}(x)^2 = #{(a,b,a',b') : a+b=a'+b'} exactly, "
        "with the per-sum decomposition.",
        AdditiveEnergyRequest,
        AdditiveEnergyResult,
        compute_additive_energy,
        "additive-combinatorics",
        "additive-energy",
        "sumset",
        "exact",
        examples=(
            example(
                "two_by_two_energy",
                "A={1,2}, B={3,4}: E(A,B)=1+4+1=6.",
                _ADDITIVE_ENERGY_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.sumset_cardinality.compute",
        "Compute the cardinality of a sumset",
        "Given two finite integer sets A and B, compute |A+B|, the support "
        "cardinality of the representation profile, with the sorted support.",
        SumsetCardinalityRequest,
        SumsetCardinalityResult,
        compute_sumset_cardinality,
        "additive-combinatorics",
        "sumset",
        "cardinality",
        "exact",
        examples=(
            example(
                "three_plus_two_sumset",
                ("A={0,1,2}, B={0,2}: A+B={0,1,2,3,4} and |A+B|=5."),
                _SUMSET_CARDINALITY_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.direct_sum_predicate.compute",
        "Direct sum / tiling predicate in a finite cyclic group",
        "Given finite sets A, B inside Z_n, decide whether A ⊕ B = Z_n, "
        "i.e. every residue class modulo n admits a unique representation "
        "(a+b) mod n with a in A and b in B. This is the exact "
        "direct-factorization predicate. Diagnostics list representatives, "
        "collisions (multiple representations), and missing residues.",
        DirectSumPredicateRequest,
        DirectSumPredicateResult,
        decide_direct_sum_predicate,
        "additive-combinatorics",
        "direct-sum",
        "tiling",
        "cyclic-group",
        "exact",
        examples=(
            example(
                "tiling_z4",
                (
                    "A={0,1}, B={0,2} in Z_4: every residue has a unique "
                    "representation, so A ⊕ B = Z_4."
                ),
                _DIRECT_SUM_EXAMPLE,
            ),
            example(
                "non_tiling_z4",
                (
                    "A={0,1}, B={0,1} in Z_4: residue 0 and residue 2 each "
                    "have two representations, so A ⊕ B ≠ Z_4."
                ),
                _DIRECT_SUM_NON_TILING_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.ordered_difference_profile.compute",
        "Compute the ordered-difference profile of a set in Z^d",
        "Given a finite set A in Z^d, return r_{A-A}(v) = |{(x,y) in A^2 : "
        "x != y, x - y = v}| for every nonzero difference v, preserving every "
        f"ordered source pair.  Inputs are bounded: 1<=d<=8, each coordinate "
        f"at most 6 digits in magnitude, vectors distinct and equal-length, set "
        f"size at most {_MAX_VECTOR_SET_SIZE}.  A Sidon decision, additive "
        "energy, or collision count is a cheap projection of this complete profile.",
        OrderedDifferenceProfileRequest,
        OrderedDifferenceProfileResult,
        compute_ordered_difference_profile,
        "additive-combinatorics",
        "difference-profile",
        "exact",
        examples=(
            example(
                "three_vectors",
                "Compute the ordered-difference profile for {(0,0), (1,0), (0,1)}; "
                "vectors must be non-empty, distinct, share the same dimension "
                "1..8, each coordinate is at most 6 digits in magnitude, and at "
                f"most {_MAX_VECTOR_SET_SIZE} vectors are accepted.",
                {
                    "vectors": {
                        "vectors": [
                            {"coordinates": ["0", "0"]},
                            {"coordinates": ["1", "0"]},
                            {"coordinates": ["0", "1"]},
                        ]
                    }
                },
            ),
        ),
    ),
)

TOOLS = ADDITIVE_COMBINATORICS_OPERATIONS

__all__ = ["TOOLS"]
