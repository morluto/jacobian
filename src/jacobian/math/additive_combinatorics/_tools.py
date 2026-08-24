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
    MultisetSumRepresentationProfileRequest,
    MultisetSumRepresentationProfileResult,
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
    compute_multiset_sum_representation_profile,
    compute_ordered_difference_profile,
    compute_representation_profile,
    compute_subset_sum_profile,
    compute_sumset_cardinality,
    decide_direct_sum_predicate,
)
from jacobian.math.additive_combinatorics._subset_sum_residue import (
    MAX_RESIDUE_PROFILE_DP_CELLS,
    MAX_RESIDUE_PROFILE_MODULUS,
    MAX_RESIDUE_PROFILE_TOTAL_DP_CELLS,
    MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS,
    SubsetSumResidueProfileRequest,
    SubsetSumResidueProfileResult,
    compute_subset_sum_residue_profile,
)
from jacobian.math.additive_combinatorics.operations import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
)
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_ITEM_DIGITS,
    MAX_SUBSET_SUM_ITEMS,
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

_MULTISET_SUM_PROFILE_EXAMPLE: dict[str, Any] = {
    "source": {"elements": ["0", "1", "2"]},
    "arity": 2,
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
        "additive.multiset_sum.representation_profile.compute",
        "Compute a fixed-arity unordered multiset-sum profile",
        "Given a canonical finite integer source A and arity k, return the exact "
        "multiplicity of every sum of a nondecreasing k-tuple of source indices, "
        "with repetition allowed. An optional closed sum window returns the "
        "complete profile only inside that interval. Admission bounds complete "
        "materialized enumeration and worst-case serialized support before "
        "execution; the result retains and replays its source, arity, and scope.",
        MultisetSumRepresentationProfileRequest,
        MultisetSumRepresentationProfileResult,
        compute_multiset_sum_representation_profile,
        "additive-combinatorics",
        "multiset-sum",
        "representation-profile",
        "exact",
        examples=(
            example(
                "three_element_pair_multisums",
                "Compute all unordered two-term sums from {0,1,2}, including "
                "repeated source elements; the source must be distinct, strictly "
                "increasing, and bounded, and omitting the window requests the "
                "complete profile.",
                _MULTISET_SUM_PROFILE_EXAMPLE,
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
                    "be canonical integers inside the schema-visible "
                    f"{MAX_SUBSET_SUM_ITEMS:,}-item, "
                    f"{MAX_SUBSET_SUM_ITEM_DIGITS:,}-digit, "
                    f"{MAX_SUBSET_SUM_PROFILE_ENTRIES:,}-row, "
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
    additive_combinatorics_operation(
        "additive.subset_sum.residue_profile.compute",
        "Compute an exact modular subset-sum profile",
        (
            "Given a materialized indexed integer tuple and a positive modulus m, "
            "return the exact number of permitted index subsets in every residue "
            "class of Z/mZ. Repeated values and zeros remain distinct positions; "
            "the empty-subset convention is explicit. Optional witnesses are "
            "canonical by minimizing sum(2**i for i in I). Computation and "
            "source-binding replay visit at most "
            f"{MAX_RESIDUE_PROFILE_TOTAL_DP_CELLS:,} item-residue cells "
            f"({MAX_RESIDUE_PROFILE_DP_CELLS:,} per pass), with modulus at most "
            f"{MAX_RESIDUE_PROFILE_MODULUS:,} and at most "
            f"{MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS:,} witness index slots."
        ),
        SubsetSumResidueProfileRequest,
        SubsetSumResidueProfileResult,
        compute_subset_sum_residue_profile,
        "additive-combinatorics",
        "subset-sum",
        "modular-arithmetic",
        "multiplicity-profile",
        "exact",
        examples=(
            example(
                "nonempty_subsets_modulo_five",
                (
                    "Count all nonempty index subsets of (2,3) in every residue "
                    "class modulo 5 and return canonical witnesses; the modulus "
                    "must be positive and the derived DP, bigint, witness, input, "
                    "and exact-result bounds must be admitted before execution."
                ),
                {
                    "source": {"items": ["2", "3"]},
                    "modulus": 5,
                    "include_empty_subset": False,
                    "include_witnesses": True,
                },
            ),
        ),
    ),
)

TOOLS = ADDITIVE_COMBINATORICS_OPERATIONS

__all__ = ["TOOLS"]
