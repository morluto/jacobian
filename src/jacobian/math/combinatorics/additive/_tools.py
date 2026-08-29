"""Exact additive combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.additive._models import (
    _MAX_DIMENSION,
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
from jacobian.math.combinatorics.additive._subset_sum_profile import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
)
from jacobian.math.combinatorics.additive._subset_sum_residue import (
    MAX_RESIDUE_PROFILE_DP_CELLS,
    MAX_RESIDUE_PROFILE_MODULUS,
    MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS,
    SubsetSumResidueProfileRequest,
    SubsetSumResidueProfileResult,
    subset_sum_residue_profile,
)
from jacobian.math.combinatorics.additive._subset_sum_target import (
    MAX_SUBSET_SUM_TRANSITIONS,
    SubsetSumTargetRequest,
    SubsetSumTargetResult,
    solve_subset_sum_target,
)
from jacobian.math.combinatorics.additive.operations import (
    additive_energy,
    direct_sum_predicate,
    multiset_sum_representation_profile,
    ordered_difference_profile,
    representation_profile,
    subset_sum_profile,
    sumset_cardinality,
)
from jacobian.math.combinatorics.additive.values import (
    MAX_SUBSET_SUM_ITEM_DIGITS,
    MAX_SUBSET_SUM_ITEMS,
    MAX_SUBSET_SUM_PROFILE_ENTRIES,
    SubsetSumProfile,
)


def _run_representation_profile(
    request: RepresentationProfileRequest,
) -> RepresentationProfileResult:
    return representation_profile(request.left, request.right)


def _run_energy(request: AdditiveEnergyRequest) -> AdditiveEnergyResult:
    return additive_energy(request.left, request.right)


def _run_sumset_cardinality(
    request: SumsetCardinalityRequest,
) -> SumsetCardinalityResult:
    return sumset_cardinality(request.left, request.right)


def _run_multiset_sum(
    request: MultisetSumRepresentationProfileRequest,
) -> MultisetSumRepresentationProfileResult:
    return multiset_sum_representation_profile(
        request.source, request.arity, request.window
    )


def _run_subset_sum_profile(request: SubsetSumProfileRequest) -> SubsetSumProfile:
    return subset_sum_profile(request.source)


def _run_direct_sum(request: DirectSumPredicateRequest) -> DirectSumPredicateResult:
    return direct_sum_predicate(request.modulus, request.left, request.right)


def _run_ordered_difference(
    request: OrderedDifferenceProfileRequest,
) -> OrderedDifferenceProfileResult:
    return ordered_difference_profile(request.vectors)


def _run_subset_sum_target(request: SubsetSumTargetRequest) -> SubsetSumTargetResult:
    return solve_subset_sum_target(
        request.source, request.target, request.allow_empty_subset
    )


def _run_subset_sum_residue(
    request: SubsetSumResidueProfileRequest,
) -> SubsetSumResidueProfileResult:
    return subset_sum_residue_profile(
        request.source,
        request.modulus,
        request.include_empty_subset,
        request.include_witnesses,
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    additive_combinatorics_operation(
        "additive.subset_sum.target.solve",
        "Solve one exact indexed subset-sum target",
        (
            "For a bounded indexed integer sequence and one integer target, "
            "return the canonical attaining index subset or establish exact "
            "non-attainment after exhausting the admitted reachable-sum state space. "
            "The bounded kernel performs at most "
            f"{MAX_SUBSET_SUM_TRANSITIONS:,} state transitions."
        ),
        SubsetSumTargetRequest,
        SubsetSumTargetResult,
        _run_subset_sum_target,
        "additive-combinatorics",
        "subset-sum",
        "decision",
        "witness",
        "exact",
        examples=(
            example(
                "two_item_target",
                "The distinct source indices 0 and 1 witness 2+3=5.",
                {
                    "source": {"items": ["2", "3"]},
                    "target": "5",
                    "allow_empty_subset": False,
                },
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.representation_profile.compute",
        "Compute the representation profile of a sumset",
        "Given two finite integer sets A and B, return r_{A+B}(x) = "
        "|{(a,b) in AxBy : a+b=x}| for every sum x, as the sorted support "
        "with multiplicities.",
        RepresentationProfileRequest,
        RepresentationProfileResult,
        _run_representation_profile,
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
        "additive.energy.compute",
        "Compute additive energy",
        "Return the exact additive energy E(A,B) as the sum of squared "
        "representation multiplicities, together with its decomposition by sum.",
        AdditiveEnergyRequest,
        AdditiveEnergyResult,
        _run_energy,
        "additive-combinatorics",
        "energy",
        "sumset",
        "exact",
        examples=(
            example(
                "two_by_two_energy",
                "For A={1,2} and B={3,4}, the representation multiplicities are 1,2,1 and the energy is 6.",
                _ADDITIVE_ENERGY_EXAMPLE,
            ),
        ),
    ),
    additive_combinatorics_operation(
        "additive.sumset_cardinality.compute",
        "Compute sumset cardinality",
        "Return the exact cardinality and sorted support of the sumset A+B.",
        SumsetCardinalityRequest,
        SumsetCardinalityResult,
        _run_sumset_cardinality,
        "additive-combinatorics",
        "sumset",
        "cardinality",
        "exact",
        examples=(
            example(
                "two_by_two_sumset",
                "For A={0,1,2} and B={0,2}, return the five-element support of A+B.",
                _SUMSET_CARDINALITY_EXAMPLE,
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
        "execution; the result retains its source, arity, and scope.",
        MultisetSumRepresentationProfileRequest,
        MultisetSumRepresentationProfileResult,
        _run_multiset_sum,
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
        _run_subset_sum_profile,
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
                    "be canonical integers inside the "
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
        "additive.direct_sum_predicate.compute",
        "Direct sum / tiling predicate in a finite cyclic group",
        "Given finite sets A, B inside Z_n, decide whether A ⊕ B = Z_n, "
        "i.e. every residue class modulo n admits a unique representation "
        "(a+b) mod n with a in A and b in B. This is the exact "
        "direct-factorization predicate. Diagnostics list representatives, "
        "collisions (multiple representations), and missing residues.",
        DirectSumPredicateRequest,
        DirectSumPredicateResult,
        _run_direct_sum,
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
        f"ordered source pair. Inputs are bounded: 1<=d<={_MAX_DIMENSION}, each coordinate "
        f"at most 6 digits in magnitude, vectors distinct and equal-length, set "
        f"size at most {_MAX_VECTOR_SET_SIZE}.  A Sidon decision, additive "
        "energy, or collision count is a cheap projection of this complete profile.",
        OrderedDifferenceProfileRequest,
        OrderedDifferenceProfileResult,
        _run_ordered_difference,
        "additive-combinatorics",
        "difference-profile",
        "exact",
        examples=(
            example(
                "three_vectors",
                "Compute the ordered-difference profile for {(0,0), (1,0), (0,1)}; "
                "vectors must be non-empty, distinct, share the same dimension "
                f"1..{_MAX_DIMENSION}, each coordinate is at most 6 digits in magnitude, and at "
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
            "canonical by minimizing sum(2**i for i in I). The dense recurrence "
            f"visits at most {MAX_RESIDUE_PROFILE_DP_CELLS:,} item-residue cells, "
            "with modulus at most "
            f"{MAX_RESIDUE_PROFILE_MODULUS:,} and at most "
            f"{MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS:,} witness index slots."
        ),
        SubsetSumResidueProfileRequest,
        SubsetSumResidueProfileResult,
        _run_subset_sum_residue,
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


__all__ = ["TOOLS"]
