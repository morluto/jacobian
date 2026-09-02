"""Exact additive combinatorics operation declarations."""

from typing import Any

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
    MathTool(
        operation_id="additive.subset_sum.target.solve",
        title="Solve one exact indexed subset-sum target",
        description=(
            "For a bounded indexed integer sequence and one integer target, "
            "return the canonical attaining index subset or establish exact "
            "non-attainment after exhausting the admitted reachable-sum state space. "
            "The bounded kernel performs at most "
            f"{MAX_SUBSET_SUM_TRANSITIONS:,} state transitions."
        ),
        request_type=SubsetSumTargetRequest,
        result_type=SubsetSumTargetResult,
        run=_run_subset_sum_target,
        tags=("additive-combinatorics", "subset-sum", "decision", "witness", "exact"),
        examples=(
            OperationExample(
                name="two_item_target",
                description="The distinct source indices 0 and 1 witness 2+3=5.",
                input={
                    "source": {"items": ["2", "3"]},
                    "target": "5",
                    "allow_empty_subset": False,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="additive.representation_profile.compute",
        title="Compute the representation profile of a sumset",
        description="Given two finite integer sets A and B, return r_{A+B}(x) = "
        "|{(a,b) in AxBy : a+b=x}| for every sum x, as the sorted support "
        "with multiplicities.",
        request_type=RepresentationProfileRequest,
        result_type=RepresentationProfileResult,
        run=_run_representation_profile,
        tags=("additive-combinatorics", "representation-function", "sumset", "exact"),
        examples=(
            OperationExample(
                name="two_by_two_sumset",
                description=(
                    "A={1,2}, B={3,4}: r(4)=1, r(5)=2, r(6)=1; "
                    "E(A,B)=6 is derivable from this profile."
                ),
                input=_REPRESENTATION_PROFILE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.energy.compute",
        title="Compute additive energy",
        description="Return the exact additive energy E(A,B) as the sum of squared "
        "representation multiplicities, together with its decomposition by sum.",
        request_type=AdditiveEnergyRequest,
        result_type=AdditiveEnergyResult,
        run=_run_energy,
        tags=("additive-combinatorics", "energy", "sumset", "exact"),
        examples=(
            OperationExample(
                name="two_by_two_energy",
                description="For A={1,2} and B={3,4}, the representation multiplicities are 1,2,1 and the energy is 6.",
                input=_ADDITIVE_ENERGY_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.sumset_cardinality.compute",
        title="Compute sumset cardinality",
        description="Return the exact cardinality and canonical finite-integer-set support "
        "of the sumset A+B.",
        request_type=SumsetCardinalityRequest,
        result_type=SumsetCardinalityResult,
        run=_run_sumset_cardinality,
        tags=("additive-combinatorics", "sumset", "cardinality", "exact"),
        examples=(
            OperationExample(
                name="two_by_two_sumset",
                description="For A={0,1,2} and B={0,2}, return the five-element support of A+B.",
                input=_SUMSET_CARDINALITY_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.multiset_sum.representation_profile.compute",
        title="Compute a fixed-arity unordered multiset-sum profile",
        description="Given a canonical finite integer source A and arity k, return the exact "
        "multiplicity of every sum of a nondecreasing k-tuple of source indices, "
        "with repetition allowed. An optional closed sum window returns the "
        "complete profile only inside that interval. Admission bounds complete "
        "materialized enumeration and worst-case serialized support before "
        "execution; the result retains its source, arity, and scope.",
        request_type=MultisetSumRepresentationProfileRequest,
        result_type=MultisetSumRepresentationProfileResult,
        run=_run_multiset_sum,
        tags=(
            "additive-combinatorics",
            "multiset-sum",
            "representation-profile",
            "exact",
        ),
        examples=(
            OperationExample(
                name="three_element_pair_multisums",
                description="Compute all unordered two-term sums from {0,1,2}, including "
                "repeated source elements; the source must be distinct, strictly "
                "increasing, and bounded, and omitting the window requests the "
                "complete profile.",
                input=_MULTISET_SUM_PROFILE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.subset_sum.profile.compute",
        title="Compute a complete indexed subset-sum multiplicity profile",
        description="Given one finite indexed integer sequence, return the exact number of "
        "index subsets attaining every integer sum. Each position is selectable "
        "at most once, equal values remain distinct positions, and the empty "
        "subset is included. Before execution, result-sensitive admission bounds "
        f"support by {MAX_SUBSET_SUM_PROFILE_ENTRIES:,} rows, two complete sparse-DP "
        f"passes by {MAX_SUBSET_SUM_DP_TRANSITIONS:,} dictionary transitions; "
        "every accepted result is complete.",
        request_type=SubsetSumProfileRequest,
        result_type=SubsetSumProfile,
        run=_run_subset_sum_profile,
        tags=(
            "additive-combinatorics",
            "subset-sum",
            "representation-profile",
            "indexed",
            "exact",
        ),
        examples=(
            OperationExample(
                name="repeated_indexed_values",
                description=(
                    "Compute all subset sums of the two indexed values [1,1], "
                    "giving multiplicities 1,2,1 at sums 0,1,2; input items must "
                    "be canonical integers inside the "
                    f"{MAX_SUBSET_SUM_ITEMS:,}-item, "
                    f"{MAX_SUBSET_SUM_ITEM_DIGITS:,}-digit, "
                    f"{MAX_SUBSET_SUM_PROFILE_ENTRIES:,}-row, "
                    f"{MAX_SUBSET_SUM_DP_TRANSITIONS:,}-transition profile bounds."
                ),
                input=_SUBSET_SUM_PROFILE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.direct_sum_predicate.compute",
        title="Direct sum / tiling predicate in a finite cyclic group",
        description="Given finite sets A, B inside Z_n, decide whether A ⊕ B = Z_n, "
        "i.e. every residue class modulo n admits a unique representation "
        "(a+b) mod n with a in A and b in B. This is the exact "
        "direct-factorization predicate. Diagnostics list representatives, "
        "collisions (multiple representations), and missing residues.",
        request_type=DirectSumPredicateRequest,
        result_type=DirectSumPredicateResult,
        run=_run_direct_sum,
        tags=(
            "additive-combinatorics",
            "direct-sum",
            "tiling",
            "cyclic-group",
            "exact",
        ),
        examples=(
            OperationExample(
                name="tiling_z4",
                description=(
                    "A={0,1}, B={0,2} in Z_4: every residue has a unique "
                    "representation, so A ⊕ B = Z_4."
                ),
                input=_DIRECT_SUM_EXAMPLE,
            ),
            OperationExample(
                name="non_tiling_z4",
                description=(
                    "A={0,1}, B={0,1} in Z_4: residue 0 and residue 2 each "
                    "have two representations, so A ⊕ B ≠ Z_4."
                ),
                input=_DIRECT_SUM_NON_TILING_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="additive.ordered_difference_profile.compute",
        title="Compute the ordered-difference profile of a set in Z^d",
        description="Given a finite set A in Z^d, return r_{A-A}(v) = |{(x,y) in A^2 : "
        "x != y, x - y = v}| for every nonzero difference v, preserving every "
        f"ordered source pair. Inputs are bounded: 1<=d<={_MAX_DIMENSION}, each coordinate "
        f"at most 6 digits in magnitude, vectors distinct and equal-length, set "
        f"size at most {_MAX_VECTOR_SET_SIZE}.  A Sidon decision, additive "
        "energy, or collision count is a cheap projection of this complete profile.",
        request_type=OrderedDifferenceProfileRequest,
        result_type=OrderedDifferenceProfileResult,
        run=_run_ordered_difference,
        tags=("additive-combinatorics", "difference-profile", "exact"),
        examples=(
            OperationExample(
                name="three_vectors",
                description="Compute the ordered-difference profile for {(0,0), (1,0), (0,1)}; "
                "vectors must be non-empty, distinct, share the same dimension "
                f"1..{_MAX_DIMENSION}, each coordinate is at most 6 digits in magnitude, and at "
                f"most {_MAX_VECTOR_SET_SIZE} vectors are accepted.",
                input={
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
    MathTool(
        operation_id="additive.subset_sum.residue_profile.compute",
        title="Compute an exact modular subset-sum profile",
        description=(
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
        request_type=SubsetSumResidueProfileRequest,
        result_type=SubsetSumResidueProfileResult,
        run=_run_subset_sum_residue,
        tags=(
            "additive-combinatorics",
            "subset-sum",
            "modular-arithmetic",
            "multiplicity-profile",
            "exact",
        ),
        examples=(
            OperationExample(
                name="nonempty_subsets_modulo_five",
                description=(
                    "Count all nonempty index subsets of (2,3) in every residue "
                    "class modulo 5 and return canonical witnesses; the modulus "
                    "must be positive and the derived DP, bigint, witness, input, "
                    "and exact-result bounds must be admitted before execution."
                ),
                input={
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
