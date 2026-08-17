"""Exact additive combinatorics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.additive_combinatorics._models import (
    AdditiveEnergyRequest,
    AdditiveEnergyResult,
    DirectSumPredicateRequest,
    DirectSumPredicateResult,
    RepresentationProfileRequest,
    RepresentationProfileResult,
    SumsetCardinalityRequest,
    SumsetCardinalityResult,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_additive_energy,
    compute_representation_profile,
    compute_sumset_cardinality,
    decide_direct_sum_predicate,
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
)


TOOLS = ADDITIVE_COMBINATORICS_OPERATIONS

__all__ = ["TOOLS"]
