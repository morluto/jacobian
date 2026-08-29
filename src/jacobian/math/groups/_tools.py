"""Exact finite group operation declarations."""

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.groups import operations as native
from jacobian.math.groups._models import (
    MAX_CONJUGACY_CLASSES_GROUP_ORDER,
    GroupConjugacyClassesRequest,
    GroupConjugacyClassesResult,
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    GroupStabilizerRequest,
    GroupStabilizerResult,
    GroupSubgroupLatticeRequest,
    GroupSubgroupLatticeResult,
    PermutationGroup,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    FiniteAbelianProductGroup,
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    decide_finite_abelian_spectral_pair,
    finite_abelian_group_factorization,
)
from jacobian.math.groups.operations import SubgroupLatticeBudgetExceededError


def compute_finite_abelian_group_factorization(
    request: FiniteAbelianGroupFactorizationRequest,
) -> FiniteAbelianGroupFactorizationResult:
    try:
        group = FiniteAbelianProductGroup(moduli=request.moduli)
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("moduli",),
            code="finite_abelian_group.invalid_group",
            message="moduli must define an admitted finite Abelian product group",
        ) from error
    return finite_abelian_group_factorization(group, request.left, request.right)


def compute_finite_abelian_spectral_pair(
    request: FiniteAbelianSpectralPairRequest,
) -> FiniteAbelianSpectralPairResult:
    return decide_finite_abelian_spectral_pair(request.source)


def compute_group_order(request: PermutationGroup) -> GroupOrderResult:
    order = native.group_order(request)
    return GroupOrderResult(order=format_canonical_integer(order))


def compute_element_order(request: GroupElementOrderRequest) -> GroupElementOrderResult:
    order = native.element_order(request.degree, list(request.generator))
    return GroupElementOrderResult(order=format_canonical_integer(order))


def compute_group_orbit(request: GroupOrbitRequest) -> GroupOrbitResult:
    orbit = native.group_orbit(request.group, request.point)
    return GroupOrbitResult(orbit=tuple(orbit), point=request.point)


def compute_group_conjugacy_classes(
    request: GroupConjugacyClassesRequest,
) -> GroupConjugacyClassesResult:
    classes = native.group_conjugacy_classes(
        request.degree,
        [list(g) for g in request.generators],
    )
    return GroupConjugacyClassesResult._from_kernel(
        tuple(tuple(tuple(p) for p in cls) for cls in classes),
    )


def compute_group_stabilizer(request: GroupStabilizerRequest) -> GroupStabilizerResult:
    return GroupStabilizerResult._from_kernel(
        request.point,
        request.group,
        native.group_stabilizer(request.group, request.point),
    )


def compute_subgroup_lattice(
    request: GroupSubgroupLatticeRequest,
) -> GroupSubgroupLatticeResult:
    try:
        source = PermutationGroup(degree=request.degree, generators=request.generators)
    except ValidationError as error:
        detail = error.errors(include_url=False, include_context=False)[0]
        raise OperationDomainValidationError(
            location=("generators",),
            code=str(detail["type"]),
            message=str(detail["msg"]),
        ) from error
    try:
        subgroups = native.subgroup_lattice(source)
    except SubgroupLatticeBudgetExceededError as error:
        return GroupSubgroupLatticeResult._limit_exceeded_from_kernel(
            request, str(error)
        )
    return GroupSubgroupLatticeResult._computed_from_kernel(request, tuple(subgroups))


def group_operation[
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


S3_GENERATORS = {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]}
S3_STABILIZER_POINT_0 = {
    "group": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
    "point": 0,
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    group_operation(
        "finite_abelian_group.exact_factorization.compute",
        "Exact finite abelian group factorization",
        "Normalize two bounded integer-vector factors in a declared product "
        "of cyclic groups, exhaustively count every sum representation, and "
        "decide whether every group element has exactly one representation.",
        FiniteAbelianGroupFactorizationRequest,
        FiniteAbelianGroupFactorizationResult,
        compute_finite_abelian_group_factorization,
        "group",
        "finite-abelian-group",
        "cyclic-product",
        "factorization",
        "exact",
        examples=(
            example(
                "z2_times_z4_transversal",
                "Verify eight representatives form a complete transversal.",
                {
                    "moduli": [2, 4],
                    "left": [
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3],
                    ],
                    "right": [[0, 0]],
                },
            ),
        ),
    ),
    group_operation(
        "finite_abelian_group.spectral_pair.decide",
        "Decide an exact finite-Abelian spectral pair",
        "Decide whether a canonical residue-tuple frequency set is a spectrum "
        "of a point set in an explicit product of cyclic groups under the "
        "positive product dual pairing.",
        FiniteAbelianSpectralPairRequest,
        FiniteAbelianSpectralPairResult,
        compute_finite_abelian_spectral_pair,
        "harmonic-analysis",
        "finite-abelian-group",
        "spectral-pair",
        "exact",
        examples=(
            example(
                "z4_even_pair",
                "Decide the two-point spectral pair A={0,2}, Lambda={0,1} in Z/4.",
                {
                    "source": {
                        "group": {"moduli": [4]},
                        "points": [[0], [2]],
                        "frequencies": [[0], [1]],
                    }
                },
            ),
        ),
    ),
    group_operation(
        "group.order.compute",
        "Compute the exact order of a finite permutation group",
        "Compute the exact order of a permutation group given by generators via SymPy's Schreier-Sims algorithm.",
        PermutationGroup,
        GroupOrderResult,
        compute_group_order,
        "group",
        "order",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_group_order_4",
                "Compute C4's order; each generator must be a bijection of 0..degree-1.",
                {
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
    group_operation(
        "group.element_order.compute",
        "Compute the exact order of one permutation",
        "Compute the order of one permutation element via SymPy's Permutation.order().",
        GroupElementOrderRequest,
        GroupElementOrderResult,
        compute_element_order,
        "group",
        "element-order",
        "permutation",
        "exact",
        examples=(
            example(
                "four_cycle_order",
                "Compute the 4-cycle's order; its generator must be a bijection of 0..degree-1.",
                {
                    "degree": 4,
                    "generator": [1, 2, 3, 0],
                },
            ),
        ),
    ),
    group_operation(
        "group.orbit.compute",
        "Compute the orbit of a point under a permutation group",
        "Compute the orbit of a point under a permutation group via SymPy's "
        "PermutationGroup.orbit(). The request takes the canonical "
        "permutation-group value, so a previous result's stabilizer or order "
        "group feeds this operation unchanged.",
        GroupOrbitRequest,
        GroupOrbitResult,
        compute_group_orbit,
        "group",
        "orbit",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_orbit",
                "Compute point 0's orbit of the group generated by (1,2,3,0); the group is the canonical value and points lie in 0..degree-1.",
                {
                    "group": {
                        "degree": 4,
                        "generators": [[1, 2, 3, 0]],
                    },
                    "point": 0,
                },
            ),
        ),
    ),
    group_operation(
        "group.conjugacy_classes.compute",
        "Compute conjugacy classes of a permutation group",
        "Given a permutation group by generators, return its conjugacy classes "
        "(the partition into conjugacy classes) as permutation array forms, "
        "using SymPy. Each class lists the elements conjugate to a "
        "representative; class sizes are orbit sizes under conjugation. "
        "Classes are canonically ordered (members sorted, classes sorted by "
        "smallest member), so the same group always yields an identical "
        "result. The generated group must have order at most "
        f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER}; larger "
        "groups are rejected before enumeration.",
        GroupConjugacyClassesRequest,
        GroupConjugacyClassesResult,
        compute_group_conjugacy_classes,
        "group",
        "permutation",
        "conjugacy",
        "exact",
        examples=(
            example(
                "s3_conjugacy_classes",
                (
                    "Conjugacy classes of S3 (generators (1,2,0) and (1,0,2)); "
                    "S3 has three classes of sizes 1, 2, 3 (identity, 3-cycles, "
                    "transpositions). Generators must be permutations of 0..n-1."
                ),
                S3_GENERATORS,
            ),
        ),
    ),
    group_operation(
        "group.stabilizer.compute",
        "Compute the stabilizer of a point in a permutation group",
        "Given a permutation group as the canonical group value and a point, "
        "return the point stabilizer subgroup as a canonical permutation-group "
        "value (elements fixing the point) using SymPy's stabilizer "
        "computation. The request accepts that canonical value directly, so a "
        "previous result's `stabilizer` subgroup chains into the next request "
        "unchanged. By the orbit-stabilizer theorem, "
        "|G| = |orbit(point)| * |stabilizer(point)|, composable with "
        "group.order.compute and group.orbit.compute.",
        GroupStabilizerRequest,
        GroupStabilizerResult,
        compute_group_stabilizer,
        "group",
        "permutation",
        "stabilizer",
        "orbit-stabilizer",
        "exact",
        examples=(
            example(
                "s3_stabilizer_of_0",
                (
                    "Stabilizer of point 0 in S3 (generators (1,2,0) and "
                    "(1,0,2)); order 2 and orbit-stabilizer gives 6 = 3 * 2. "
                    "Pass a previous result's `stabilizer` as `group` to chain."
                ),
                S3_STABILIZER_POINT_0,
            ),
        ),
    ),
    group_operation(
        "group.subgroup_lattice.compute",
        "Enumerate all subgroups of a bounded permutation group",
        "Enumerate all subgroups of a bounded permutation group via SymPy. "
        "Each subgroup is returned with its generators and order. Bounded "
        "to groups of order at most 64; the traversal carries an explicit "
        "closure-construction budget and reports exhaustion as a typed "
        "LIMIT_EXCEEDED outcome.",
        GroupSubgroupLatticeRequest,
        GroupSubgroupLatticeResult,
        compute_subgroup_lattice,
        "group",
        "subgroup",
        "permutation",
        "exact",
        examples=(
            example(
                "c4_subgroups",
                "Enumerate all subgroups of C4; generators must be bijections.",
                {
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
