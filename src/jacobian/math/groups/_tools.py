"""Exact finite group operation declarations."""

from typing import Any

from pydantic import ValidationError

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
    FiniteAbelianCharacterSumIntervalProfileRequest,
    FiniteAbelianCharacterSumIntervalProfileResult,
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    FiniteAbelianProductGroup,
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    decide_finite_abelian_spectral_pair,
    finite_abelian_group_factorization,
)
from jacobian.math.groups.finite_abelian import (
    compute_finite_abelian_character_sum_interval_profile as native_character_sum_interval_profile,
)


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


def compute_finite_abelian_character_sum_interval_profile(
    request: FiniteAbelianCharacterSumIntervalProfileRequest,
) -> FiniteAbelianCharacterSumIntervalProfileResult:
    try:
        return native_character_sum_interval_profile(request.source)
    except OperationDomainValidationError:
        raise
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="finite_abelian_group.character_sum_not_admitted",
            message=str(exc),
        ) from exc


def compute_group_order(request: PermutationGroup) -> GroupOrderResult:
    order = native.group_order(request)
    return GroupOrderResult(source=request, order=order)


def compute_element_order(request: GroupElementOrderRequest) -> GroupElementOrderResult:
    order = native.element_order(request.degree, list(request.generator))
    source = PermutationGroup(degree=request.degree, generators=(request.generator,))
    return GroupElementOrderResult(
        source=source, element=request.generator, order=order
    )


def compute_group_orbit(request: GroupOrbitRequest) -> GroupOrbitResult:
    orbit = native.group_orbit(request.group, request.point)
    return GroupOrbitResult(
        source=request.group, orbit=tuple(orbit), point=request.point
    )


def compute_group_conjugacy_classes(
    request: GroupConjugacyClassesRequest,
) -> GroupConjugacyClassesResult:
    classes = native.group_conjugacy_classes(
        request.degree,
        [list(g) for g in request.generators],
    )
    return GroupConjugacyClassesResult._from_kernel(
        PermutationGroup(degree=request.degree, generators=request.generators),
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
    subgroups = native.subgroup_lattice(source)
    return GroupSubgroupLatticeResult._computed_from_kernel(request, tuple(subgroups))


S3_GENERATORS = {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]}
S3_STABILIZER_POINT_0 = {
    "group": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
    "point": 0,
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="finite_abelian_group.exact_factorization.compute",
        title="Exact finite abelian group factorization",
        description="Normalize two bounded integer-vector factors in a declared product "
        "of cyclic groups, exhaustively count every sum representation, and "
        "decide whether every group element has exactly one representation.",
        request_type=FiniteAbelianGroupFactorizationRequest,
        result_type=FiniteAbelianGroupFactorizationResult,
        run=compute_finite_abelian_group_factorization,
        tags=(
            "group",
            "finite-abelian-group",
            "cyclic-product",
            "factorization",
            "exact",
        ),
        examples=(
            OperationExample(
                name="z2_times_z4_transversal",
                description="Verify eight representatives form a complete transversal.",
                input={
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
    MathTool(
        operation_id="finite_abelian_group.spectral_pair.decide",
        title="Decide an exact finite-Abelian spectral pair",
        description="Decide whether a canonical residue-tuple frequency set is a spectrum "
        "of a point set in an explicit product of cyclic groups under the "
        "positive product dual pairing.",
        request_type=FiniteAbelianSpectralPairRequest,
        result_type=FiniteAbelianSpectralPairResult,
        run=compute_finite_abelian_spectral_pair,
        tags=("harmonic-analysis", "finite-abelian-group", "spectral-pair", "exact"),
        examples=(
            OperationExample(
                name="z4_even_pair",
                description="Decide the two-point spectral pair A={0,2}, Lambda={0,1} in Z/4.",
                input={
                    "source": {
                        "group": {"moduli": [4]},
                        "points": [[0], [2]],
                        "frequencies": [[0], [1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_abelian_group.character_sum_interval_profile.compute",
        title="Compute exact finite-Abelian character sums on labelled intervals",
        description="For one explicit finite sequence of elements of a finite product "
        "of cyclic groups, a duplicate-free frequency set, and a duplicate-free list "
        "of half-open index intervals [a,b), return every exact character sum "
        "S(lambda;a,b)=sum_{a<=t<b} chi_lambda(x_t) as a canonical dense remainder "
        "modulo the group-exponent cyclotomic polynomial. The pairing is the "
        "positive product dual pairing chi_lambda(a)=exp(2*pi*i*sum lambda_j a_j/m_j). "
        "The sequence order and repetitions are retained; each frequency-interval "
        "cell is reduced modulo Phi_N with N=lcm(m_j) and zero is the all-zero remainder.",
        request_type=FiniteAbelianCharacterSumIntervalProfileRequest,
        result_type=FiniteAbelianCharacterSumIntervalProfileResult,
        run=compute_finite_abelian_character_sum_interval_profile,
        tags=(
            "harmonic-analysis",
            "finite-abelian-group",
            "character-sum",
            "interval",
            "cyclotomic",
            "exact",
        ),
        examples=(
            OperationExample(
                name="z4_labelled_sequence_two_intervals",
                description="Exact sums for G=Z/4, labelled sequence (0,1,2,3), frequencies 0 and 1, intervals [0,4) and [1,3); the second interval separates the labelled sums from a set transform.",
                input={
                    "source": {
                        "group": {"moduli": [4]},
                        "sequence": [[0], [1], [2], [3]],
                        "frequencies": [[0], [1]],
                        "intervals": [[0, 4], [1, 3]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group.order.compute",
        title="Compute the exact order of a finite permutation group",
        description="Compute the exact order of a permutation group given by generators via SymPy's Schreier-Sims algorithm.",
        request_type=PermutationGroup,
        result_type=GroupOrderResult,
        run=compute_group_order,
        tags=("group", "order", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_group_order_4",
                description="Compute C4's order; each generator must be a bijection of 0..degree-1.",
                input={
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group.element_order.compute",
        title="Compute the exact order of one permutation",
        description="Compute the order of one permutation element via SymPy's Permutation.order().",
        request_type=GroupElementOrderRequest,
        result_type=GroupElementOrderResult,
        run=compute_element_order,
        tags=("group", "element-order", "permutation", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_order",
                description="Compute the 4-cycle's order; its generator must be a bijection of 0..degree-1.",
                input={
                    "degree": 4,
                    "generator": [1, 2, 3, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group.orbit.compute",
        title="Compute the orbit of a point under a permutation group",
        description="Compute the orbit of a point under a permutation group via SymPy's "
        "PermutationGroup.orbit(). The request takes the canonical "
        "permutation-group value, so a previous result's stabilizer or order "
        "group feeds this operation unchanged.",
        request_type=GroupOrbitRequest,
        result_type=GroupOrbitResult,
        run=compute_group_orbit,
        tags=("group", "orbit", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_orbit",
                description="Compute point 0's orbit of the group generated by (1,2,3,0); the group is the canonical value and points lie in 0..degree-1.",
                input={
                    "group": {
                        "degree": 4,
                        "generators": [[1, 2, 3, 0]],
                    },
                    "point": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group.conjugacy_classes.compute",
        title="Compute conjugacy classes of a permutation group",
        description="Given a permutation group by generators, return its conjugacy classes "
        "(the partition into conjugacy classes) as permutation array forms, "
        "using SymPy. Each class lists the elements conjugate to a "
        "representative; class sizes are orbit sizes under conjugation. "
        "Classes are canonically ordered (members sorted, classes sorted by "
        "smallest member), so the same group always yields an identical "
        "result. The generated group must have order at most "
        f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER}; larger "
        "groups are rejected before enumeration.",
        request_type=GroupConjugacyClassesRequest,
        result_type=GroupConjugacyClassesResult,
        run=compute_group_conjugacy_classes,
        tags=("group", "permutation", "conjugacy", "exact"),
        examples=(
            OperationExample(
                name="s3_conjugacy_classes",
                description=(
                    "Conjugacy classes of S3 (generators (1,2,0) and (1,0,2)); "
                    "S3 has three classes of sizes 1, 2, 3 (identity, 3-cycles, "
                    "transpositions). Generators must be permutations of 0..n-1."
                ),
                input=S3_GENERATORS,
            ),
        ),
    ),
    MathTool(
        operation_id="group.stabilizer.compute",
        title="Compute the stabilizer of a point in a permutation group",
        description="Given a permutation group as the canonical group value and a point, "
        "return the point stabilizer subgroup as a canonical permutation-group "
        "value (elements fixing the point) using SymPy's stabilizer "
        "computation. The request accepts that canonical value directly, so a "
        "previous result's `stabilizer` subgroup chains into the next request "
        "unchanged. By the orbit-stabilizer theorem, "
        "|G| = |orbit(point)| * |stabilizer(point)|, composable with "
        "group.order.compute and group.orbit.compute.",
        request_type=GroupStabilizerRequest,
        result_type=GroupStabilizerResult,
        run=compute_group_stabilizer,
        tags=("group", "permutation", "stabilizer", "orbit-stabilizer", "exact"),
        examples=(
            OperationExample(
                name="s3_stabilizer_of_0",
                description=(
                    "Stabilizer of point 0 in S3 (generators (1,2,0) and "
                    "(1,0,2)); order 2 and orbit-stabilizer gives 6 = 3 * 2. "
                    "Pass a previous result's `stabilizer` as `group` to chain."
                ),
                input=S3_STABILIZER_POINT_0,
            ),
        ),
    ),
    MathTool(
        operation_id="group.subgroup_lattice.compute",
        title="Enumerate all subgroups of a bounded permutation group",
        description="Enumerate all subgroups of a bounded permutation group via SymPy. "
        "Each subgroup is returned with its generators and order. Bounded "
        "to groups of order at most 64; traversal exhaustion is an execution "
        "failure and establishes no subgroup lattice.",
        request_type=GroupSubgroupLatticeRequest,
        result_type=GroupSubgroupLatticeResult,
        run=compute_subgroup_lattice,
        tags=("group", "subgroup", "permutation", "exact"),
        examples=(
            OperationExample(
                name="c4_subgroups",
                description="Enumerate all subgroups of C4; generators must be bijections.",
                input={
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
