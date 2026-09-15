"""Finite semigroup operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_semigroups._models import (
    AdjoinIdentityRequest,
    AdjoinIdentityResult,
    AdjoinZeroRequest,
    AdjoinZeroResult,
    ElementPowerRequest,
    ElementPowerResult,
    GeneratedSubsemigroupRequest,
    GeneratedSubsemigroupResult,
    GreenRelationsRequest,
    GreenRelationsResult,
    IdealEnumerationRequest,
    IdealEnumerationResult,
    IdempotentsRequest,
    IdempotentsResult,
    KaroubiProjectionRequest,
    KaroubiProjectionResult,
    LocalStructureRequest,
    LocalStructureResult,
    NilpotentElementsRequest,
    NilpotentElementsResult,
    OppositeRequest,
    OppositeResult,
    PowerProfileRequest,
    PowerProfileResult,
    PrincipalIdealsRequest,
    PrincipalIdealsResult,
    ProductRequest,
    ProductResult,
    ReesQuotientRequest,
    ReesQuotientResult,
    RegularElementsRequest,
    RegularElementsResult,
)
from jacobian.math.finite_semigroups.operations import (
    adjoin_identity,
    adjoin_zero,
    element_power,
    generated_subsemigroup,
    green_relations,
    ideal_enumeration,
    idempotents,
    karoubi_projection,
    local_structure,
    nilpotent_elements,
    opposite_semigroup,
    power_profile,
    principal_ideals,
    product_semigroup,
    rees_quotient,
    regular_elements,
)


def _run_power_profile(request: PowerProfileRequest) -> PowerProfileResult:
    return power_profile(request.semigroup, request.element)


def _run_generated_subsemigroup(
    request: GeneratedSubsemigroupRequest,
) -> GeneratedSubsemigroupResult:
    return generated_subsemigroup(request.semigroup, request.generators)


def _run_element_power(request: ElementPowerRequest) -> ElementPowerResult:
    return element_power(request.semigroup, request.element, request.exponent)


def _run_idempotents(request: IdempotentsRequest) -> IdempotentsResult:
    return idempotents(request.semigroup)


def _run_regular_elements(request: RegularElementsRequest) -> RegularElementsResult:
    return regular_elements(request.semigroup)


def _run_nilpotent_elements(
    request: NilpotentElementsRequest,
) -> NilpotentElementsResult:
    return nilpotent_elements(request.semigroup, request.zero)


def _run_principal_ideals(request: PrincipalIdealsRequest) -> PrincipalIdealsResult:
    return principal_ideals(request.semigroup, request.elements)


def _run_green_relations(request: GreenRelationsRequest) -> GreenRelationsResult:
    return green_relations(request.semigroup)


def _run_local_structure(request: LocalStructureRequest) -> LocalStructureResult:
    return local_structure(request.semigroup)


def _run_ideal_enumeration(
    request: IdealEnumerationRequest,
) -> IdealEnumerationResult:
    return ideal_enumeration(request.semigroup)


def _run_opposite(request: OppositeRequest) -> OppositeResult:
    return opposite_semigroup(request.semigroup)


def _run_product(request: ProductRequest) -> ProductResult:
    return product_semigroup(request.left, request.right)


def _run_adjoin_identity(request: AdjoinIdentityRequest) -> AdjoinIdentityResult:
    return adjoin_identity(request.semigroup)


def _run_adjoin_zero(request: AdjoinZeroRequest) -> AdjoinZeroResult:
    return adjoin_zero(request.semigroup)


def _run_rees_quotient(request: ReesQuotientRequest) -> ReesQuotientResult:
    return rees_quotient(request.semigroup, request.ideal)


def _run_karoubi(request: KaroubiProjectionRequest) -> KaroubiProjectionResult:
    return karoubi_projection(request.semigroup)


# The cyclic group Z/3Z as a semigroup (which is also a group)
# Elements: 0, 1, 2 with addition mod 3
_SEMIGROUP = {
    "elements": ["0", "1", "2"],
    "multiplication": [
        ["0", "1", "2"],
        ["1", "2", "0"],
        ["2", "0", "1"],
    ],
}

_NILPOTENT_CHAIN = {
    "elements": ["0", "a", "b", "c"],
    "multiplication": [
        ["0", "0", "0", "0"],
        ["0", "b", "0", "0"],
        ["0", "0", "0", "0"],
        ["0", "0", "0", "0"],
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="semigroup.element.power_profile.compute",
        title="Compute the power profile of a semigroup element",
        description="Compute the sequence a, a^2, a^3, ... until the first repeat, "
        "determining the one-based index, period, idempotent limit, and "
        "complete cyclic subsemigroup of one element in a finite semigroup.",
        request_type=PowerProfileRequest,
        result_type=PowerProfileResult,
        run=_run_power_profile,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="cyclic_group_z3",
                description="Compute the power profile of element 1 in Z/3Z; "
                "the semigroup must be associative.",
                input={
                    "semigroup": _SEMIGROUP,
                    "element": "1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.generated_subsemigroup.compute",
        title="Compute the subsemigroup generated by a set of elements",
        description="Compute the closure of a set of generators under the semigroup "
        "multiplication, returning all elements in the generated "
        "subsemigroup.",
        request_type=GeneratedSubsemigroupRequest,
        result_type=GeneratedSubsemigroupResult,
        run=_run_generated_subsemigroup,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="generate_from_1_in_z3",
                description="Compute the subsemigroup generated by {1} in Z/3Z; "
                "generators must be in the semigroup.",
                input={
                    "semigroup": _SEMIGROUP,
                    "generators": ["1"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.element.power.compute",
        title="Compute the power of a semigroup element",
        description="Compute the exact iterated product element^exponent for a positive "
        "exponent in a finite semigroup using its finite eventual period.",
        request_type=ElementPowerRequest,
        result_type=ElementPowerResult,
        run=_run_element_power,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="power_2_in_cyclic_group",
                description="Compute 1^2 in Z/3Z.",
                input={
                    "semigroup": _SEMIGROUP,
                    "element": "1",
                    "exponent": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.idempotents.compute",
        title="Find the idempotent elements of a finite semigroup",
        description="Return every element e with e*e = e in a finite semigroup.",
        request_type=IdempotentsRequest,
        result_type=IdempotentsResult,
        run=_run_idempotents,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="idempotents_of_z3",
                description="Find the idempotents of Z/3Z; only the zero element is idempotent.",
                input={
                    "semigroup": _SEMIGROUP,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.principal_ideals.compute",
        title="Compute principal ideals in a finite semigroup",
        description="For each requested element compute its principal two-sided ideal S^1 a S^1.",
        request_type=PrincipalIdealsRequest,
        result_type=PrincipalIdealsResult,
        run=_run_principal_ideals,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="ideal_in_z3",
                description="Compute the principal ideal of 1 in Z/3Z.",
                input={
                    "semigroup": _SEMIGROUP,
                    "elements": ["1"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.regular_elements.compute",
        title="Find regular elements of a finite semigroup",
        description=(
            "Return source-ordered (a, x) rows for every regular element, using "
            "the first declared witness x satisfying a*x*a=a."
        ),
        request_type=RegularElementsRequest,
        result_type=RegularElementsResult,
        run=_run_regular_elements,
        tags=("algebra", "semigroup", "regular", "exact"),
        examples=(
            OperationExample(
                name="regular_elements_z3",
                description="Find regular elements in Z/3Z; the semigroup must be associative.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.nilpotent_elements.compute",
        title="Find nilpotent elements relative to a zero",
        description=(
            "Return source-ordered (a, k) rows where k is the least positive "
            "exponent with a^k equal to the supplied absorbing zero; the zero "
            "must absorb every source element on both sides."
        ),
        request_type=NilpotentElementsRequest,
        result_type=NilpotentElementsResult,
        run=_run_nilpotent_elements,
        tags=("algebra", "semigroup", "nilpotent", "exact"),
        examples=(
            OperationExample(
                name="nilpotents_in_absorbing_chain",
                description=(
                    "Find elements and least exponents reaching zero in a nilpotent "
                    "chain; the supplied zero must be absorbing."
                ),
                input={"semigroup": _NILPOTENT_CHAIN, "zero": "0"},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.green_relations.compute",
        title="Compute Green relations of a finite semigroup",
        description="Compute the Green relations L, R, H, D, and J of a finite semigroup. "
        "L relates elements with the same principal left ideal, R with the "
        "same principal right ideal, H = L ∩ R, D = L ∨ R (join), and J is "  # noqa: RUF001
        "the two-sided Green relation defined by principal two-sided ideals.",
        request_type=GreenRelationsRequest,
        result_type=GreenRelationsResult,
        run=_run_green_relations,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="green_relations_z3",
                description="Compute the Green relations of Z/3Z.",
                input={
                    "semigroup": _SEMIGROUP,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.local_structure.compute",
        title="Compute units, local monoids, and maximal subgroups",
        description="Return the identity (if any), the unit group, one local monoid per "
        "idempotent with its maximal subgroup, and the minimal ideal; all outputs "
        "are complete functions of the supplied table.",
        request_type=LocalStructureRequest,
        result_type=LocalStructureResult,
        run=_run_local_structure,
        tags=("algebra", "semigroup", "exact", "complete"),
        examples=(
            OperationExample(
                name="local_structure_z3",
                description="Local structure of Z/3Z; the semigroup must be associative.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.ideals.subsemigroups.enumerate",
        title="Enumerate ideals and subsemigroups",
        description="Enumerate all two-sided ideals and subsemigroups under an honest "
        "admission bound (at most 12 elements); every subset is checked exactly.",
        request_type=IdealEnumerationRequest,
        result_type=IdealEnumerationResult,
        run=_run_ideal_enumeration,
        tags=("algebra", "semigroup", "exact", "complete"),
        examples=(
            OperationExample(
                name="enumerate_z3",
                description="Enumerate ideals of Z/3Z; enumeration is admitted for small tables.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.opposite.compute",
        title="Compute the opposite semigroup",
        description="Transpose the multiplication table; the carrier labels are preserved "
        "and associativity is re-established.",
        request_type=OppositeRequest,
        result_type=OppositeResult,
        run=_run_opposite,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="opposite_z3",
                description="Opposite of Z/3Z; the carrier is preserved.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.product.compute",
        title="Compute a direct product of semigroups",
        description="Return the componentwise product with explicit left/right projections; "
        "the product size must fit the 50-element bound.",
        request_type=ProductRequest,
        result_type=ProductResult,
        run=_run_product,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="product_z3_z3",
                description="Product of Z/3Z with itself; both factors must be associative.",
                input={"left": _SEMIGROUP, "right": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.adjoin_identity.compute",
        title="Adjoin an identity element",
        description="Adjoin a fresh two-sided identity with the inclusion embedding; "
        "the fresh label avoids the existing carrier.",
        request_type=AdjoinIdentityRequest,
        result_type=AdjoinIdentityResult,
        run=_run_adjoin_identity,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="adjoin_identity_z3",
                description="Adjoin an identity to Z/3Z; the source must be associative.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.adjoin_zero.compute",
        title="Adjoin an absorbing zero",
        description="Adjoin a fresh absorbing zero with the inclusion embedding; "
        "the fresh label avoids the existing carrier.",
        request_type=AdjoinZeroRequest,
        result_type=AdjoinZeroResult,
        run=_run_adjoin_zero,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="adjoin_zero_z3",
                description="Adjoin a zero to Z/3Z; the source must be associative.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.rees_quotient.compute",
        title="Compute a Rees quotient by an ideal",
        description="Collapse a two-sided ideal to a single zero class with the explicit "
        "projection; the ideal must be two-sided and use declared order.",
        request_type=ReesQuotientRequest,
        result_type=ReesQuotientResult,
        run=_run_rees_quotient,
        tags=("algebra", "semigroup", "exact"),
        examples=(
            OperationExample(
                name="rees_collapse_z3",
                description="Collapse the whole of Z/3Z; the ideal must be two-sided.",
                input={"semigroup": _SEMIGROUP, "ideal": ["0", "1", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="semigroup.karoubi_projection.compute",
        title="Project a semigroup onto its Karoubi envelope",
        description="Return the idempotent-splitting finite category with objects as "
        "idempotents and morphisms f*s*e triples; composition replays the semigroup.",
        request_type=KaroubiProjectionRequest,
        result_type=KaroubiProjectionResult,
        run=_run_karoubi,
        tags=("algebra", "semigroup", "category", "exact"),
        examples=(
            OperationExample(
                name="karoubi_z3",
                description="Karoubi envelope of Z/3Z; the source must be associative.",
                input={"semigroup": _SEMIGROUP},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
