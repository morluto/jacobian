"""Exact contract tests for finite-coset crossed-product multiplication."""

from __future__ import annotations

from collections.abc import Iterable
from math import isqrt

import pytest

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.crossed_products._budget import MAX_CONVOLUTION_PAIRS
from jacobian.math.crossed_products._models import (
    CrossedProductMultiplyRequest,
    CrossedProductMultiplyResult,
)
from jacobian.math.crossed_products._operations import (
    compute_product,
)
from jacobian.math.crossed_products._tools import TOOLS
from jacobian.math.crossed_products.operations import multiply
from jacobian.math.crossed_products.values import (
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductPresentation,
    FiniteCosetCrossedProductTerm,
    _integer_matrix_vector_product,
)

Exponent = tuple[int, ...]
Support = set[Exponent]


def test_integer_matrix_vector_product_handles_zero_and_full_rank_actions() -> None:
    assert _integer_matrix_vector_product((), ()) == ()
    assert _integer_matrix_vector_product(((2, -1), (3, 4)), (5, -2)) == (12, 7)


def _c2_presentation(
    *,
    characteristic: int = 5,
    action: tuple[tuple[int, ...], ...] = ((-1,),),
    cocycle_square: tuple[int, ...] = (0,),
) -> FiniteCosetCrossedProductPresentation:
    dimension = len(action)
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(dimension))
        for row in range(dimension)
    )
    zero = (0,) * dimension
    return FiniteCosetCrossedProductPresentation(
        characteristic=characteristic,
        lattice_basis=tuple(f"t{position}" for position in range(dimension)),
        cosets=("e", "a"),
        identity_coset="e",
        quotient_multiplication=(("e", "a"), ("a", "e")),
        action_matrices=tuple(
            tuple(
                tuple(format_canonical_integer(entry) for entry in row)
                for row in matrix
            )
            for matrix in (identity, action)
        ),
        cocycle_table=tuple(
            tuple(
                tuple(format_canonical_integer(entry) for entry in vector)
                for vector in row
            )
            for row in ((zero, zero), (zero, cocycle_square))
        ),
    )


def _element(
    presentation: FiniteCosetCrossedProductPresentation,
    components: dict[str, Iterable[Exponent]],
    *,
    coefficient: int = 1,
) -> FiniteCosetCrossedProductElement:
    positions = {label: index for index, label in enumerate(presentation.cosets)}
    support = sorted(
        (
            positions[coset],
            exponents,
            coset,
        )
        for coset, exponents_set in components.items()
        for exponents in exponents_set
    )
    return FiniteCosetCrossedProductElement(
        presentation=presentation,
        terms=tuple(
            FiniteCosetCrossedProductTerm(
                coefficient=coefficient,
                coset=coset,
                exponents=tuple(
                    format_canonical_integer(exponent) for exponent in exponents
                ),
            )
            for _, exponents, coset in support
        ),
    )


def _multiply_f2_support(left: Support, right: Support) -> Support:
    result: Support = set()
    for first in left:
        for second in right:
            exponent = tuple(
                first_coordinate + second_coordinate
                for first_coordinate, second_coordinate in zip(
                    first, second, strict=True
                )
            )
            if exponent in result:
                result.remove(exponent)
            else:
                result.add(exponent)
    return result


def _shift(support: Support, shift: Exponent) -> Support:
    return {
        tuple(
            coordinate + displacement
            for coordinate, displacement in zip(exponent, shift, strict=True)
        )
        for exponent in support
    }


def _gardam_presentation() -> FiniteCosetCrossedProductPresentation:
    zero = ("0", "0", "0")
    return FiniteCosetCrossedProductPresentation(
        characteristic=2,
        lattice_basis=("x", "y", "z"),
        cosets=("1", "a", "b", "ab"),
        identity_coset="1",
        quotient_multiplication=(
            ("1", "a", "b", "ab"),
            ("a", "1", "ab", "b"),
            ("b", "ab", "1", "a"),
            ("ab", "b", "a", "1"),
        ),
        action_matrices=(
            (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")),
            (("1", "0", "0"), ("0", "-1", "0"), ("0", "0", "-1")),
            (("-1", "0", "0"), ("0", "1", "0"), ("0", "0", "-1")),
            (("-1", "0", "0"), ("0", "-1", "0"), ("0", "0", "1")),
        ),
        cocycle_table=(
            (zero, zero, zero, zero),
            (zero, ("1", "0", "0"), zero, ("1", "0", "0")),
            (
                zero,
                ("-1", "1", "-1"),
                ("0", "1", "0"),
                ("-1", "0", "-1"),
            ),
            (
                zero,
                ("0", "-1", "1"),
                ("0", "-1", "0"),
                ("0", "0", "1"),
            ),
        ),
    )


def _gardam_elements() -> tuple[
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductElement,
]:
    presentation = _gardam_presentation()
    one: Support = {(0, 0, 0)}
    x: Support = {(1, 0, 0)}
    y: Support = {(0, 1, 0)}
    z_inverse: Support = {(0, 0, -1)}
    p = _multiply_f2_support(_multiply_f2_support(one | x, one | y), one | z_inverse)
    q: Support = {(-1, -1, 0), (1, 0, 0), (0, -1, 1), (0, 0, 1)}
    r: Support = {(0, 0, 0), (1, 0, 0), (0, -1, 1), (1, 1, 1)}
    s: Support = {
        (0, 0, 0),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, -1),
        (0, -1, -1),
    }
    alpha = _element(presentation, {"1": p, "a": q, "b": r, "ab": s})

    def action_a(exponent: Exponent) -> Exponent:
        return exponent[0], -exponent[1], -exponent[2]

    action_a_p: Support = {action_a(term) for term in p}
    action_a_s: Support = {action_a(term) for term in s}
    inverse = _element(
        presentation,
        {
            "1": _shift(action_a_p, (-1, 0, 0)),
            "a": _shift(q, (-1, 0, 0)),
            "b": _shift(r, (0, -1, 0)),
            "ab": _shift(action_a_s, (0, 0, -1)),
        },
    )
    identity = _element(presentation, {"1": {(0, 0, 0)}})
    return alpha, inverse, identity


def _component_support_counts(
    element: FiniteCosetCrossedProductElement,
) -> tuple[int, ...]:
    return tuple(
        sum(term.coset == coset for term in element.terms)
        for coset in element.presentation.cosets
    )


def test_left_action_and_cocycle_define_basis_multiplication() -> None:
    presentation = _c2_presentation(action=((1,),), cocycle_square=(3,))
    left = _element(presentation, {"a": {(2,)}})
    right = _element(presentation, {"a": {(5,)}})

    assert multiply(left, right) == _element(presentation, {"e": {(10,)}})


def test_left_action_convention_is_noncommutative() -> None:
    presentation = _c2_presentation()
    first = _element(presentation, {"a": {(1,)}})
    second = _element(presentation, {"a": {(2,)}})

    assert multiply(first, second) == _element(presentation, {"e": {(-1,)}})
    assert multiply(second, first) == _element(presentation, {"e": {(1,)}})


def test_modular_convolution_cancels_to_parent_bound_zero() -> None:
    presentation = _c2_presentation(characteristic=2)
    value = _element(presentation, {"e": {(0,)}, "a": {(0,)}})

    product = multiply(value, value)

    assert product.presentation == presentation
    assert product.terms == ()


def test_zero_rank_specializes_to_a_finite_group_algebra() -> None:
    presentation = _c2_presentation(characteristic=3, action=(), cocycle_square=())
    value = _element(presentation, {"e": {()}, "a": {()}})

    assert multiply(value, value) == _element(
        presentation,
        {"e": {()}, "a": {()}},
        coefficient=2,
    )


def test_gardam_unit_and_inverse_replay_in_both_orders() -> None:
    alpha, inverse, identity = _gardam_elements()

    assert _component_support_counts(alpha) == (8, 4, 4, 5)
    assert _component_support_counts(inverse) == (8, 4, 4, 5)
    assert multiply(inverse, alpha) == identity
    assert multiply(alpha, inverse) == identity


def test_gardam_one_symbol_mutation_preserves_counts_but_breaks_identity() -> None:
    alpha, inverse, identity = _gardam_elements()
    mutated_terms = tuple(
        FiniteCosetCrossedProductTerm(
            coefficient=term.coefficient,
            coset=term.coset,
            exponents=("0", "1", "0")
            if term.coset == "a" and term.exponents == ("0", "0", "1")
            else term.exponents,
        )
        for term in alpha.terms
    )
    mutated = FiniteCosetCrossedProductElement(
        presentation=alpha.presentation,
        terms=tuple(
            sorted(
                mutated_terms,
                key=lambda term: (
                    alpha.presentation.cosets.index(term.coset),
                    tuple(int(value) for value in term.exponents),
                ),
            )
        ),
    )

    assert _component_support_counts(mutated) == (8, 4, 4, 5)
    assert multiply(mutated, inverse) != identity
    assert multiply(inverse, mutated) != identity


def test_wire_result_is_structural_and_preserves_the_operand_presentation() -> None:
    alpha, inverse, identity = _gardam_elements()
    result = compute_product(CrossedProductMultiplyRequest(left=alpha, right=inverse))
    assert result.product == identity

    payload = result.model_dump(mode="json")
    payload["product"]["terms"][0]["exponents"] = ["1", "0", "0"]
    claim = CrossedProductMultiplyResult.model_validate(payload)
    assert claim.left == alpha
    assert claim.right == inverse
    assert claim.product != identity


def test_compute_product_binds_fresh_kernel_output() -> None:
    alpha, inverse, identity = _gardam_elements()

    result = compute_product(CrossedProductMultiplyRequest(left=alpha, right=inverse))

    assert result.product == identity
    assert (result.left, result.right) == (alpha, inverse)


def test_deserialized_result_can_be_verified_explicitly() -> None:
    alpha, inverse, identity = _gardam_elements()
    payload = compute_product(
        CrossedProductMultiplyRequest(left=alpha, right=inverse)
    ).model_dump(mode="json")

    replayed = CrossedProductMultiplyResult.model_validate(payload)

    assert replayed.product == identity


def test_element_requires_unique_canonical_coset_and_exponent_order() -> None:
    presentation = _c2_presentation()
    term = FiniteCosetCrossedProductTerm(coefficient=1, coset="e", exponents=("0",))
    with pytest.raises(ValueError, match="unique"):
        FiniteCosetCrossedProductElement(presentation=presentation, terms=(term, term))
    with pytest.raises(ValueError, match="coset order"):
        FiniteCosetCrossedProductElement(
            presentation=presentation,
            terms=(
                FiniteCosetCrossedProductTerm(
                    coefficient=1, coset="a", exponents=("0",)
                ),
                term,
            ),
        )


def test_presentation_rejects_non_group_table() -> None:
    payload = _c2_presentation().model_dump(mode="json")
    payload["quotient_multiplication"][1][1] = "a"
    with pytest.raises(ValueError, match=r"inverse|associative"):
        FiniteCosetCrossedProductPresentation.model_validate(payload)


def test_presentation_requires_a_prime_characteristic() -> None:
    with pytest.raises(ValueError, match="characteristic must be prime"):
        _c2_presentation(characteristic=4)


def test_presentation_rejects_non_unimodular_action() -> None:
    payload = _c2_presentation().model_dump(mode="json")
    payload["action_matrices"][1][0][0] = "2"
    with pytest.raises(ValueError, match="unimodular"):
        FiniteCosetCrossedProductPresentation.model_validate(payload)


def test_presentation_rejects_unimodular_non_action() -> None:
    with pytest.raises(ValueError, match=r"rho\(qr\)"):
        _c2_presentation(
            action=((1, 1), (0, 1)),
            cocycle_square=(0, 0),
        )


def test_presentation_rejects_non_normalized_cocycle() -> None:
    payload = _c2_presentation().model_dump(mode="json")
    payload["cocycle_table"][0][1][0] = "1"
    with pytest.raises(ValueError, match="normalized"):
        FiniteCosetCrossedProductPresentation.model_validate(payload)


def test_presentation_rejects_cocycle_equation_failure() -> None:
    payload = _gardam_presentation().model_dump(mode="json")
    payload["cocycle_table"][1][1][0] = "2"
    with pytest.raises(ValueError, match="cocycle must satisfy"):
        FiniteCosetCrossedProductPresentation.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "oversized_value"),
    (
        ("quotient_multiplication", ["e"] * 17),
        ("action_matrices", [["1"] * 9]),
        ("cocycle_table", [["0"] * 9]),
    ),
)
def test_presentation_rejects_oversized_nested_rows(
    field: str, oversized_value: list[object]
) -> None:
    payload = _c2_presentation().model_dump(mode="json")
    payload[field][0] = oversized_value

    with pytest.raises(ValueError, match="at most"):
        FiniteCosetCrossedProductPresentation.model_validate(payload)


def test_request_rejects_pairwise_convolution_before_expansion() -> None:
    presentation = _c2_presentation()
    convolution_side = isqrt(MAX_CONVOLUTION_PAIRS) + 1
    left = _element(
        presentation,
        {"e": {(position,) for position in range(convolution_side)}},
    )
    right = _element(
        presentation,
        {"a": {(position,) for position in range(convolution_side)}},
    )

    request = CrossedProductMultiplyRequest(left=left, right=right)
    with pytest.raises(
        OperationDomainValidationError, match=rf"{MAX_CONVOLUTION_PAIRS}-pair"
    ):
        compute_product(request)


def test_request_rejects_mismatched_presentations_with_a_typed_error() -> None:
    left = _element(_c2_presentation(characteristic=3), {"e": {(0,)}})
    right = _element(_c2_presentation(characteristic=5), {"e": {(0,)}})

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_product(CrossedProductMultiplyRequest(left=left, right=right))

    assert exc_info.value.errors()[0]["type"] == (
        "crossed_product.presentation_mismatch"
    )


def test_request_rejects_scalar_work_before_expansion() -> None:
    dimension = 8
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(dimension))
        for row in range(dimension)
    )
    presentation = _c2_presentation(
        action=identity,
        cocycle_square=(0,) * dimension,
    )
    support = {(position,) + (0,) * (dimension - 1) for position in range(30)}
    left = _element(presentation, {"e": support})
    right = _element(presentation, {"a": support})

    request = CrossedProductMultiplyRequest(left=left, right=right)
    with pytest.raises(OperationDomainValidationError, match="scalar-work"):
        compute_product(request)


def test_request_rejects_predicted_exponent_growth_before_expansion() -> None:
    shear = 9_999_999_999_999_999
    presentation = _c2_presentation(action=((1, shear), (0, -1)), cocycle_square=(0, 0))
    left = _element(presentation, {"a": {(0, 0)}})
    right = _element(presentation, {"e": {(0, int("9" * 64))}})

    request = CrossedProductMultiplyRequest(left=left, right=right)
    with pytest.raises(
        OperationDomainValidationError, match="predicted product exponents"
    ):
        compute_product(request)


def test_owner_declares_only_the_admitted_atomic_operation() -> None:
    assert {tool.operation_id for tool in TOOLS} == {"crossed_product.multiply.compute"}


def test_published_example_is_valid_and_runs() -> None:
    (tool,) = TOOLS
    (invocation_example,) = tool.examples
    request = tool.request_type.model_validate(invocation_example.input)

    result = tool.run(request)

    assert result.product.terms == (
        FiniteCosetCrossedProductTerm(
            coefficient=1,
            coset="e",
            exponents=("-1",),
        ),
    )
