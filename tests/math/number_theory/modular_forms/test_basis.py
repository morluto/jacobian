"""Exact basis and coordinate contracts for level-one modular forms."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms._models import ModularFormBasisRequest
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    BASIS_ID,
    modular_form_basis_q_expansions,
    modular_form_coordinates_hecke,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_u2,
    modular_form_coordinates_v2,
    modular_form_coordinates_v3,
    modular_form_coordinates_v_degeneracy,
    modular_form_operator_image,
    modular_form_operator_image_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_LEVEL_ONE_BASIS_PRECISION,
    ModularFormCoordinates,
    ModularFormOperatorImage,
    ModularFormOperatorImagePrefix,
    ModularFormSpace,
)


def _space(weight: int, kind: str = "M") -> ModularFormSpace:
    return ModularFormSpace(level=1, weight=weight, kind=kind)


def _coordinate_values(*values: tuple[int, int]) -> tuple[CanonicalRational, ...]:
    return tuple(
        CanonicalRational(num=numerator, den=denominator)
        for numerator, denominator in values
    )


def test_weight_twelve_basis_has_the_exact_e4_cubed_and_e6_squared_prefixes() -> None:
    basis = modular_form_basis_q_expansions(_space(12), 3)

    assert basis.basis_id == BASIS_ID
    assert tuple(element.label for element in basis.elements) == ("E4^3", "E6^2")
    assert tuple(
        tuple(c.as_fraction() for c in element.expansion.q_expansion.coefficients)
        for element in basis.elements
    ) == (
        (Fraction(1), Fraction(720), Fraction(179_280)),
        (Fraction(1), Fraction(-1_008), Fraction(220_752)),
    )
    # The first two Fourier coefficients give a nonzero determinant, so the
    # two vectors are independent; the exact dimension is two.
    assert 1 * -1_008 - 1 * 720 == -1_728
    assert len(basis.elements) == 2
    assert type(basis).model_validate_json(basis.model_dump_json()) == basis


def test_cusp_basis_is_delta_times_the_inner_weight_basis() -> None:
    basis = modular_form_basis_q_expansions(_space(12, "S"), 4)

    assert tuple(element.label for element in basis.elements) == ("Delta*1",)
    assert tuple(
        coefficient.as_fraction()
        for coefficient in basis.elements[0].expansion.q_expansion.coefficients
    ) == (Fraction(0), Fraction(1), Fraction(-24), Fraction(252))


def test_zero_dimensional_cusp_space_has_an_empty_basis_and_only_zero_form() -> None:
    space = _space(10, "S")
    basis = modular_form_basis_q_expansions(space, 4)
    zero = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=(),
    )
    assert basis.elements == ()
    assert ModularFormCoordinates.model_validate(zero.model_dump()) == zero
    assert (
        tuple(
            coefficient.as_fraction()
            for coefficient in modular_form_coordinates_q_expansion(
                zero, 4
            ).q_expansion.coefficients
        )
        == (Fraction(0),) * 4
    )


def test_v3_reconstructs_level_one_form_in_gamma0_three_coordinates() -> None:
    e4 = ModularFormCoordinates(
        space=_space(4),
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )

    image = modular_form_coordinates_v3(e4)
    assert image.space.level == 3
    assert image.space.weight == 4
    assert image.basis_id == "gamma0-three-weight-2-4-6-hypersurface-v1"

    # E4(q^3) has constant coefficient 1 and q coefficient 0. The Gamma0(3)
    # Sturm bound is 1 at weight 4, so these coefficients determine the form.
    expansion = modular_form_coordinates_q_expansion(image, 2)
    assert tuple(c.as_fraction() for c in expansion.q_expansion.coefficients) == (
        Fraction(1),
        Fraction(0),
    )


def test_v3_preserves_zero_and_weight_zero_identity() -> None:
    zero = ModularFormCoordinates(
        space=_space(4), basis_id=BASIS_ID, coordinates=_coordinate_values((0, 1))
    )
    assert all(
        coordinate.as_fraction() == 0
        for coordinate in modular_form_coordinates_v3(zero).coordinates
    )

    constant = ModularFormCoordinates(
        space=_space(0), basis_id=BASIS_ID, coordinates=_coordinate_values((1, 1))
    )
    constant_image = modular_form_coordinates_v3(constant)
    assert constant_image.space.level == 3
    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_q_expansion(
            constant_image, 1
        ).q_expansion.coefficients
    ) == (Fraction(1),)


def test_v3_is_discoverable() -> None:
    assert "modular_form.coordinates.v3.apply" in {tool.operation_id for tool in TOOLS}


def test_rational_coordinates_construct_an_arbitrary_global_form() -> None:
    space = _space(12)
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((2, 3), (-1, 3)),
    )

    expansion = modular_form_coordinates_q_expansion(form, 3)

    assert expansion.space == space
    assert expansion.basis_id == BASIS_ID
    assert tuple(c.as_fraction() for c in expansion.q_expansion.coefficients) == (
        Fraction(1, 3),
        Fraction(816),
        Fraction(45_936),
    )


def test_coordinates_remain_bound_to_their_space_and_basis() -> None:
    space = _space(12)
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 2), (4, 5)),
    )
    same_form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 2), (4, 5)),
    )
    different_form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 2), (3, 5)),
    )

    assert form == same_form
    assert form != different_form
    with pytest.raises(ValidationError):
        ModularFormCoordinates(
            space=space,
            basis_id="another-basis",
            coordinates=(),
        )
    other_space = ModularFormCoordinates(
        space=_space(10),
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    assert form != other_space


def test_basis_rejects_levels_and_weights_outside_its_reviewed_family() -> None:
    with pytest.raises(OperationDomainValidationError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=5, weight=12, kind="M"), 4
        )
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_basis_q_expansions(_space(122), 4)
    with pytest.raises(ValidationError):
        ModularFormBasisRequest.model_validate(
            {
                "space": _space(12).model_dump(),
                "precision": MAX_LEVEL_ONE_BASIS_PRECISION + 1,
            }
        )


def test_upper_admitted_weight_and_precision_are_computable() -> None:
    basis = modular_form_basis_q_expansions(_space(120), MAX_LEVEL_ONE_BASIS_PRECISION)

    assert len(basis.elements) == 11
    assert all(
        element.expansion.q_expansion.truncation_order == MAX_LEVEL_ONE_BASIS_PRECISION
        for element in basis.elements
    )


def test_hecke_on_delta_returns_exact_rational_eigenvalue_coordinates() -> None:
    space = _space(12, "S")
    delta = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )

    image = modular_form_coordinates_hecke(delta, 2)

    assert image.space == space
    assert tuple(value.as_fraction() for value in image.coordinates) == (Fraction(-24),)
    # Independent q-expansion check: T_2 Delta has q coefficient tau(2)=-24.
    prefix = modular_form_coordinates_q_expansion(image, 2)
    assert tuple(c.as_fraction() for c in prefix.q_expansion.coefficients) == (
        Fraction(0),
        Fraction(-24),
    )


def test_hecke_coordinate_matrix_matches_direct_coefficients_through_sturm() -> None:
    space = _space(12)
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((2, 3), (-1, 3)),
    )
    image = modular_form_coordinates_hecke(form, 2)
    actual = modular_form_coordinates_q_expansion(image, 2)
    source = modular_form_coordinates_q_expansion(form, 3)
    # The Sturm integer is one at weight 12; compute T_2 directly from its
    # defining coefficient formula, without using the coordinate operator.
    a0, a1, a2 = tuple(c.as_fraction() for c in source.q_expansion.coefficients)
    assert tuple(c.as_fraction() for c in actual.q_expansion.coefficients) == (
        2_049 * a0,
        a2,
    )
    assert a0 == Fraction(1, 3)
    assert a1 == Fraction(816)
    assert a2 == Fraction(45_936)


def test_hecke_preflights_source_precision_from_the_sturm_bound() -> None:
    space = _space(12, "S")
    delta = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )

    # B=1, so T_128 would require source order 129, beyond the basis kernel's
    # admitted precision. Rejection happens before coefficient construction.
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_hecke(delta, 128)


def test_gamma0_four_trivial_hecke_uses_sturm_reconstruction_and_preserves_parent() -> (
    None
):
    space = ModularFormSpace(level=4, weight=2, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=_coordinate_values((2, 3), (-1, 4)),
    )

    image = modular_form_coordinates_hecke(form, 3)

    assert image.space == space
    assert image.basis_id == form.basis_id
    source = modular_form_coordinates_q_expansion(form, 4)
    actual = modular_form_coordinates_q_expansion(image, 2)
    source_coefficients = tuple(
        value.as_fraction() for value in source.q_expansion.coefficients
    )
    # Independent T_3 coefficient formula at the level-four Sturm bound B=1.
    assert tuple(value.as_fraction() for value in actual.q_expansion.coefficients) == (
        4 * source_coefficients[0],
        source_coefficients[3],
    )


def test_gamma0_four_b4_is_a_t3_eigenform() -> None:
    space = ModularFormSpace(level=4, weight=2, kind="M")
    b4 = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=_coordinate_values((1, 1), (0, 1)),
    )

    image = modular_form_coordinates_hecke(b4, 3)

    assert tuple(value.as_fraction() for value in image.coordinates) == (
        Fraction(4),
        Fraction(0),
    )


def test_gamma0_four_hecke_rejects_indices_not_coprime_to_level() -> None:
    space = ModularFormSpace(level=4, weight=2, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=_coordinate_values((1, 1), (0, 1)),
    )

    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_hecke(form, 2)


def test_gamma0_four_hecke_preflights_source_precision_before_expansion() -> None:
    space = ModularFormSpace(level=4, weight=2, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=_coordinate_values((1, 1), (0, 1)),
    )

    # B=1 and T_129 needs coefficients through q^129, beyond basis precision.
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_hecke(form, 129)


def test_gamma0_four_trivial_u2_matches_direct_prefix_and_preserves_parent() -> None:
    space = ModularFormSpace(level=4, weight=2, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=_coordinate_values((2, 3), (-1, 4)),
    )

    image = modular_form_coordinates_u2(form)

    assert image.space == space
    assert image.basis_id == form.basis_id
    source = modular_form_coordinates_q_expansion(form, 3)
    actual = modular_form_coordinates_q_expansion(image, 2)
    assert tuple(value.as_fraction() for value in actual.q_expansion.coefficients) == (
        source.q_expansion.coefficients[0].as_fraction(),
        source.q_expansion.coefficients[2].as_fraction(),
    )


def test_v2_from_level_one_constants_and_positive_weight_into_gamma0_two() -> None:
    constant_space = _space(0)
    constant = ModularFormCoordinates(
        space=constant_space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    constant_image = modular_form_coordinates_v2(constant)
    assert constant_image.space == ModularFormSpace(level=2, weight=0, kind="M")
    assert constant_image.basis_id == "gamma0-two-weight-2-4-monomials-v1"
    assert tuple(value.as_fraction() for value in constant_image.coordinates) == (
        Fraction(1),
    )

    space = _space(8)
    e4_squared = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    image = modular_form_coordinates_v2(e4_squared)
    assert image.space == ModularFormSpace(level=2, weight=8, kind="M")
    assert image.basis_id == "gamma0-two-weight-2-4-monomials-v1"

    source = modular_form_coordinates_q_expansion(e4_squared, 2)
    actual = modular_form_coordinates_q_expansion(image, 3)
    source_coefficients = tuple(
        value.as_fraction() for value in source.q_expansion.coefficients
    )
    # For B_2=2, source precision floor(B_2/2)+1=2; the independent
    # degeneracy-map oracle reads a_0 at q^0 and a_1 at q^2.
    assert source_coefficients == (Fraction(1), Fraction(480))
    assert tuple(value.as_fraction() for value in actual.q_expansion.coefficients) == (
        source_coefficients[0],
        Fraction(0),
        source_coefficients[1],
    )


def test_v2_from_level_one_rejects_unsupported_weight_before_expansion() -> None:
    form = ModularFormCoordinates(
        space=_space(122),
        basis_id=BASIS_ID,
        coordinates=(),
    )

    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_v2(form)


def test_level_one_v2_catalog_example_runs_in_its_exact_target_space() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.v2.apply"
    )
    example = next(
        example
        for example in operation.examples
        if example.name == "v2_level_one_e4_squared"
    )
    request = operation.request_type.model_validate_json(json.dumps(example.input))

    image = operation.run(request)

    assert image.space == ModularFormSpace(level=2, weight=8, kind="M")
    assert image.basis_id == "gamma0-two-weight-2-4-monomials-v1"


def test_upper_weight_sturm_source_is_accepted_and_matches_direct_formula() -> None:
    space = _space(120)
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            *(CanonicalRational(num=0, den=1) for _ in range(10)),
        ),
    )
    image = modular_form_coordinates_hecke(form, 12)
    actual = modular_form_coordinates_q_expansion(image, 11)
    source = modular_form_coordinates_q_expansion(form, 121)
    source_coefficients = tuple(
        coefficient.as_fraction() for coefficient in source.q_expansion.coefficients
    )
    expected = []
    for m in range(11):
        expected.append(
            sum(
                (
                    Fraction(divisor**119)
                    * source_coefficients[12 * m // (divisor * divisor)]
                    for divisor in range(1, 13)
                    if 12 % divisor == 0 and m % divisor == 0
                ),
                Fraction(0),
            )
        )

    assert len(image.coordinates) == 11
    assert tuple(c.as_fraction() for c in actual.q_expansion.coefficients) == tuple(
        expected
    )


def test_u2_operator_image_retains_source_and_matches_independent_q_prefix() -> None:
    source_space = _space(12, "S")
    delta = ModularFormCoordinates(
        space=source_space,
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    image = modular_form_operator_image(delta, "U", 2)
    result = modular_form_operator_image_q_expansion(image, 5)
    source = modular_form_coordinates_q_expansion(delta, 9)
    source_coefficients = tuple(
        coefficient.as_fraction() for coefficient in source.q_expansion.coefficients
    )

    assert image.source_form == delta
    assert image.operator == "U"
    assert image.prime == 2
    assert image.codomain == ModularFormSpace(level=2, weight=12, kind="S")
    assert tuple(c.as_fraction() for c in result.q_expansion.coefficients) == tuple(
        source_coefficients[2 * index] for index in range(5)
    )
    assert tuple(c.as_fraction() for c in result.q_expansion.coefficients) == (
        Fraction(0),
        Fraction(-24),
        Fraction(-1_472),
        Fraction(-6_048),
        Fraction(84_480),
    )
    assert (
        ModularFormOperatorImage.model_validate_json(image.model_dump_json()) == image
    )
    assert (
        ModularFormOperatorImagePrefix.model_validate_json(result.model_dump_json())
        == result
    )


def test_v2_operator_image_dilates_exact_q_prefix() -> None:
    delta = ModularFormCoordinates(
        space=_space(12, "S"),
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    image = modular_form_operator_image(delta, "V", 2)
    result = modular_form_operator_image_q_expansion(image, 9)

    assert tuple(c.as_fraction() for c in result.q_expansion.coefficients) == (
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(-24),
        Fraction(0),
        Fraction(252),
        Fraction(0),
        Fraction(-1_472),
    )
    assert result.image == image
    assert result.image.codomain.level == 2


def test_operator_image_rejects_non_level_one_source() -> None:
    higher_level = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=4, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=_coordinate_values((1, 1), (0, 1)),
    )
    with pytest.raises(OperationDomainValidationError, match="level-one"):
        modular_form_operator_image(higher_level, "U", 3)


def test_operator_image_rejects_composite_prime_and_insufficient_source_order() -> None:
    delta = ModularFormCoordinates(
        space=_space(12, "S"),
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    with pytest.raises(OperationDomainValidationError):
        modular_form_operator_image(delta, "U", 4)

    image = modular_form_operator_image(delta, "U", 2)
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_operator_image_q_expansion(image, 65)


def test_operator_image_prefix_composes_for_empty_cusp_space() -> None:
    zero = ModularFormCoordinates(
        space=_space(10, "S"),
        basis_id=BASIS_ID,
        coordinates=(),
    )
    image = modular_form_operator_image(zero, "V", 3)
    result = modular_form_operator_image_q_expansion(image, 8)

    assert image.codomain == ModularFormSpace(level=3, weight=10, kind="S")
    assert (
        tuple(c.as_fraction() for c in result.q_expansion.coefficients)
        == (Fraction(0),) * 8
    )


def test_v_degeneracy_has_exact_parent_and_independent_sparse_expansion() -> None:
    source = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=4, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=_coordinate_values((1, 1), (0, 1)),
    )
    source_series = modular_form_coordinates_q_expansion(source, 2)
    result = modular_form_coordinates_v_degeneracy(source, 2)
    actual = modular_form_coordinates_q_expansion(result, 3)
    coefficients = tuple(
        c.as_fraction() for c in source_series.q_expansion.coefficients
    )
    assert result.space == ModularFormSpace(level=4, weight=4, kind="M")
    assert result.basis_id == "gamma0-four-weight-2-generators-v1"
    assert tuple(c.as_fraction() for c in actual.q_expansion.coefficients) == (
        coefficients[0],
        Fraction(0),
        coefficients[1],
    )
    assert (
        ModularFormCoordinates.model_validate_json(result.model_dump_json()) == result
    )


def test_v_degeneracy_composes_through_exact_intermediate_parent() -> None:
    source = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=4, kind="M"),
        basis_id=BASIS_ID,
        coordinates=_coordinate_values((1, 1)),
    )
    via_two = modular_form_coordinates_v_degeneracy(source, 2)
    composed = modular_form_coordinates_v_degeneracy(via_two, 2)
    direct = modular_form_coordinates_v_degeneracy(source, 4)
    assert composed == direct


def test_v_degeneracy_rejects_target_level_outside_basis_envelope() -> None:
    source = ModularFormCoordinates(
        space=ModularFormSpace(level=4_000, weight=4, kind="M"),
        basis_id="gamma0-rational-gamma0-sturm-rref-v1",
        coordinates=_coordinate_values((1, 1)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_v_degeneracy(source, 3)


def test_v_degeneracy_catalog_example_executes_in_advertised_parent() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.v_degeneracy.apply"
    )
    example = operation.examples[0]
    request = operation.request_type.model_validate_json(json.dumps(example.input))
    result = operation.run(request)
    assert result.space == ModularFormSpace(level=4, weight=4, kind="M")
