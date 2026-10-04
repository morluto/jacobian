"""Global equality across exact character spaces and field embeddings."""

from __future__ import annotations

from itertools import product
from math import gcd
from typing import Literal

import pytest
from tests.math.number_theory.modular_forms._typing import basis_id as typed_basis_id
from tests.math.number_theory.modular_forms._typing import rational

from jacobian._exact import CanonicalRational
from jacobian._execution import current_request_execution
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
)
from jacobian.math.number_theory.modular_forms.global_equality import (
    CyclotomicFieldEmbedding,
)
from jacobian.math.number_theory.modular_forms.global_equality.operations import (
    _map_element,
    modular_form_coordinates_global_equal,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_SOURCE_FIELD = RationalCyclotomicField(order=6)
_TARGET_FIELD = RationalCyclotomicField(order=12)


def _inflated_character(level: int, coordinate: int) -> DirichletCharacter:
    source = dirichlet_character(character_group(13), (coordinate,))
    target_group = character_group(level)
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        candidate = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source, residue).value
            for residue in range(level)
            if gcd(residue, level) == 1
        ):
            return candidate
    raise AssertionError("character inflation fixture was not found")


def _space(
    level: int,
    coordinate: int = 4,
    kind: Literal["M", "S"] = "M",
    weight: int = 2,
) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=weight,
        kind=kind,
        character=_inflated_character(level, coordinate),
        coefficient_domain=_SOURCE_FIELD,
    )


def _element(constant: int = 0, zeta: int = 0) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=_SOURCE_FIELD,
        coefficients_ascending=(
            CanonicalRational(num=constant, den=1),
            CanonicalRational(num=zeta, den=1),
        ),
    )


def _form(space: ModularFormSpace, first_coordinate: int = 0) -> ModularFormCoordinates:
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        _admit_coordinate_space,
    )

    context = _admit_coordinate_space(space)
    dimension = context.dimension
    coordinates = tuple(
        _element(int(index == 0) if first_coordinate else 0)
        for index in range(dimension)
    )
    return ModularFormCoordinates(
        space=space,
        basis_id=typed_basis_id(context.basis_id),
        coordinates=coordinates,
    )


def _basis_form(space: ModularFormSpace, basis_index: int) -> ModularFormCoordinates:
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        _admit_coordinate_space,
    )

    context = _admit_coordinate_space(space)
    assert 0 <= basis_index < context.dimension
    return ModularFormCoordinates(
        space=space,
        basis_id=typed_basis_id(context.basis_id),
        coordinates=tuple(
            _element(int(index == basis_index)) for index in range(context.dimension)
        ),
    )


def _embedding() -> CyclotomicFieldEmbedding:
    # The inclusion Q(zeta_6) -> Q(zeta_12) sends zeta_6 to zeta_12^2.
    image = RationalCyclotomicElement(
        field=_TARGET_FIELD,
        coefficients_ascending=tuple(
            CanonicalRational(num=int(index == 2), den=1) for index in range(4)
        ),
    )
    return CyclotomicFieldEmbedding(
        source_order=6, target_field=_TARGET_FIELD, generator_image=image
    )


def _conjugate_embedding() -> CyclotomicFieldEmbedding:
    # zeta_6 maps to zeta_12^-2 = 1 - zeta_12^2.
    image = RationalCyclotomicElement(
        field=_TARGET_FIELD,
        coefficients_ascending=tuple(
            CanonicalRational(num=(1 if index == 0 else -1 if index == 2 else 0), den=1)
            for index in range(4)
        ),
    )
    return CyclotomicFieldEmbedding(
        source_order=6, target_field=_TARGET_FIELD, generator_image=image
    )


def test_embedding_maps_large_rational_coordinate_without_generic_product_bound() -> (
    None
):
    from fractions import Fraction

    from jacobian._exact import CanonicalRational

    value = RationalCyclotomicElement(
        field=_SOURCE_FIELD,
        coefficients_ascending=(
            CanonicalRational(num=10**25, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    mapped = _map_element(value, _embedding().generator_image, _TARGET_FIELD)

    assert rational(mapped.coefficients_ascending[0]).as_fraction() == Fraction(10**25)
    assert all(
        rational(coefficient).as_fraction() == 0
        for coefficient in mapped.coefficients_ascending[1:]
    )


def test_global_equality_compares_nonzero_cross_embeddings() -> None:
    # Characters 2 and 10 are Galois conjugate. Mapping their generators by
    # opposite embeddings gives the same common Nebentypus and the same form.
    left = _basis_form(_space(13, 2, kind="S"), basis_index=0)
    right = _basis_form(_space(13, 10, kind="S"), basis_index=0)
    assert modular_form_coordinates_global_equal(
        left, _embedding(), right, _conjugate_embedding()
    )

    # Independently realize each source coordinate with the exact character
    # q-expansion API, then compare after applying the declared embeddings.
    left_q = modular_character_coordinates_q_expansion(left).coefficients
    right_q = modular_character_coordinates_q_expansion(right).coefficients
    left_image = _embedding().generator_image
    right_image = _conjugate_embedding().generator_image
    assert tuple(
        _map_element(value, left_image, _TARGET_FIELD) for value in left_q
    ) == tuple(_map_element(value, right_image, _TARGET_FIELD) for value in right_q)

    # A distinct nonzero scalar multiple is not equal to the first form.
    unequal_right = ModularFormCoordinates(
        space=right.space,
        basis_id=right.basis_id,
        coordinates=(_element(2),),
    )
    assert not modular_form_coordinates_global_equal(
        left, _embedding(), unequal_right, _conjugate_embedding()
    )
    unequal_q = modular_character_coordinates_q_expansion(unequal_right).coefficients
    assert tuple(
        _map_element(value, right_image, _TARGET_FIELD) for value in right_q
    ) != tuple(_map_element(value, right_image, _TARGET_FIELD) for value in unequal_q)

    # Distinct mapped characters cannot describe the same nonzero form.
    with pytest.raises(OperationDomainValidationError) as mismatch:
        modular_form_coordinates_global_equal(
            left, _embedding(), _basis_form(_space(13, 10, kind="S"), 0), _embedding()
        )
    assert mismatch.value.errors()[0]["type"] == (
        "modular_form.global_equality_character_mismatch"
    )

    with pytest.raises(OperationDomainValidationError) as same_map:
        modular_form_coordinates_global_equal(
            left, _embedding(), _basis_form(_space(13, 2, kind="S"), 0), _embedding()
        )
    assert same_map.value.errors()[0]["type"] == (
        "modular_form.global_equality_same_embedding"
    )

    wrong_weight = ModularFormCoordinates.model_construct(
        space=_space(13, 10, weight=4),
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_global_equal(
            left, _embedding(), wrong_weight, _conjugate_embedding()
        )
    assert error.value.errors()[0]["type"] == "modular_form.global_equality_weight"


def test_direct_native_comparison_shares_one_execution_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _basis_form(_space(13, 2, kind="S"), basis_index=0)
    right = _basis_form(_space(13, 10, kind="S"), basis_index=0)
    executions = []

    def record_execution(
        _form: ModularFormCoordinates,
        _context: object,
        _image: RationalCyclotomicElement,
        target: RationalCyclotomicField,
        precision: int,
    ) -> tuple[RationalCyclotomicElement, ...]:
        executions.append(current_request_execution())
        zero = RationalCyclotomicElement(
            field=target,
            coefficients_ascending=tuple(
                CanonicalRational(num=0, den=1) for _ in range(target.degree)
            ),
        )
        return (zero,) * precision

    monkeypatch.setattr(
        "jacobian.math.number_theory.modular_forms.global_equality.operations"
        "._mapped_prefix",
        record_execution,
    )

    assert modular_form_coordinates_global_equal(
        left, _embedding(), right, _conjugate_embedding()
    )
    assert len(executions) == 2
    assert executions[0] is not None and executions[0] is executions[1]


def test_global_equality_rejects_invalid_native_argument_types() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    valid = _form(_space(13, 4, kind="S"))
    for arguments, location in (
        ((object(), _embedding(), valid, _conjugate_embedding()), "left"),
        ((valid, _embedding(), object(), _conjugate_embedding()), "right"),
        ((valid, object(), valid, _conjugate_embedding()), "left_embedding"),
        ((valid, _embedding(), valid, object()), "right_embedding"),
    ):
        with pytest.raises(OperationDomainValidationError) as error:
            modular_form_coordinates_global_equal(*arguments)
        assert error.value.errors()[0]["loc"] == (location,)
        assert error.value.errors()[0]["type"].startswith(
            "modular_form.global_equality_"
        )


def test_global_equality_rejects_forged_native_arguments_as_domain_errors() -> None:
    valid = _form(_space(13, 4, kind="S"))
    forged_form = ModularFormCoordinates.model_construct()
    forged_embedding = CyclotomicFieldEmbedding.model_construct()
    for arguments, location in (
        ((forged_form, _embedding(), valid, _conjugate_embedding()), "left"),
        ((valid, forged_embedding, valid, _conjugate_embedding()), "left_embedding"),
        ((valid, _embedding(), forged_form, _conjugate_embedding()), "right"),
        ((valid, _embedding(), valid, forged_embedding), "right_embedding"),
    ):
        with pytest.raises(OperationDomainValidationError) as error:
            modular_form_coordinates_global_equal(*arguments)
        assert error.value.errors()[0]["loc"] == (location,)
        assert error.value.errors()[0]["type"] == (
            "modular_form.global_equality_argument_invalid"
        )


def test_global_equality_handles_zero_spaces_and_unequal_levels() -> None:
    zero_left = _form(_space(13, 4, kind="S"))
    zero_right = _form(_space(13, 8, kind="S"))
    assert not zero_left.coordinates and not zero_right.coordinates
    assert modular_form_coordinates_global_equal(
        zero_left, _embedding(), zero_right, _conjugate_embedding()
    )

    # Same source map at both ends belongs to the adjacent same-character
    # transport/equality contract, including across unequal levels.
    lower = _form(_space(13, 2))
    upper = _form(_space(26, 2))
    with pytest.raises(OperationDomainValidationError) as same_map_error:
        modular_form_coordinates_global_equal(lower, _embedding(), upper, _embedding())
    assert (
        same_map_error.value.errors()[0]["type"]
        == "modular_form.global_equality_same_embedding"
    )


def test_global_equality_reaches_gamma1_78_sturm_boundary() -> None:
    # [SL2(Z):Gamma1(78)] = 4032, hence weight two needs 673 coefficients.
    # This exercises the high-precision worker lane and compares nonzero forms.
    left = _form(_space(26, 4), first_coordinate=1)
    right = _form(_space(39, 8), first_coordinate=1)
    assert not modular_form_coordinates_global_equal(
        left, _embedding(), right, _conjugate_embedding()
    )
