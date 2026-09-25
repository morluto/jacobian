"""Exact field-valued modular-form basis coverage for the admitted slice."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms import (
    character_basis as character_basis_module,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    CHARACTER_BASIS_ID,
    modular_character_basis_q_expansions,
    modular_character_coordinates_hecke,
    modular_character_coordinates_product,
    modular_character_coordinates_q_expansion,
    modular_character_hecke_matrix,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterBasis,
    ModularCharacterBasisRequest,
    ModularCharacterHeckeMatrix,
    ModularCharacterHeckeMatrixRequest,
)
from jacobian.math.number_theory.modular_forms.pari_backend import (
    _pari_character_request,
    pari_character_basis,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _character_tool(operation_id: str):
    from jacobian.math.number_theory.modular_forms._tools import TOOLS

    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _space(character_coordinate: int = 2) -> ModularFormSpace:
    character = dirichlet_character(character_group(13), (character_coordinate,))
    return ModularFormSpace(
        level=13,
        weight=2,
        kind="S",
        character=character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )


def _coords(value: RationalCyclotomicElement) -> tuple[tuple[int, int], ...]:
    return tuple(
        (coefficient.num, coefficient.den)
        for coefficient in value.coefficients_ascending
    )


def _scalar(value: int) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=RationalCyclotomicField(order=6),
        coefficients_ascending=(
            {"num": value, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


def _form(value: int, coordinate: int = 2) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=_space(coordinate),
        basis_id=CHARACTER_BASIS_ID,
        coordinates=(_scalar(value),),
    )


@pytest.mark.parametrize("coordinate", [2, 10])
def test_exact_character_basis_is_parented_and_sturm_determining(
    coordinate: int,
) -> None:
    space = _space(coordinate)
    result = modular_character_basis_q_expansions(space)

    assert result.space == space
    assert result.precision == 3
    assert result.basis_id == "gamma0-13-even-order6-character-sturm-v1"
    assert len(result.elements) == 1
    expansion = result.elements[0].expansion
    assert expansion.space == space
    assert len(expansion.coefficients) == 3
    assert all(
        coefficient.field == RationalCyclotomicField(order=6)
        for coefficient in expansion.coefficients
    )
    assert _coords(expansion.coefficients[0]) == ((0, 1), (0, 1))
    assert _coords(expansion.coefficients[1]) == ((1, 1), (0, 1))
    if coordinate == 2:
        assert _coords(expansion.coefficients[2]) == ((-1, 1), (-1, 1))
    else:
        assert _coords(expansion.coefficients[2]) == ((-2, 1), (1, 1))
    restored = TypeAdapter(ModularCharacterBasis).validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_character_basis_rejects_other_modular_space() -> None:
    assert ModularCharacterBasisRequest(space=_space()).space == _space()
    assert _character_tool("modular_form.character_basis.compute") is not None
    with pytest.raises(OperationDomainValidationError, match="supports S2"):
        modular_character_basis_q_expansions(
            ModularFormSpace(
                level=13,
                weight=2,
                kind="M",
                character=dirichlet_character(character_group(13), (2,)),
                coefficient_domain=RationalCyclotomicField(order=6),
            ),
        )


@pytest.mark.parametrize("coordinates", [(12,), ("2",)])
def test_forged_character_coordinates_are_rejected_before_pari(
    monkeypatch: pytest.MonkeyPatch, coordinates: tuple[object, ...]
) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis
    from jacobian.math.number_theory.modular_forms.pari_backend import (
        _pari_character_request,
    )

    valid = _space()
    forged_character = type(valid.character).model_construct(
        group=valid.character.group,
        coordinates=coordinates,
    )
    forged_space = ModularFormSpace.model_construct(
        group=valid.group,
        level=valid.level,
        weight=valid.weight,
        kind=valid.kind,
        character=forged_character,
        coefficient_domain=valid.coefficient_domain,
    )

    def backend_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("PARI adapter ran for malformed caller coordinates")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    with pytest.raises(OperationDomainValidationError, match="coordinates"):
        _pari_character_request(forged_space)
    with pytest.raises(OperationDomainValidationError, match="coordinates"):
        modular_character_basis_q_expansions(forged_space)


def test_character_basis_carrier_rejects_foreign_coefficient_parent() -> None:
    space = _space()
    foreign = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=3),
        coefficients_ascending=(
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
        ),
    )
    with pytest.raises(ValidationError, match="belong to the space coefficient field"):
        from jacobian.math.number_theory.modular_forms.character_basis_models import (
            ModularCharacterQExpansion,
        )

        ModularCharacterQExpansion(
            space=space,
            basis_id="gamma0-13-even-order6-character-sturm-v1",
            coefficients=(foreign, foreign, foreign),
        )


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        (0, (((0, 1), (0, 1)),) * 3),
        (1, (((0, 1), (0, 1)), ((1, 1), (0, 1)), ((-1, 1), (-1, 1)))),
        (2, (((0, 1), (0, 1)), ((2, 1), (0, 1)), ((-2, 1), (-2, 1)))),
    ],
)
def test_character_coordinates_realize_exact_sturm_prefix(
    scalar: int, expected: tuple[tuple[tuple[int, int], ...], ...]
) -> None:
    form = _form(scalar)
    expansion = modular_character_coordinates_q_expansion(form)
    assert expansion.space == form.space
    assert expansion.basis_id == CHARACTER_BASIS_ID
    assert tuple(_coords(value) for value in expansion.coefficients) == expected
    assert (
        _character_tool("modular_form.character_coordinates.q_expansion.compute")
        is not None
    )


@pytest.mark.parametrize(
    ("character_coordinate", "expected_eigenvalue"),
    [(2, ((-1, 1), (-1, 1))), (10, ((-2, 1), (1, 1)))],
)
def test_character_hecke_t2_returns_same_space_exact_coordinates(
    character_coordinate: int,
    expected_eigenvalue: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    form = _form(1, coordinate=character_coordinate)

    result = modular_character_coordinates_hecke(form, 2)

    assert result.space == form.space
    assert result.basis_id == form.basis_id
    assert _coords(result.coordinates[0]) == expected_eigenvalue
    assert _character_tool("modular_form.character_coordinates.hecke.apply")


def test_character_hecke_maximum_admitted_index_returns_exact_parent() -> None:
    form = _form(1)

    result = modular_character_coordinates_hecke(form, 32)

    assert result.space == form.space
    assert result.basis_id == form.basis_id
    assert len(result.coordinates) == 1
    assert result.coordinates[0].field == form.coordinates[0].field


def test_character_hecke_index_five_uses_power_basis_height_bound() -> None:
    # Through q^(2n), the embedding bound is 1+...+10 = 55. For
    # u+v*zeta_6, both power-basis coordinates need the safe 3*55 envelope.
    embedding_bound = 10 * 11 // 2
    coordinate_bound = 3 * embedding_bound
    assert (embedding_bound, coordinate_bound, len(str(coordinate_bound))) == (
        55,
        165,
        3,
    )
    result = modular_character_coordinates_hecke(_form(1), 5)
    assert _coords(result.coordinates[0]) == ((1, 1), (-2, 1))


def test_character_hecke_rejects_power_basis_coordinate_above_admitted_bound(
    monkeypatch,
) -> None:
    form = _form(1)

    def out_of_bound_prefix(*_args, **_kwargs):
        prefix = [(Fraction(0), Fraction(0)) for _ in range(11)]
        prefix[1] = (Fraction(1), Fraction(0))
        prefix[10] = (Fraction(166), Fraction(0))
        return (tuple(prefix),)

    monkeypatch.setattr(
        character_basis_module, "pari_character_basis", out_of_bound_prefix
    )
    with pytest.raises(RuntimeError, match="power-basis bound"):
        modular_character_coordinates_hecke(form, 5)


def test_conjugate_character_product_returns_sturm_reconstructed_target() -> None:
    from fractions import Fraction

    from jacobian.math.number_theory.modular_forms import cyclotomic
    from jacobian.math.number_theory.modular_forms.character_basis import _coefficient
    from jacobian.math.number_theory.modular_forms.pari_backend import (
        _pari_character_request,
        pari_character_basis,
    )

    left = _form(1, coordinate=2)
    right = _form(1, coordinate=10)

    product = modular_character_coordinates_product(left, right)

    assert product.space == ModularFormSpace(
        level=13,
        weight=4,
        kind="S",
        coefficient_domain=RationalCyclotomicField(order=6),
    )
    assert len(product.coefficients) == 5
    assert _character_tool("modular_form.character_coordinates.product.compute")

    precision = 5  # Sturm bound 4 for S4(Gamma0(13)).
    left_basis = pari_character_basis(
        left.space, precision, 1, character_request=_pari_character_request(left.space)
    )
    right_basis = pari_character_basis(
        right.space,
        precision,
        1,
        character_request=_pari_character_request(right.space),
    )
    field = RationalCyclotomicField(order=6)
    left_prefix = tuple(_coefficient(field, term) for term in left_basis[0])
    right_prefix = tuple(_coefficient(field, term) for term in right_basis[0])
    expected = []
    for degree in range(precision):
        value = _coefficient(field, (Fraction(0), Fraction(0)))
        for left_index in range(degree + 1):
            value = cyclotomic.add(
                value,
                cyclotomic.multiply(
                    left_prefix[left_index], right_prefix[degree - left_index]
                ),
            )
        coordinates = cyclotomic._validate_element(value)[1]
        assert coordinates[1] == 0
        expected.append(coordinates[0])

    assert tuple(
        tuple(
            coordinate.as_fraction()
            for coordinate in coefficient.coefficients_ascending
        )
        for coefficient in product.coefficients
    ) == tuple((value, Fraction(0)) for value in expected)


def test_character_hecke_t1_is_identity_and_bad_level_index_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis

    def backend_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("PARI ran for an identity or rejected Hecke index")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    assert modular_character_coordinates_hecke(_form(1), 1) == _form(1)
    with pytest.raises(OperationDomainValidationError, match="gcd"):
        modular_character_coordinates_hecke(_form(1), 13)


@pytest.mark.parametrize("coordinate", [2, 10])
def test_character_hecke_matrix_is_bound_and_matches_normalized_a_n(
    coordinate: int,
) -> None:
    space = _space(coordinate)
    index = 5
    matrix = modular_character_hecke_matrix(space, index)
    assert matrix.space == space
    assert matrix.basis_id == CHARACTER_BASIS_ID
    assert matrix.index == index
    assert matrix.entries[0][0].field == space.coefficient_domain

    # Compare with the separately requested normalized q coefficient a_n.
    coefficients = pari_character_basis(
        space,
        index + 1,
        1,
        character_request=_pari_character_request(space),
    )
    expected = character_basis_module._coefficient(
        space.coefficient_domain, coefficients[0][index]
    )
    assert matrix.entries[0][0] == expected

    # In this one-dimensional basis, matrix action is exactly scalar
    # multiplication and agrees with the existing coordinate operation.
    scalar = _form(7, coordinate)
    image = modular_character_coordinates_hecke(scalar, index)
    assert image.coordinates == (
        character_basis_module.cyclotomic.multiply(
            matrix.entries[0][0], scalar.coordinates[0]
        ),
    )
    assert ModularCharacterHeckeMatrixRequest(space=space, index=index).space == space
    assert (
        ModularCharacterHeckeMatrix.model_validate_json(matrix.model_dump_json())
        == matrix
    )
    tool = _character_tool("modular_form.character_hecke_matrix.compute")
    assert tool is not None
    assert (
        tool.run(ModularCharacterHeckeMatrixRequest(space=space, index=index)) == matrix
    )


def test_zero_character_form_avoids_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis

    def backend_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("PARI ran for a zero character form")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    expansion = modular_character_coordinates_q_expansion(_form(0))
    assert all(
        not any(coefficient.num for coefficient in value.coefficients_ascending)
        for value in expansion.coefficients
    )


def test_character_coordinate_operation_rejects_forged_scalar_before_pari(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis

    malformed_scalar = RationalCyclotomicElement.model_construct(
        field=RationalCyclotomicField(order=6),
        coefficients_ascending=({"num": 1, "den": 0}, {"num": 0, "den": 1}),
    )
    forged_form = ModularFormCoordinates.model_construct(
        space=_space(),
        basis_id=CHARACTER_BASIS_ID,
        coordinates=(malformed_scalar,),
    )

    def backend_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("PARI ran for malformed character coordinates")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        modular_character_coordinates_q_expansion(forged_form)
