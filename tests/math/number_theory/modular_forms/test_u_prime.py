"""Exact same-space coordinate U_p checks for general Gamma0 levels."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms import basis
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormCoordinatesUPrimeRequest,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_PARI_BASIS_ID = "gamma0-rational-gamma0-sturm-rref-v1"
_TOOL = next(
    tool for tool in TOOLS if tool.operation_id == "modular_form.u_operator.apply"
)


def _rational_tuple(values: object) -> tuple[Fraction, ...]:
    return tuple(Fraction(value.num, value.den) for value in values)  # type: ignore[attr-defined]


def test_u_prime_gamma0_eleven_s2_matches_coefficient_selection_and_tool_lane() -> None:
    space = ModularFormSpace(level=11, weight=2, kind="S")
    sturm = 2
    prime = 11
    basis_prefix = basis.modular_form_basis_q_expansions(space, prime * sturm + 1)
    assert basis_prefix.basis_id == _PARI_BASIS_ID
    assert len(basis_prefix.elements) == 1
    form = ModularFormCoordinates(
        space=space,
        basis_id=basis_prefix.basis_id,
        coordinates=(CanonicalRational(num=1, den=1),),
    )

    request = ModularFormCoordinatesUPrimeRequest(form=form, prime=prime)
    image = _TOOL.run(request)
    assert image.space == space
    assert image.basis_id == _PARI_BASIS_ID
    assert len(image.coordinates) == 1

    source = _rational_tuple(
        basis_prefix.elements[0].expansion.q_expansion.coefficients
    )
    image_prefix = basis.modular_form_coordinates_q_expansion(image, sturm + 1)
    expected = tuple(source[prime * n] for n in range(sturm + 1))
    assert _rational_tuple(image_prefix.q_expansion.coefficients) == expected
    # The Gamma0(11) S_2 basis is normalized by its q coefficient, so this
    # independently checks the coordinate returned by exact reconstruction.
    assert image.coordinates[0] == CanonicalRational(
        num=source[prime].numerator, den=source[prime].denominator
    )


def test_u_prime_rejects_nondivisor_composite_and_character_parent() -> None:
    level_eleven = ModularFormSpace(level=11, weight=2, kind="S")
    form = ModularFormCoordinates(
        space=level_eleven,
        basis_id=_PARI_BASIS_ID,
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    with pytest.raises(OperationDomainValidationError, match="when p divides N"):
        basis.modular_form_coordinates_u_prime(form, 2)

    level_four = ModularFormSpace(level=4, weight=4, kind="M")
    level_four_basis = basis.modular_form_basis_q_expansions(level_four, 2)
    composite_form = ModularFormCoordinates(
        space=level_four,
        basis_id=level_four_basis.basis_id,
        coordinates=tuple(
            CanonicalRational(num=1 if i == 0 else 0, den=1)
            for i in range(len(level_four_basis.elements))
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="requires a prime"):
        basis.modular_form_coordinates_u_prime(composite_form, 4)

    chi4 = dirichlet_character(character_group(4), (1,))
    character_space = ModularFormSpace(
        level=4, weight=1, kind="M", character=chi4, coefficient_domain="QQ"
    )
    character_form = ModularFormCoordinates(
        space=character_space,
        basis_id="gamma0-four-chi4-weight-one-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    with pytest.raises(OperationDomainValidationError, match="trivial-character"):
        basis.modular_form_coordinates_u_prime(character_form, 2)


def test_u_prime_zero_dimensional_cusp_edge_is_same_space() -> None:
    space = ModularFormSpace(level=2, weight=1, kind="M")
    basis_prefix = basis.modular_form_basis_q_expansions(space, 1)
    assert not basis_prefix.elements
    form = ModularFormCoordinates(
        space=space, basis_id=basis_prefix.basis_id, coordinates=()
    )

    image = basis.modular_form_coordinates_u_prime(form, 2)

    assert image.space == space
    assert image.basis_id == basis_prefix.basis_id
    assert image.coordinates == ()


def test_u_prime_tool_contract_has_typed_request_and_result_lanes() -> None:
    assert _TOOL.request_type is ModularFormCoordinatesUPrimeRequest
    assert _TOOL.result_type is ModularFormCoordinates
    request_schema = _TOOL.request_type.model_json_schema()
    assert request_schema["properties"]["prime"]["type"] == "integer"
    assert "form" in request_schema["properties"]
    result_schema = _TOOL.result_type.model_json_schema()
    assert {"space", "basis_id", "coordinates"} <= set(result_schema["properties"])
