"""Exact Jacobian-to-PARI Dirichlet-character bridge checks."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms._pari_basis_worker import (
    _character_vector,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
    _pari_character_request,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace


class _FakePariGroup:
    def __init__(self, modulus: int, generator: int) -> None:
        self.modulus = modulus
        self.generator = generator

    def __getitem__(self, index: int) -> object:
        if index == 1:
            return _FakePariUnitGroup()
        raise IndexError(index)


class _FakePariUnitGroup:
    def __getitem__(self, index: int) -> tuple[int, ...]:
        if index == 1:
            return (12,)
        if index == 2:
            return (6,)
        raise IndexError(index)


class _FakePari:
    """PARI-shaped interface with a deliberately different generator."""

    def __init__(self) -> None:
        # 6 = 2^5 is another generator of (Z/13Z)^*.
        self.group = _FakePariGroup(13, 6)
        self.calls: list[tuple[int, _FakePariGroup]] = []

    def znstar(self, modulus: int, flag: int) -> _FakePariGroup:
        assert (modulus, flag) == (13, 1)
        return self.group

    def __call__(self, vector: list[int]) -> tuple[int, ...]:
        return tuple(vector)

    def znlog(self, residue: int, group: _FakePariGroup) -> tuple[int, ...]:
        assert group is self.group
        power = 1
        for exponent in range(12):
            if power == residue:
                return (exponent,)
            power = power * group.generator % group.modulus
        raise ValueError("residue is not a unit")


def _space() -> ModularFormSpace:
    character = dirichlet_character(character_group(13), (2,))
    return ModularFormSpace(
        level=13,
        weight=2,
        kind="S",
        character=character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )


def test_order_six_even_character_serializes_with_explicit_field_order() -> None:
    request = _pari_character_request(_space())

    assert request["coefficient_field_order"] == 6
    character = request["character"]
    assert character == {
        "modulus": 13,
        "unit_residues": list(range(1, 13)),
        "generator_orders": [12],
        "unit_coordinates": [
            [index] for index in (0, 1, 4, 2, 9, 5, 11, 3, 8, 10, 7, 6)
        ],
        "coordinates": [2],
    }
    # chi(-1) = exp(2 pi i * 2*6/12) = 1, so it has the parity of weight 2.
    assert Fraction(2 * character["unit_coordinates"][11][0], 12) % 1 == 0


def test_pari_standard_vector_agrees_on_every_unit_without_ordering_assumption() -> (
    None
):
    payload = _pari_character_request(_space())
    fake_pari = _FakePari()

    group, character_vector = _character_vector(
        fake_pari,
        payload["character"],
        payload["coefficient_field_order"],
    )

    assert group is fake_pari.group
    assert character_vector == (10,)
    for residue, jacobian_row in zip(
        payload["character"]["unit_residues"],
        payload["character"]["unit_coordinates"],
        strict=True,
    ):
        jacobian_value = Fraction(2 * jacobian_row[0], 12) % 1
        pari_value = Fraction(10 * fake_pari.znlog(residue, group)[0], 12) % 1
        assert pari_value == jacobian_value


def test_live_cypari_standard_vector_and_every_unit_agreement() -> None:
    cypari = pytest.importorskip("cypari")
    payload = _pari_character_request(_space())

    group, character_vector = _character_vector(
        cypari.pari,
        payload["character"],
        payload["coefficient_field_order"],
    )

    assert tuple(int(value) for value in group[1][1]) == (12,)
    assert tuple(int(value) for value in character_vector) == (2,)
    unit_rows = dict(
        zip(
            payload["character"]["unit_residues"],
            payload["character"]["unit_coordinates"],
            strict=True,
        )
    )
    for residue, row in unit_rows.items():
        assert cypari.pari.chareval(group, character_vector, residue) == (
            Fraction(2 * row[0], 12) % 1
        )


def test_character_bridge_rejects_malformed_worker_request_before_pari_access() -> None:
    class UnavailablePari:
        def znstar(self, *_args: object) -> object:
            pytest.fail("PARI was accessed before worker request validation")

    with pytest.raises(ValueError, match="invalid shape"):
        _character_vector(UnavailablePari(), {"modulus": 13}, 6)


def test_character_serializer_rejects_field_that_does_not_contain_values() -> None:
    space = _space().model_copy(
        update={"coefficient_domain": RationalCyclotomicField(order=3)}
    )

    with pytest.raises(OperationDomainValidationError, match="must contain"):
        _pari_character_request(space)
