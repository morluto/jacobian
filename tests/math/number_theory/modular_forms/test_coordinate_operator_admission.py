"""Admission ordering for exact modular-form coordinate operators."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import basis
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def test_unsupported_coordinate_operators_reject_before_pari(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    form = ModularFormCoordinates(
        space=ModularFormSpace(level=5, weight=0, kind="M"),
        basis_id="gamma0-rational-gamma0-sturm-rref-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )

    def unexpected_backend(*args: object, **kwargs: object) -> object:
        pytest.fail("PARI ran before coordinate operator parent admission")

    monkeypatch.setattr(basis, "pari_gamma0_rational_basis", unexpected_backend)

    with pytest.raises(OperationDomainValidationError) as hecke_error:
        basis.modular_form_coordinates_hecke(form, 2)
    assert hecke_error.value.errors()[0]["type"] == (
        "modular_form.coordinates_hecke_unsupported_level"
    )

    with pytest.raises(OperationDomainValidationError) as u2_error:
        basis.modular_form_coordinates_u2(form)
    assert u2_error.value.errors()[0]["type"] == (
        "modular_form.coordinates_u2_unsupported_level"
    )
