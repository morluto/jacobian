"""The PARI frame conversion owns one materialization per request."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    basis,
)
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormChangeOfBasisFrame,
)


def _pari_frame_and_form() -> tuple[
    ModularFormChangeOfBasisFrame, ModularFormCoordinates
]:
    space = ModularFormSpace(level=5, weight=4, kind="M")
    zero = CanonicalRational(num=0, den=1)
    one = CanonicalRational(num=1, den=1)
    frame = ModularFormBasisFrameRequest.model_validate(
        {
            "space": space,
            "source_basis_id": "gamma0-rational-gamma0-sturm-rref-v1",
            "source_labels": ("q^0", "q^1", "q^2"),
            "labels": ("f0", "f1", "f2"),
            "entries": ((one, zero, zero), (zero, one, zero), (zero, zero, one)),
        }
    ).as_frame()
    form = ModularFormCoordinates.model_validate(
        {
            "space": space,
            "basis_id": "gamma0-rational-gamma0-sturm-rref-v1",
            "coordinates": (one, zero, zero),
        }
    )
    return frame, form


def test_to_frame_reuses_the_basis_materialized_by_frame_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, form = _pari_frame_and_form()
    one = CanonicalRational(num=1, den=1)
    zero = CanonicalRational(num=0, den=1)
    calls: list[object] = []
    materialize = basis._materialize_pari_basis

    def counting(plan: basis._BasisPlan) -> basis._BasisPlan:
        calls.append(plan)
        return materialize(plan)

    monkeypatch.setattr(basis, "_materialize_pari_basis", counting)
    result = basis.modular_form_coordinates_to_frame(frame, form)
    assert result.coordinates == (one, zero, zero)
    assert len(calls) == 1


def test_noncoprime_framed_hecke_rejects_before_pari(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = _pari_frame_and_form()

    def cannot_materialize(_plan: basis._BasisPlan) -> None:
        pytest.fail("a noncoprime Hecke index must fail before PARI materialization")

    monkeypatch.setattr(basis, "_materialize_pari_basis", cannot_materialize)
    with pytest.raises(OperationDomainValidationError, match="coprime"):
        basis.modular_form_hecke_matrix_in_frame(frame, 5)


def test_zero_dimensional_pari_basis_does_not_launch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cannot_launch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an empty basis is already determined by the dimension formula")

    monkeypatch.setattr(basis, "pari_gamma0_rational_basis", cannot_launch)
    space = ModularFormSpace(level=5, weight=2, kind="S")
    result = basis.modular_form_basis_q_expansions(space, 3)
    assert result.elements == ()
    assert result.basis_id == "gamma0-rational-gamma0-sturm-rref-v1"
