"""The PARI frame conversion owns one materialization per request."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    basis,
)
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)


def test_to_frame_reuses_the_basis_materialized_by_frame_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    calls: list[object] = []
    materialize = basis._materialize_pari_basis

    def counting(plan: basis._BasisPlan) -> basis._BasisPlan:
        calls.append(plan)
        return materialize(plan)

    monkeypatch.setattr(basis, "_materialize_pari_basis", counting)
    result = basis.modular_form_coordinates_to_frame(frame, form)
    assert result.coordinates == (one, zero, zero)
    assert len(calls) == 1
