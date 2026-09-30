"""Public integral-content admission bounds retained forms before copying."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)


def test_catalog_integral_content_bounds_support_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Catalog.open().operation("quadratic_form.integral_content.compute")
    assert operation is not None
    axis = tuple(f"x{i}" for i in range(4097))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(CanonicalRational(num=1, den=1) for _ in axis),
    )
    request = operation.request_type(form=form)

    def refuse_dump(*_args: object, **_kwargs: object) -> None:
        pytest.fail("over-budget forms must not be copied during catalog execution")

    monkeypatch.setattr(RationalQuadraticForm, "model_dump", refuse_dump)

    with pytest.raises(OperationResourceAdmissionError) as refusal:
        operation.run(request)

    assert refusal.value.errors()[0]["type"] == "quadratic_form.invariant_support_bound"
