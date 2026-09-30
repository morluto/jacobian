"""Public integral-content admission bounds retained forms before copying."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    integral_coefficient_content,
)
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


@pytest.mark.parametrize("support", [1, 4097])
def test_catalog_and_native_content_admit_support_before_integrality(
    support: int,
) -> None:
    operation = Catalog.open().operation("quadratic_form.integral_content.compute")
    assert operation is not None
    axis = tuple(f"x{i}" for i in range(support))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(CanonicalRational(num=1, den=2) for _ in axis),
    )
    request = operation.request_type(form=form)
    expected = (
        OperationResourceAdmissionError
        if support > 4096
        else OperationDomainValidationError
    )
    code = (
        "quadratic_form.invariant_support_bound"
        if support > 4096
        else "quadratic_form.nonintegral_form"
    )

    with pytest.raises(expected) as catalog_refusal:
        operation.run(request)
    with pytest.raises(expected) as native_refusal:
        integral_coefficient_content(form)

    assert catalog_refusal.value.errors()[0]["type"] == code
    assert native_refusal.value.errors()[0]["type"] == code


def test_catalog_content_refuses_forged_request_form_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Catalog.open().operation("quadratic_form.integral_content.compute")
    assert operation is not None
    form = RationalQuadraticForm.model_construct(
        axis=(["nested"] * 4097,),
        diagonal_coefficients=(CanonicalRational(num=1, den=1),),
    )
    request = operation.request_type.model_construct(form=form)

    def refuse_dump(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a forged wrapper must not bypass native form admission")

    monkeypatch.setattr(RationalQuadraticForm, "model_dump", refuse_dump)
    with pytest.raises(OperationDomainValidationError) as refusal:
        operation.run(request)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


def test_catalog_content_refuses_a_forged_request_missing_its_form() -> None:
    operation = Catalog.open().operation("quadratic_form.integral_content.compute")
    assert operation is not None
    request = operation.request_type.model_construct()
    with pytest.raises(OperationDomainValidationError) as refusal:
        operation.run(request)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_type"


def test_catalog_content_refuses_invalid_native_request_type() -> None:
    operation = Catalog.open().operation("quadratic_form.integral_content.compute")
    assert operation is not None
    with pytest.raises(OperationDomainValidationError) as refusal:
        operation.run(None)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.content_request_type"
