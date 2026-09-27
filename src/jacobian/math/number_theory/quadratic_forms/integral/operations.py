"""Exact scalar extension of integral quadratic polynomials to rational forms."""

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_AXIS,
    MAX_INTEGRAL_QUADRATIC_FORM_TERMS,
    IntegralQuadraticForm,
    IntegralQuadraticFormInclusion,
)


def integral_form_to_rational(
    source: IntegralQuadraticForm,
) -> IntegralQuadraticFormInclusion:
    """Apply coefficient-wise ``ZZ -> QQ`` while preserving the coordinate axis."""

    if not isinstance(source, IntegralQuadraticForm):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.integral.form_type",
            message="expected a canonical integral quadratic form",
        )
    if not all(
        isinstance(value, tuple)
        for value in (source.axis, source.diagonal_coefficients, source.cross_terms)
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.integral.form_shape",
            message="integral quadratic form containers must be tuples",
        )
    if len(source.axis) > MAX_INTEGRAL_QUADRATIC_FORM_AXIS:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.integral.axis_bound",
            message="integral quadratic form axes are limited to 128 labels",
        )
    if len(source.axis) + len(source.cross_terms) > MAX_INTEGRAL_QUADRATIC_FORM_TERMS:
        raise OperationResourceAdmissionError(
            location=("form", "cross_terms"),
            code="quadratic_form.integral.term_bound",
            message="integral quadratic form support exceeds its term bound",
        )
    try:
        source = IntegralQuadraticForm.model_validate(source.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.integral.form_shape",
            message="expected a canonical bounded integral quadratic form",
        ) from exc

    target = RationalQuadraticForm(
        axis=source.axis,
        diagonal_coefficients=tuple(
            CanonicalRational.from_integer_ratio(value, 1)
            for value in source.diagonal_coefficients
        ),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=term.left,
                right=term.right,
                coefficient=CanonicalRational.from_integer_ratio(term.coefficient, 1),
            )
            for term in source.cross_terms
        ),
    )
    return IntegralQuadraticFormInclusion.model_construct(
        source=source,
        target=target,
    )


__all__ = ["integral_form_to_rational"]
