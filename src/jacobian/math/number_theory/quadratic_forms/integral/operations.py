"""Exact scalar extension of integral quadratic polynomials to rational forms."""

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
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
