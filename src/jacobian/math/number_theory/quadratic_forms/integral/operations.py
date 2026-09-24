"""Exact scalar extension of integral quadratic polynomials to rational forms."""

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_OUTPUT_BYTES,
    IntegralQuadraticFormInclusion,
    IntegralQuadraticFormInclusionRequest,
)


def integral_form_to_rational(
    request: IntegralQuadraticFormInclusionRequest,
) -> IntegralQuadraticFormInclusion:
    """Apply coefficient-wise ``ZZ -> QQ`` while preserving the coordinate axis."""

    source = request.form
    coefficient_count = len(source.diagonal_coefficients) + len(source.cross_terms)
    # Source and target each retain every coefficient. Include conservative
    # JSON punctuation and labels before constructing the rational target.
    output_byte_bound = (
        coefficient_count * (2 * 258 + 96)
        + len(source.cross_terms) * 96
        + sum(len(label.encode("utf-8")) + 8 for label in source.axis)
        + 1_024
    )
    if output_byte_bound > MAX_INTEGRAL_QUADRATIC_FORM_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.integral.rational_extension_output_bound",
            message=(
                "source and target quadratic forms exceed the admitted "
                "serialized output byte bound"
            ),
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
    return IntegralQuadraticFormInclusion(
        source=source,
        target=target,
    )


__all__ = ["integral_form_to_rational"]
