"""Typed wire contracts for exact quadratic-form scalar multiplication."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)

MAX_QUADRATIC_SCALE_AXIS = 128
MAX_QUADRATIC_SCALE_SUPPORT = 4_096

__all__ = [
    "MAX_QUADRATIC_SCALE_AXIS",
    "MAX_QUADRATIC_SCALE_SUPPORT",
    "QuadraticFormScaleRequest",
    "QuadraticFormScaleResult",
]


class QuadraticFormScaleRequest(StrictModel):
    """Scale one rational form by an exact rational factor."""

    form: RationalQuadraticForm
    factor: CanonicalRational


class QuadraticFormScaleResult(StrictModel):
    """Source form, exact factor, and the coefficientwise scaled form."""

    source_form: RationalQuadraticForm
    factor: CanonicalRational
    form: RationalQuadraticForm

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: RationalQuadraticForm,
        factor: CanonicalRational,
        form: RationalQuadraticForm,
    ) -> QuadraticFormScaleResult:
        """Build a result after the admitted kernel established its values."""

        return cls.model_construct(
            source_form=source_form,
            factor=factor,
            form=form,
        )

    def require_scaling_agrees_with_its_sources(self) -> QuadraticFormScaleResult:
        """Check the declared relation without replaying the kernel.

        The result carries the source form and the exact factor, so a consumer
        can confirm that ``form`` is their coefficientwise product on the same
        ordered axis. This is a structural check on the result, not a second
        computation of the value.
        """

        from jacobian.math.number_theory.quadratic_forms.general.scaling_operations import (
            _scaled_factors,
        )

        if self.form.axis != self.source_form.axis:
            raise PydanticCustomError(
                "quadratic_form.scale_axis_mismatch",
                "scalar multiplication must preserve the form axis",
            )
        if len(self.form.diagonal_coefficients) != len(
            self.source_form.diagonal_coefficients
        ):
            raise PydanticCustomError(
                "quadratic_form.scale_diagonal_axis",
                "scaling must preserve every diagonal coefficient slot",
            )
        expected_diagonal = []
        for coefficient in self.source_form.diagonal_coefficients:
            left_num, right_num, left_den, right_den = _scaled_factors(
                coefficient, self.factor
            )
            expected_diagonal.append(
                CanonicalRational.from_integer_ratio(
                    left_num * right_num, left_den * right_den
                )
            )
        if tuple(expected_diagonal) != self.form.diagonal_coefficients:
            raise PydanticCustomError(
                "quadratic_form.scale_diagonal",
                "scaled diagonal coefficients must equal factor times the source",
            )
        source_cross = {
            (term.left, term.right): term.coefficient
            for term in self.source_form.cross_terms
        }
        if len(self.form.cross_terms) != len(source_cross):
            raise PydanticCustomError(
                "quadratic_form.scale_cross_axis",
                "scaling must preserve every cross-term slot",
            )
        for term in self.form.cross_terms:
            source_coefficient = source_cross.get((term.left, term.right))
            if source_coefficient is None:
                raise PydanticCustomError(
                    "quadratic_form.scale_cross_axis",
                    "a scaled cross term has no source term at the same position",
                )
            left_num, right_num, left_den, right_den = _scaled_factors(
                source_coefficient, self.factor
            )
            if term.coefficient != CanonicalRational.from_integer_ratio(
                left_num * right_num, left_den * right_den
            ):
                raise PydanticCustomError(
                    "quadratic_form.scale_cross",
                    "a scaled cross term must equal factor times its source",
                )
        return self
