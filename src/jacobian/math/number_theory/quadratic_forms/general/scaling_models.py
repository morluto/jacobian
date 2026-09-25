"""Typed contracts for exact scalar multiplication of rational forms."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)

MAX_QUADRATIC_SCALE_AXIS = 128
MAX_QUADRATIC_SCALE_SUPPORT = 4_096
MAX_QUADRATIC_SCALE_OUTPUT_BYTES = 1_500_000


class QuadraticFormScaleRequest(StrictModel):
    """Scale one rational form by an exact rational factor."""

    form: RationalQuadraticForm = Field(
        description=(
            "Rational quadratic form; scalar multiplication admits axis length "
            f"at most {MAX_QUADRATIC_SCALE_AXIS} and total diagonal/cross-term "
            f"support at most {MAX_QUADRATIC_SCALE_SUPPORT}."
        )
    )
    factor: CanonicalRational = Field(
        description="Exact rational scalar; numerator and denominator are admitted by the operation."
    )


class QuadraticFormScaleResult(StrictModel):
    """Source form, exact factor, and the coefficientwise scaled form."""

    source_form: RationalQuadraticForm
    factor: CanonicalRational
    form: RationalQuadraticForm

    @model_validator(mode="after")
    def require_preserved_axis(self) -> Self:
        if self.form.axis != self.source_form.axis:
            raise PydanticCustomError(
                "quadratic_form.scale_axis_mismatch",
                "scalar multiplication must preserve the form axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: RationalQuadraticForm,
        factor: CanonicalRational,
        form: RationalQuadraticForm,
    ) -> Self:
        """Build a result after the admitted kernel established its values."""

        return cls.model_construct(
            source_form=source_form,
            factor=factor,
            form=form,
        )


__all__ = [
    "MAX_QUADRATIC_SCALE_AXIS",
    "MAX_QUADRATIC_SCALE_OUTPUT_BYTES",
    "MAX_QUADRATIC_SCALE_SUPPORT",
    "QuadraticFormScaleRequest",
    "QuadraticFormScaleResult",
]
