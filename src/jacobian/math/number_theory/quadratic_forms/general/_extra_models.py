"""Additional exact structural quadratic-form contracts."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import RationalMatrix, RationalVectorSpaceBasis
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)

MAX_QUADRATIC_PULLBACK_AXIS = 128
MAX_QUADRATIC_PULLBACK_WORK = 2_000_000
MAX_QUADRATIC_PULLBACK_OUTPUT_ENTRIES = MAX_QUADRATIC_PULLBACK_AXIS**2


class FormRequest(StrictModel):
    form: RationalQuadraticForm


class SignatureResult(StrictModel):
    form: RationalQuadraticForm
    positive_index: int = Field(ge=0)
    negative_index: int = Field(ge=0)
    zero_index: int = Field(ge=0)
    signature: int

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.positive_index + self.negative_index + self.zero_index != len(
            self.form.axis
        ):
            raise ValueError("signature counts must sum to form dimension")
        if self.signature != self.positive_index - self.negative_index:
            raise ValueError("signature must equal positive minus negative index")
        return self


class RadicalResult(StrictModel):
    form: RationalQuadraticForm
    rank: int = Field(ge=0)
    radical: RationalVectorSpaceBasis

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.radical.ambient_dimension != len(
            self.form.axis
        ) or self.rank + self.radical.vectors.__len__() != len(self.form.axis):
            raise ValueError("rank and radical must span the form dimension")
        return self


class PullbackRequest(StrictModel):
    form: RationalQuadraticForm
    matrix: RationalMatrix
    target_axis: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.matrix.row_count != len(
            self.form.axis
        ) or self.matrix.column_count != len(self.target_axis):
            raise ValueError(
                "pullback matrix dimensions must map target axis to source axis"
            )
        if any(not label or label != label.strip() for label in self.target_axis):
            raise ValueError("target axis labels must be nonempty and trimmed")
        if len(set(self.target_axis)) != len(self.target_axis):
            raise ValueError("target axis labels must be unique")
        return self


class PullbackResult(StrictModel):
    source_form: RationalQuadraticForm
    matrix: RationalMatrix
    form: RationalQuadraticForm
    source_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_QUADRATIC_PULLBACK_AXIS)
    target_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_QUADRATIC_PULLBACK_AXIS)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.source_form.axis) > MAX_QUADRATIC_PULLBACK_AXIS:
            raise ValueError("pullback source axis exceeds the owner envelope")
        if self.source_axis != self.source_form.axis:
            raise ValueError("pullback source axis must match the source form")
        if self.target_axis != self.form.axis:
            raise ValueError("pullback target axis must match the result form")
        if self.matrix.row_count != len(self.source_axis):
            raise ValueError("pullback matrix rows must match the source axis")
        if self.matrix.column_count != len(self.target_axis):
            raise ValueError("pullback matrix columns must match the target axis")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: RationalQuadraticForm,
        matrix: RationalMatrix,
        form: RationalQuadraticForm,
        source_axis: tuple[OpaqueLabel, ...],
        target_axis: tuple[OpaqueLabel, ...],
    ) -> Self:
        return cls.model_construct(
            source_form=source_form,
            matrix=matrix,
            form=form,
            source_axis=source_axis,
            target_axis=target_axis,
        )


class DiagonalizationResult(StrictModel):
    form: RationalQuadraticForm
    diagonal: tuple[CanonicalRational, ...]
    change: RationalMatrix

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.form.axis)
        if (
            len(self.diagonal) != n
            or self.change.row_count != n
            or self.change.column_count != n
        ):
            raise ValueError("diagonalization dimensions must match form axis")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: RationalQuadraticForm,
        diagonal: tuple[CanonicalRational, ...],
        change: RationalMatrix,
    ) -> Self:
        return cls.model_construct(form=form, diagonal=diagonal, change=change)


class ModularProfileRequest(StrictModel):
    form: RationalQuadraticForm
    modulus: int = Field(ge=1, le=64)


class ModularProfileResult(StrictModel):
    form: RationalQuadraticForm
    modulus: int
    histogram: tuple[int, ...]
    total: int

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.histogram) != self.modulus or self.total != self.modulus ** len(
            self.form.axis
        ):
            raise ValueError("modular histogram total mismatch")
        return self


__all__ = [
    "DiagonalizationResult",
    "FormRequest",
    "ModularProfileRequest",
    "ModularProfileResult",
    "PullbackRequest",
    "PullbackResult",
    "RadicalResult",
    "SignatureResult",
]
