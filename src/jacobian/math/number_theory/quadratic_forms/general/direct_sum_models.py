"""Typed source-bound contract for direct sums of rational quadratic forms."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)

MAX_DIRECT_SUM_AXIS = 128
MAX_DIRECT_SUM_COMPONENTS = 128
MAX_DIRECT_SUM_FORM_TERMS = 4_096
_MAX_FORM_COEFFICIENT_DIGITS = 256


def direct_sum_output_byte_upper_bound(
    dimension: int,
    support: int,
    *,
    map_component_digits: int = 1,
) -> int:
    """Conservatively bound the canonical JSON result before dense expansion.

    Source and result each retain every polynomial coefficient. Map entries
    are only 0 or 1. The final term covers coefficient wrappers and indices,
    source/output axes, model keys, and container separators.
    """

    retained_rational_bytes = (
        2 * support * (2 * (_MAX_FORM_COEFFICIENT_DIGITS + 1) + 40)
    )
    map_scalar_bytes = 2 * (map_component_digits + 1) + 40
    dense_map_bytes = 2 * dimension * dimension * map_scalar_bytes
    term_and_label_bytes = 2 * support * 100 + dimension * 300 + 100_000
    return retained_rational_bytes + dense_map_bytes + term_and_label_bytes


class QuadraticFormDirectSumRequest(StrictModel):
    """An ordered finite family of rational forms to sum orthogonally."""

    forms: tuple[RationalQuadraticForm, ...] = Field(
        max_length=MAX_DIRECT_SUM_COMPONENTS
    )

    @model_validator(mode="after")
    def admit_aggregate_shape(self) -> Self:
        dimension = sum(len(form.axis) for form in self.forms)
        support = sum(
            len(form.diagonal_coefficients) + len(form.cross_terms)
            for form in self.forms
        )
        if dimension > MAX_DIRECT_SUM_AXIS:
            raise ValueError(
                "quadratic-form direct sum exceeds the aggregate axis bound"
            )
        if support > MAX_DIRECT_SUM_FORM_TERMS:
            raise ValueError(
                "quadratic-form direct sum exceeds the aggregate support bound"
            )
        if (
            direct_sum_output_byte_upper_bound(dimension, support)
            > CanonicalLimits().max_output_bytes
        ):
            raise ValueError(
                "quadratic-form direct sum exceeds the canonical output byte bound"
            )
        return self


class QuadraticFormRestrictionRequest(StrictModel):
    """Restrict a rational form to an ordered coordinate subset."""

    form: RationalQuadraticForm
    selected_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_DIRECT_SUM_AXIS)

    @model_validator(mode="after")
    def admit_coordinate_subset(self) -> Self:
        if len(self.form.axis) > MAX_DIRECT_SUM_AXIS:
            raise ValueError("quadratic-form restriction exceeds the axis bound")
        if len(set(self.selected_axis)) != len(self.selected_axis):
            raise ValueError("selected quadratic-form coordinates must be unique")
        if any(label not in self.form.axis for label in self.selected_axis):
            raise ValueError("selected coordinates must belong to the source axis")
        support = len(self.form.diagonal_coefficients) + len(self.form.cross_terms)
        if support > MAX_DIRECT_SUM_FORM_TERMS:
            raise ValueError("quadratic-form restriction exceeds the support bound")
        if (
            direct_sum_output_byte_upper_bound(len(self.form.axis), support)
            > CanonicalLimits().max_output_bytes
        ):
            raise ValueError("quadratic-form restriction exceeds the output byte bound")
        return self


class QuadraticFormRestrictionResult(StrictModel):
    """A coordinate restriction and its source-coordinate inclusion matrix."""

    source_form: RationalQuadraticForm
    selected_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_DIRECT_SUM_AXIS)
    form: RationalQuadraticForm
    inclusion: RationalMatrix

    @model_validator(mode="after")
    def require_coordinate_inclusion_shape(self) -> Self:
        if len(self.source_form.axis) > MAX_DIRECT_SUM_AXIS:
            raise ValueError("quadratic-form restriction exceeds the axis bound")
        if len(set(self.selected_axis)) != len(self.selected_axis):
            raise ValueError("selected quadratic-form coordinates must be unique")
        if any(label not in self.source_form.axis for label in self.selected_axis):
            raise ValueError("selected coordinates must belong to the source axis")
        if self.form.axis != self.selected_axis:
            raise ValueError(
                "restricted form axis must preserve selected coordinate order"
            )
        if (self.inclusion.row_count, self.inclusion.column_count) != (
            len(self.source_form.axis),
            len(self.selected_axis),
        ):
            raise ValueError("coordinate inclusion has inconsistent dimensions")
        source_positions = {
            label: index for index, label in enumerate(self.source_form.axis)
        }
        expected = tuple(
            tuple(
                CanonicalRational.from_integer_ratio(
                    int(row == source_positions[label]), 1
                )
                for label in self.selected_axis
            )
            for row in range(len(self.source_form.axis))
        )
        if self.inclusion.entries != expected:
            raise ValueError(
                "coordinate inclusion must select the declared source axes"
            )
        support = len(self.source_form.diagonal_coefficients) + len(
            self.source_form.cross_terms
        )
        map_digits = max(
            (
                canonical_rational_component_digits(value)
                for row in self.inclusion.entries
                for value in row
            ),
            default=1,
        )
        if (
            direct_sum_output_byte_upper_bound(
                len(self.source_form.axis), support, map_component_digits=map_digits
            )
            > CanonicalLimits().max_output_bytes
        ):
            raise ValueError("quadratic-form restriction exceeds the output byte bound")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: RationalQuadraticForm,
        selected_axis: tuple[OpaqueLabel, ...],
        form: RationalQuadraticForm,
        inclusion: RationalMatrix,
    ) -> Self:
        """Build from a pre-admitted coordinate slice."""

        return cls.model_construct(
            source_form=source_form,
            selected_axis=selected_axis,
            form=form,
            inclusion=inclusion,
        )


class QuadraticFormDirectSumResult(StrictModel):
    """The orthogonal sum and coordinate inclusions/projections for its factors."""

    source_forms: tuple[RationalQuadraticForm, ...] = Field(
        max_length=MAX_DIRECT_SUM_COMPONENTS
    )
    form: RationalQuadraticForm
    coordinate_inclusions: tuple[RationalMatrix, ...] = Field(
        max_length=MAX_DIRECT_SUM_COMPONENTS
    )
    coordinate_projections: tuple[RationalMatrix, ...] = Field(
        max_length=MAX_DIRECT_SUM_COMPONENTS
    )

    @model_validator(mode="after")
    def require_transport_shapes(self) -> Self:
        if not (
            len(self.source_forms)
            == len(self.coordinate_inclusions)
            == len(self.coordinate_projections)
        ):
            raise ValueError("direct-sum factors and coordinate maps must correspond")
        total = sum(len(source.axis) for source in self.source_forms)
        if total != len(self.form.axis):
            raise ValueError(
                "direct-sum form dimension must equal the summed dimensions"
            )
        support = sum(
            len(source.diagonal_coefficients) + len(source.cross_terms)
            for source in self.source_forms
        )
        if total > MAX_DIRECT_SUM_AXIS or support > MAX_DIRECT_SUM_FORM_TERMS:
            raise ValueError(
                "direct-sum result exceeds its admitted aggregate envelope"
            )
        map_component_digits = max(
            (
                canonical_rational_component_digits(value)
                for matrix in (
                    *self.coordinate_inclusions,
                    *self.coordinate_projections,
                )
                for row in matrix.entries
                for value in row
            ),
            default=1,
        )
        if (
            direct_sum_output_byte_upper_bound(
                total, support, map_component_digits=map_component_digits
            )
            > CanonicalLimits().max_output_bytes
        ):
            raise ValueError(
                "direct-sum result exceeds the canonical output byte bound"
            )

        for source, inclusion, projection in zip(
            self.source_forms,
            self.coordinate_inclusions,
            self.coordinate_projections,
            strict=True,
        ):
            dimension = len(source.axis)
            if (inclusion.row_count, inclusion.column_count) != (total, dimension):
                raise ValueError("direct-sum inclusion has inconsistent dimensions")
            if (projection.row_count, projection.column_count) != (dimension, total):
                raise ValueError("direct-sum projection has inconsistent dimensions")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_forms: tuple[RationalQuadraticForm, ...],
        form: RationalQuadraticForm,
        coordinate_inclusions: tuple[RationalMatrix, ...],
        coordinate_projections: tuple[RationalMatrix, ...],
    ) -> Self:
        """Build from the admitted canonical kernel without replaying relations."""

        return cls.model_construct(
            source_forms=source_forms,
            form=form,
            coordinate_inclusions=coordinate_inclusions,
            coordinate_projections=coordinate_projections,
        )


__all__ = [
    "MAX_DIRECT_SUM_AXIS",
    "MAX_DIRECT_SUM_COMPONENTS",
    "MAX_DIRECT_SUM_FORM_TERMS",
    "QuadraticFormDirectSumRequest",
    "QuadraticFormDirectSumResult",
]
