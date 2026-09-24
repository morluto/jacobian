"""Typed source-bound contract for direct sums of rational quadratic forms."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    RationalQuadraticForm,
)

MAX_DIRECT_SUM_AXIS = 128
MAX_DIRECT_SUM_COMPONENTS = 128
MAX_DIRECT_SUM_FORM_TERMS = 4_096
MAX_DIRECT_SUM_OUTPUT_DIGITS = 8_000_000


def direct_sum_output_digit_upper_bound(
    dimension: int,
    retained_coefficients: int,
    *,
    map_component_digits: int = 1,
) -> int:
    """Conservatively bound the result's serialized decimal digits.

    Every retained coefficient component carries at most
    ``MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS`` digits, and each dense map
    entry carries two components of at most ``map_component_digits + 1``
    digits. The dimension and coefficient counts bound the serialized
    structure linearly, so this digit bound together with the aggregate
    cardinality envelopes bounds the canonical output without transport
    policy.
    """

    retained_rational_digits = (
        retained_coefficients * 2 * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS + 1)
    )
    dense_map_digits = 2 * dimension * dimension * 2 * (map_component_digits + 1)
    return retained_rational_digits + dense_map_digits


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
        # The kernel's output is exactly the retained coefficients and 0/1
        # block maps, so the axis and support cardinality envelopes already
        # bound the aggregate output digits below MAX_DIRECT_SUM_OUTPUT_DIGITS;
        # the result constructor re-establishes the bound for deserialized
        # values whose map entries are caller supplied.
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
        # As above, the axis and support envelopes bound the restricted
        # output's aggregate digits; see the result constructor.
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
        source_support = len(self.source_form.diagonal_coefficients) + len(
            self.source_form.cross_terms
        )
        output_support = len(self.form.axis) + len(self.form.cross_terms)
        # The inclusion equality check above proves every map entry is 0 or 1,
        # so the map digits are known and only the retained coefficient counts
        # bound the serialized output.
        if (
            direct_sum_output_digit_upper_bound(
                len(self.source_form.axis), source_support + output_support
            )
            > MAX_DIRECT_SUM_OUTPUT_DIGITS
        ):
            raise ValueError(
                "quadratic-form restriction exceeds the admitted output digit bound"
            )
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
            direct_sum_output_digit_upper_bound(
                total,
                support + len(self.form.axis) + len(self.form.cross_terms),
                map_component_digits=map_component_digits,
            )
            > MAX_DIRECT_SUM_OUTPUT_DIGITS
        ):
            raise ValueError(
                "direct-sum result exceeds the admitted output digit bound"
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
    "MAX_DIRECT_SUM_OUTPUT_DIGITS",
    "QuadraticFormDirectSumRequest",
    "QuadraticFormDirectSumResult",
    "direct_sum_output_digit_upper_bound",
]
