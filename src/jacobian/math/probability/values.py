"""Canonical exact values for finite-table probability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel, canonicalize_json_containers

MAX_FINITE_JOINT_TABLE_ROWS = 16
MAX_FINITE_JOINT_TABLE_COLUMNS = 16
MAX_FINITE_JOINT_TABLE_CELLS = (
    MAX_FINITE_JOINT_TABLE_ROWS * MAX_FINITE_JOINT_TABLE_COLUMNS
)
MAX_INPUT_RATIONAL_DIGITS = 256
MAX_MUTUAL_INFORMATION_SCALE_BITS = 1_024
MAX_MUTUAL_INFORMATION_POWER_COST_BITS = 32_768
MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS = (
    MAX_MUTUAL_INFORMATION_POWER_COST_BITS * 30103 // 100000 + 1
)

# These are derived from the largest producer support values.  A marginal is
# a sum of at most 16 input rationals; a likelihood ratio divides one input
# cell by two such marginals.  The small slack covers decimal addition.
MAX_MUTUAL_INFORMATION_MARGINAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_FINITE_JOINT_TABLE_ROWS + 2
)
MAX_MUTUAL_INFORMATION_LIKELIHOOD_RATIO_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * (1 + 2 * MAX_FINITE_JOINT_TABLE_ROWS) + 4
)
_MAX_INPUT_RATIONAL_MAGNITUDE = 10**MAX_INPUT_RATIONAL_DIGITS


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.mutual_information_invariant", message)


FiniteJointLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, strict=True),
]
FiniteJointProbabilityRow = Annotated[
    tuple[CanonicalRational, ...],
    Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_COLUMNS,
    ),
]
FiniteJointRowMarginals = Annotated[
    tuple[CanonicalRational, ...],
    Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_ROWS,
    ),
]
FiniteJointColumnMarginals = Annotated[
    tuple[CanonicalRational, ...],
    Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_COLUMNS,
    ),
]


def _bound_raw_probability_cell(cell: object) -> None:
    if not isinstance(cell, Mapping):
        return
    for component in ("num", "den"):
        raw_component = cell.get(component)
        if (
            isinstance(raw_component, str)
            and len(raw_component.lstrip("-")) > MAX_INPUT_RATIONAL_DIGITS
        ) or (
            type(raw_component) is int
            and abs(raw_component) >= 10**MAX_INPUT_RATIONAL_DIGITS
        ):
            raise _validation_error(
                "joint-table probability exceeds the "
                f"{MAX_INPUT_RATIONAL_DIGITS}-digit bound"
            )


def _bound_raw_probability_row(row: object) -> int:
    if not isinstance(row, (list, tuple)):
        return 0
    if len(row) > MAX_FINITE_JOINT_TABLE_COLUMNS:
        raise _validation_error("joint table exceeds the bounded column count")
    for cell in row:
        _bound_raw_probability_cell(cell)
    return len(row)


def _bound_raw_probability_matrix(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    raw_table = value.get("probabilities")
    if not isinstance(raw_table, (list, tuple)):
        return value
    if len(raw_table) > MAX_FINITE_JOINT_TABLE_ROWS:
        raise _validation_error("joint table exceeds the bounded row count")
    for row in raw_table:
        _bound_raw_probability_row(row)
    prepared = dict(value)
    for field_name in ("row_labels", "column_labels"):
        raw_labels = prepared.get(field_name)
        if isinstance(raw_labels, list):
            prepared[field_name] = tuple(raw_labels)
    prepared["probabilities"] = tuple(
        tuple(row) if isinstance(row, list) else row for row in raw_table
    )
    return prepared


def _bound_raw_rational(
    value: object,
    *,
    max_digits: int,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    for component in ("num", "den"):
        raw_component = value.get(component)
        if (
            isinstance(raw_component, str)
            and len(raw_component.lstrip("-")) > max_digits
        ) or (type(raw_component) is int and abs(raw_component) >= 10**max_digits):
            raise _validation_error(f"{label} exceeds the {max_digits}-digit bound")


def _bound_raw_result_rationals(value: Mapping[str, object]) -> None:
    for field_name in ("row_marginals", "column_marginals"):
        raw_values = value.get(field_name)
        if isinstance(raw_values, (list, tuple)):
            for index, raw_value in enumerate(raw_values):
                _bound_raw_rational(
                    raw_value,
                    max_digits=MAX_MUTUAL_INFORMATION_MARGINAL_DIGITS,
                    label=f"{field_name}[{index}]",
                )
    raw_support = value.get("positive_support")
    if isinstance(raw_support, (list, tuple)):
        for index, raw_term in enumerate(raw_support):
            if not isinstance(raw_term, Mapping):
                continue
            for field_name in (
                "probability",
                "row_marginal",
                "column_marginal",
                "likelihood_ratio",
            ):
                _bound_raw_rational(
                    raw_term.get(field_name),
                    max_digits=(
                        MAX_INPUT_RATIONAL_DIGITS
                        if field_name == "probability"
                        else (
                            MAX_MUTUAL_INFORMATION_LIKELIHOOD_RATIO_DIGITS
                            if field_name == "likelihood_ratio"
                            else MAX_MUTUAL_INFORMATION_MARGINAL_DIGITS
                        )
                    ),
                    label=f"positive_support[{index}].{field_name}",
                )
    logarithmic_value = value.get("exact_logarithmic_value")
    if isinstance(logarithmic_value, Mapping):
        _bound_raw_rational(
            logarithmic_value.get("product"),
            max_digits=MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
            label="mutual-information logarithmic value product",
        )
    _bound_raw_rational(
        value.get("exact_value"),
        max_digits=MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
        label="mutual-information exact value",
    )


def _require_native_probability_shape(
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    probabilities: tuple[tuple[object, ...], ...],
) -> None:
    if len(probabilities) != len(row_labels):
        raise _validation_error("joint-table row count must match row labels")
    if any(len(row) != len(column_labels) for row in probabilities):
        raise _validation_error("joint-table rows must match column labels")


class FiniteJointTable(StrictModel):
    """One canonical labelled rational joint table; normalization is admitted by consumers."""

    row_labels: tuple[FiniteJointLabel, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_ROWS,
    )
    column_labels: tuple[FiniteJointLabel, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_COLUMNS,
    )
    probabilities: tuple[FiniteJointProbabilityRow, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_ROWS,
    )
    log_base: StrictInt = Field(default=2, ge=2, le=36)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_probability_matrix(cls, value: Any) -> Any:
        """Reject oversized collections before parsing any rational cell model."""

        value = canonicalize_json_containers(value)
        return _bound_raw_probability_matrix(value)

    @model_validator(mode="after")
    def require_structural_table(self) -> Self:
        _require_native_probability_shape(
            self.row_labels, self.column_labels, self.probabilities
        )
        if len(set(self.row_labels)) != len(self.row_labels) or len(
            set(self.column_labels)
        ) != len(self.column_labels):
            raise _validation_error("joint-table labels must be unique on each axis")
        for row in self.probabilities:
            for value in row:
                if value.num < 0:
                    raise _validation_error(
                        "joint-table probabilities must be nonnegative"
                    )
                if (
                    abs(value.num) >= _MAX_INPUT_RATIONAL_MAGNITUDE
                    or value.den >= _MAX_INPUT_RATIONAL_MAGNITUDE
                ):
                    raise _validation_error(
                        "joint-table probability exceeds the 256-digit bound"
                    )
        return self


class MutualInformationTerm(StrictModel):
    """One claimed positive-support contribution on the result's labelled axes."""

    row_index: StrictInt = Field(ge=0, lt=MAX_FINITE_JOINT_TABLE_ROWS)
    column_index: StrictInt = Field(ge=0, lt=MAX_FINITE_JOINT_TABLE_COLUMNS)
    probability: CanonicalRational
    row_marginal: CanonicalRational
    column_marginal: CanonicalRational
    likelihood_ratio: CanonicalRational


class MutualInformationLogRepresentation(StrictModel):
    """Canonical wire form of ``scale * I = log_base(product)``."""

    scale: ExactInteger
    product: CanonicalRational
    identity: Literal["SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT"] = (
        "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT"
    )

    @model_validator(mode="after")
    def require_positive_scale_and_product(self) -> Self:
        if self.scale.bit_length() > MAX_MUTUAL_INFORMATION_SCALE_BITS:
            raise _validation_error(
                "mutual-information logarithmic value scale exceeds the bound"
            )
        if self.scale <= 0:
            raise _validation_error(
                "mutual-information logarithmic value scale must be positive"
            )
        if self.product.as_fraction() <= 0:
            raise _validation_error(
                "mutual-information logarithmic value product must be positive"
            )
        return self


FiniteJointPositiveSupport = Annotated[
    tuple[MutualInformationTerm, ...],
    Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_CELLS,
    ),
]


class MutualInformationResult(StrictModel):
    """Exact mutual information retaining both ordered axes, including zero marginals."""

    row_labels: tuple[FiniteJointLabel, ...] = Field(
        min_length=1, max_length=MAX_FINITE_JOINT_TABLE_ROWS
    )
    column_labels: tuple[FiniteJointLabel, ...] = Field(
        min_length=1, max_length=MAX_FINITE_JOINT_TABLE_COLUMNS
    )
    row_marginals: FiniteJointRowMarginals
    column_marginals: FiniteJointColumnMarginals
    positive_support: FiniteJointPositiveSupport
    log_base: StrictInt = Field(ge=2, le=36)
    exact_logarithmic_value: MutualInformationLogRepresentation
    exact_value: CanonicalRational | None = None
    sign: Literal["ZERO", "POSITIVE"]
    zero_cell_convention: Literal["ZERO_MASS_TERMS_OMITTED"] = "ZERO_MASS_TERMS_OMITTED"

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result_collections(cls, value: Any) -> Any:
        """Reject impossible candidates before parsing their nested item models."""

        value = canonicalize_json_containers(value)

        if not isinstance(value, Mapping):
            return value
        bounds = {
            "row_marginals": MAX_FINITE_JOINT_TABLE_ROWS,
            "column_marginals": MAX_FINITE_JOINT_TABLE_COLUMNS,
            "positive_support": MAX_FINITE_JOINT_TABLE_CELLS,
        }
        for field_name, maximum in bounds.items():
            raw = value.get(field_name)
            if isinstance(raw, (list, tuple)) and len(raw) > maximum:
                raise _validation_error(
                    f"{field_name} exceeds the bounded result cardinality"
                )
        _bound_raw_result_rationals(value)
        logarithmic_value = value.get("exact_logarithmic_value")
        if isinstance(logarithmic_value, Mapping):
            scale = logarithmic_value.get("scale")
            if isinstance(scale, str) and len(scale.lstrip("-")) > 309:
                raise _validation_error(
                    "mutual-information logarithmic value scale exceeds the bound"
                )
        return value

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if len(self.row_labels) != len(self.row_marginals) or len(
            self.column_labels
        ) != len(self.column_marginals):
            raise _validation_error("marginals must retain one entry per axis label")
        if len(set(self.row_labels)) != len(self.row_labels) or len(
            set(self.column_labels)
        ) != len(self.column_labels):
            raise _validation_error("result labels must be unique on each axis")
        positions = tuple(
            (term.row_index, term.column_index) for term in self.positive_support
        )
        if positions != tuple(sorted(set(positions))):
            raise _validation_error(
                "positive support must be unique and row-major ordered"
            )
        for term in self.positive_support:
            if term.row_index >= len(self.row_marginals):
                raise _validation_error(
                    "positive support row index lies outside the result"
                )
            if term.column_index >= len(self.column_marginals):
                raise _validation_error(
                    "positive support column index lies outside the result"
                )
            if term.row_marginal != self.row_marginals[term.row_index]:
                raise _validation_error("positive support row marginal is inconsistent")
            if term.column_marginal != self.column_marginals[term.column_index]:
                raise _validation_error(
                    "positive support column marginal is inconsistent"
                )
        product = self.exact_logarithmic_value.product.as_fraction()
        if self.sign != ("ZERO" if product == 1 else "POSITIVE"):
            raise _validation_error(
                "mutual-information sign must match the exact product"
            )
        if product < 1:
            raise _validation_error(
                "mutual-information product contradicts nonnegativity"
            )
        return self


__all__ = [
    "FiniteJointTable",
    "MutualInformationLogRepresentation",
    "MutualInformationResult",
    "MutualInformationTerm",
]
