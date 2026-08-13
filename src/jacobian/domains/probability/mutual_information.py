"""Thin wire binding for exact finite-table mutual information."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ContractModel
from jacobian.domains._examples import example
from jacobian.math.probability.mutual_information import (
    FiniteJointTable,
    MutualInformationResult,
    mutual_information,
)
from jacobian.math.probability.values import (
    MAX_FINITE_JOINT_TABLE_CELLS,
    MAX_FINITE_JOINT_TABLE_COLUMNS,
    MAX_FINITE_JOINT_TABLE_ROWS,
    MAX_INPUT_RATIONAL_DIGITS,
    MAX_MUTUAL_INFORMATION_LIKELIHOOD_RATIO_DIGITS,
    MAX_MUTUAL_INFORMATION_MARGINAL_DIGITS,
    MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import OperationRefusalError

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
        ):
            raise ValueError(
                "joint-table probability exceeds the "
                f"{MAX_INPUT_RATIONAL_DIGITS}-digit bound"
            )


def _bound_raw_probability_row(row: object) -> int:
    if not isinstance(row, (list, tuple)):
        return 0
    if len(row) > MAX_FINITE_JOINT_TABLE_COLUMNS:
        raise ValueError("joint table exceeds the bounded column count")
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
        raise ValueError("joint table exceeds the bounded row count")
    cell_count = 0
    for row in raw_table:
        cell_count += _bound_raw_probability_row(row)
        if cell_count > MAX_FINITE_JOINT_TABLE_CELLS:
            raise ValueError("joint table exceeds the bounded cell count")
    return value


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
        ):
            raise ValueError(f"{label} exceeds the {max_digits}-digit bound")


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
    certificate = value.get("log_product_certificate")
    if isinstance(certificate, Mapping):
        _bound_raw_rational(
            certificate.get("product"),
            max_digits=MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
            label="mutual-information certificate product",
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
        raise ValueError("joint-table row count must match row labels")
    if any(len(row) != len(column_labels) for row in probabilities):
        raise ValueError("joint-table rows must match column labels")
    if len(row_labels) * len(column_labels) > MAX_FINITE_JOINT_TABLE_CELLS:
        raise ValueError("joint table exceeds the bounded cell count")


def _require_native_probability_values(
    probabilities: tuple[tuple[Fraction, ...], ...],
) -> None:
    total = Fraction()
    for row in probabilities:
        for probability in row:
            if type(probability) is not Fraction:
                raise TypeError("native joint-table probabilities must be Fractions")
            if probability < 0:
                raise ValueError("joint-table probabilities must be nonnegative")
            total += probability
    if total != 1:
        raise ValueError("joint-table probabilities must sum exactly to 1")


class FiniteJointTableMutualInformationRequest(ContractModel):
    """Canonical wire request for one bounded normalized rational joint table."""

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
        return _bound_raw_probability_matrix(value)

    @model_validator(mode="after")
    def require_normalized_rectangular_table(self) -> Self:
        if len(set(self.row_labels)) != len(self.row_labels):
            raise ValueError("joint-table row labels must be unique")
        if len(set(self.column_labels)) != len(self.column_labels):
            raise ValueError("joint-table column labels must be unique")
        _require_native_probability_shape(
            self.row_labels,
            self.column_labels,
            self.probabilities,
        )
        total = Fraction()
        for row in self.probabilities:
            for probability in row:
                require_bounded_rational(
                    probability,
                    max_digits=MAX_INPUT_RATIONAL_DIGITS,
                    label="joint-table probability",
                )
                native = probability.as_fraction()
                if native < 0:
                    raise ValueError("joint-table probabilities must be nonnegative")
                total += native
        if total != 1:
            raise ValueError("joint-table probabilities must sum exactly to 1")
        return self

    def as_native(self) -> FiniteJointTable:
        """Parse the canonical wire request once into native mathematical values."""

        return FiniteJointTable(
            row_labels=self.row_labels,
            column_labels=self.column_labels,
            probabilities=tuple(
                tuple(probability.as_fraction() for probability in row)
                for row in self.probabilities
            ),
            log_base=self.log_base,
        )


class FiniteJointLikelihoodRatio(ContractModel):
    """Canonical wire encoding of one positive-support native term."""

    row_index: StrictInt = Field(ge=0, lt=MAX_FINITE_JOINT_TABLE_ROWS)
    column_index: StrictInt = Field(ge=0, lt=MAX_FINITE_JOINT_TABLE_COLUMNS)
    probability: CanonicalRational
    row_marginal: CanonicalRational
    column_marginal: CanonicalRational
    likelihood_ratio: CanonicalRational


class MutualInformationLogProductCertificate(ContractModel):
    """Canonical wire form of ``scale * I = log_base(product)``."""

    scale: CanonicalInteger
    product: CanonicalRational
    identity: Literal["SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT"] = (
        "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT"
    )

    @model_validator(mode="after")
    def require_positive_scale_and_product(self) -> Self:
        if parse_canonical_integer(self.scale) <= 0:
            raise ValueError("mutual-information certificate scale must be positive")
        if self.product.as_fraction() <= 0:
            raise ValueError("mutual-information certificate product must be positive")
        return self


FiniteJointPositiveSupport = Annotated[
    tuple[FiniteJointLikelihoodRatio, ...],
    Field(
        min_length=1,
        max_length=MAX_FINITE_JOINT_TABLE_CELLS,
    ),
]


class FiniteJointTableMutualInformationResult(ContractModel):
    """Canonical wire projection of one native mutual-information result."""

    row_marginals: FiniteJointRowMarginals
    column_marginals: FiniteJointColumnMarginals
    positive_support: FiniteJointPositiveSupport
    log_base: StrictInt = Field(ge=2, le=36)
    log_product_certificate: MutualInformationLogProductCertificate
    exact_value: CanonicalRational | None = None
    sign: Literal["ZERO", "POSITIVE"]
    zero_cell_convention: Literal["ZERO_MASS_TERMS_OMITTED"] = "ZERO_MASS_TERMS_OMITTED"

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result_collections(cls, value: Any) -> Any:
        """Reject impossible candidates before parsing their nested item models."""

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
                raise ValueError(f"{field_name} exceeds the bounded result cardinality")
        _bound_raw_result_rationals(value)
        certificate = value.get("log_product_certificate")
        if isinstance(certificate, Mapping):
            scale = certificate.get("scale")
            if isinstance(scale, str) and len(scale.lstrip("-")) > 309:
                raise ValueError(
                    "mutual-information certificate scale exceeds the replay bound"
                )
        return value

    @model_validator(mode="after")
    def bind_certificate(self) -> Self:
        positions = tuple(
            (term.row_index, term.column_index) for term in self.positive_support
        )
        if positions != tuple(sorted(set(positions))):
            raise ValueError("positive support must be unique and row-major ordered")
        for term in self.positive_support:
            if term.row_index >= len(self.row_marginals):
                raise ValueError("positive support row index lies outside the result")
            if term.column_index >= len(self.column_marginals):
                raise ValueError(
                    "positive support column index lies outside the result"
                )
            if term.row_marginal != self.row_marginals[term.row_index]:
                raise ValueError("positive support row marginal is inconsistent")
            if term.column_marginal != self.column_marginals[term.column_index]:
                raise ValueError("positive support column marginal is inconsistent")
        product = self.log_product_certificate.product.as_fraction()
        if self.sign != ("ZERO" if product == 1 else "POSITIVE"):
            raise ValueError("mutual-information sign must match the exact product")
        if product < 1:
            raise ValueError("mutual-information product contradicts nonnegativity")
        return self

    @classmethod
    def from_native(
        cls,
        result: MutualInformationResult,
    ) -> FiniteJointTableMutualInformationResult:
        """Encode a native result exactly at the operation boundary."""

        return cls(
            row_marginals=tuple(
                CanonicalRational.from_fraction(value) for value in result.row_marginals
            ),
            column_marginals=tuple(
                CanonicalRational.from_fraction(value)
                for value in result.column_marginals
            ),
            positive_support=tuple(
                FiniteJointLikelihoodRatio(
                    row_index=term.row_index,
                    column_index=term.column_index,
                    probability=CanonicalRational.from_fraction(term.probability),
                    row_marginal=CanonicalRational.from_fraction(term.row_marginal),
                    column_marginal=CanonicalRational.from_fraction(
                        term.column_marginal
                    ),
                    likelihood_ratio=CanonicalRational.from_fraction(
                        term.likelihood_ratio
                    ),
                )
                for term in result.positive_support
            ),
            log_base=result.log_base,
            log_product_certificate=MutualInformationLogProductCertificate(
                scale=format_canonical_integer(result.certificate.scale),
                product=CanonicalRational.from_fraction(result.certificate.product),
            ),
            exact_value=(
                None
                if result.exact_value is None
                else CanonicalRational.from_fraction(result.exact_value)
            ),
            sign=result.sign,
        )


_INVALID_REQUEST = OperationDiagnostic(
    code="INVALID_FINITE_JOINT_TABLE_REQUEST",
    stage="finite_joint_table_input_validation",
    message="Input does not satisfy the bounded normalized rational joint-table contract.",
    hint=(
        "Supply unique ordered labels, a nonnegative rational table with at most "
        "64 cells summing exactly to one, and a log base from 2 to 36."
    ),
)


def _execute(
    request: FiniteJointTableMutualInformationRequest,
) -> FiniteJointTableMutualInformationResult:
    try:
        return FiniteJointTableMutualInformationResult.from_native(
            mutual_information(request.as_native())
        )
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        raise OperationRefusalError(
            OperationDiagnostic(
                code="MUTUAL_INFORMATION_CERTIFICATE_BOUND_EXCEEDED",
                stage="finite_joint_mutual_information",
                message=str(exc),
                hint="Reduce rational denominator sizes or the positive support.",
            )
        ) from exc


MUTUAL_INFORMATION_OPERATION = inline_operation(
    OperationDeclaration(
        operation_id="probability.joint.mutual_information.compute",
        version="1",
        title="Exact finite-table mutual information certificate",
        description=(
            "Compute ordered marginals and every positive-support likelihood "
            "ratio for one bounded normalized rational joint table. Return the "
            "exact identity scale*I=log_base(product), without floating point."
        ),
        request_type=FiniteJointTableMutualInformationRequest,
        result_type=FiniteJointTableMutualInformationResult,
        execute=_execute,
        tags=(
            "probability",
            "information-theory",
            "mutual-information",
            "finite",
            "exact",
            "certificate",
        ),
        invalid_request=_INVALID_REQUEST,
        examples=(
            example(
                "perfectly_correlated_fair_bits",
                "Compute exact base-two mutual information for two identical fair bits.",
                {
                    "row_labels": ["0", "1"],
                    "column_labels": ["0", "1"],
                    "probabilities": [
                        [
                            {"num": "1", "den": "2"},
                            {"num": "0", "den": "1"},
                        ],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "2"},
                        ],
                    ],
                    "log_base": 2,
                },
            ),
        ),
    ),
)

__all__ = [
    "MUTUAL_INFORMATION_OPERATION",
    "FiniteJointTableMutualInformationRequest",
    "FiniteJointTableMutualInformationResult",
]
