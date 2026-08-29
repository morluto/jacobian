"""Exact rational linear-system contracts."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

MAX_LINEAR_DIMENSION = 8_192
MAX_LINEAR_NONZERO_COUNT = 32_768
MAX_LINEAR_SCALAR_WORK = 100_000_000
MAX_LINEAR_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_RATIONAL_DIGITS = 256

LinearVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$",
        strict=True,
    ),
]


def _require_bounded_rationals(values: tuple[CanonicalRational, ...]) -> None:
    for value in values:
        try:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="linear-system rational",
            )
        except ValueError as error:
            raise _validation_error("rational_bound", str(error)) from error


class LinearRationalCoefficient(StrictModel):
    """One nonzero coefficient at a stable zero-based row and column."""

    row: int = Field(ge=0, le=MAX_LINEAR_DIMENSION - 1)
    column: int = Field(ge=0, le=MAX_LINEAR_DIMENSION - 1)
    value: CanonicalRational


class LinearRationalSystem(StrictModel):
    """One declared finite system ``A x = b`` over exact rationals."""

    domain: Literal["QQ"] = "QQ"
    relation: Literal["AX_EQUALS_B"] = "AX_EQUALS_B"
    variables: tuple[LinearVariableName, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    row_count: int = Field(ge=1, le=MAX_LINEAR_DIMENSION)
    coefficients: tuple[LinearRationalCoefficient, ...] = Field(
        default=(), max_length=MAX_LINEAR_NONZERO_COUNT
    )
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_sparse_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        variables = data.get("variables")
        coefficients = data.get("coefficients")
        rhs = data.get("rhs")
        if (
            isinstance(variables, (list, tuple))
            and len(variables) > MAX_LINEAR_DIMENSION
        ):
            raise _validation_error(
                "budget_exceeded",
                f"linear systems have at most {MAX_LINEAR_DIMENSION} variables",
            )
        if isinstance(coefficients, (list, tuple)) and len(coefficients) > (
            MAX_LINEAR_NONZERO_COUNT
        ):
            raise _validation_error(
                "budget_exceeded",
                f"linear systems store at most {MAX_LINEAR_NONZERO_COUNT} nonzeros",
            )
        if isinstance(rhs, (list, tuple)) and len(rhs) > MAX_LINEAR_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                f"linear systems have at most {MAX_LINEAR_DIMENSION} rows",
            )
        raw_values: list[object] = list(rhs) if isinstance(rhs, (list, tuple)) else []
        if isinstance(coefficients, (list, tuple)):
            raw_values.extend(
                item.get("value") for item in coefficients if isinstance(item, dict)
            )
        for value in raw_values:
            components = (
                (value.get("num"), value.get("den"))
                if isinstance(value, dict)
                else (value,)
            )
            if any(
                isinstance(component, (str, int))
                and len(str(component).lstrip("-")) > MAX_RATIONAL_DIGITS
                for component in components
            ):
                raise _validation_error(
                    "rational_bound",
                    f"linear-system rationals are limited to {MAX_RATIONAL_DIGITS} decimal digits",
                )
        return data

    @model_validator(mode="after")
    def require_matching_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "budget_exceeded", "linear-system variable names must be unique"
            )
        if self.row_count != len(self.rhs):
            raise _validation_error(
                "budget_exceeded",
                "the right-hand side length must equal the coefficient row count",
            )
        coordinates = tuple((item.row, item.column) for item in self.coefficients)
        if coordinates != tuple(sorted(set(coordinates))):
            raise _validation_error(
                "shape_mismatch",
                "sparse coefficient coordinates must be unique and row-major sorted",
            )
        if any(
            item.row >= self.row_count or item.column >= len(self.variables)
            for item in self.coefficients
        ):
            raise _validation_error(
                "shape_mismatch", "sparse coefficient coordinates exceed system axes"
            )
        if any(item.value.as_fraction() == 0 for item in self.coefficients):
            raise _validation_error(
                "shape_mismatch", "sparse coefficients must not store explicit zeros"
            )
        _require_bounded_rationals(
            tuple(item.value for item in self.coefficients) + self.rhs
        )
        _require_execution_envelope(self)
        return self


def _decimal_digits_from_bits(bits: int) -> int:
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _minor_component_bits(
    rows: tuple[tuple[Fraction, ...], ...], *, column_count: int
) -> int:
    rank_bound = min(len(rows), column_count)
    numerator_bits: list[int] = []
    denominator_bits: list[int] = []
    for row in rows:
        denominator = 1
        for value in row:
            factor = value.denominator // gcd(denominator, value.denominator)
            denominator *= factor
            if _decimal_digits_from_bits(denominator.bit_length()) > (
                MAX_CANONICAL_RATIONAL_DIGITS
            ):
                return MAX_CANONICAL_RATIONAL_DIGITS * 4
        largest = max(
            (
                abs(value.numerator) * (denominator // value.denominator)
                for value in row
            ),
            default=0,
        )
        numerator_bits.append(
            largest.bit_length() + (column_count.bit_length() + 1) // 2
        )
        denominator_bits.append(denominator.bit_length())
    return sum(sorted(numerator_bits, reverse=True)[:rank_bound]) + sum(
        sorted(denominator_bits, reverse=True)[:rank_bound]
    )


def _require_execution_envelope(system: LinearRationalSystem) -> None:
    rows = system.row_count
    columns = len(system.variables)
    scalar_digits = max(
        len(component.lstrip("-"))
        for value in tuple(item.value for item in system.coefficients) + system.rhs
        for component in (value.num, value.den)
    )
    solution_work = rows * (columns + 1) * min(rows, columns + 1)
    witness_work = (columns + 1) * (rows + 1) * min(columns + 1, rows + 1)
    if max(solution_work, witness_work) * scalar_digits > MAX_LINEAR_SCALAR_WORK:
        raise _validation_error(
            "budget_exceeded",
            "sparse exact linear algebra exceeds the "
            f"{MAX_LINEAR_SCALAR_WORK:,}-unit scalar-work budget",
        )

    solution_values: dict[int, list[Fraction]] = {row: [] for row in range(rows)}
    witness_values: dict[int, list[Fraction]] = {
        column: [] for column in range(columns)
    }
    for item in system.coefficients:
        fraction = item.value.as_fraction()
        solution_values[item.row].append(fraction)
        witness_values[item.column].append(fraction)
    for row, bound in enumerate(system.rhs):
        if bound.as_fraction():
            solution_values[row].append(bound.as_fraction())
    solution_rows = tuple(tuple(solution_values[row]) for row in range(rows))
    witness_rows = (
        *(tuple(witness_values[column]) for column in range(columns)),
        tuple(value.as_fraction() for value in system.rhs if value.as_fraction()),
    )
    result_digits = max(
        _decimal_digits_from_bits(
            _minor_component_bits(solution_rows, column_count=columns + 1)
        ),
        _decimal_digits_from_bits(
            _minor_component_bits(witness_rows, column_count=rows + 1)
        ),
    )
    if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "sparse exact linear algebra exceeds the canonical result-height bound",
        )
    source_bytes = sum(len(name) + 3 for name in system.variables)
    source_bytes += sum(
        len(item.value.num)
        + len(item.value.den)
        + len(str(item.row))
        + len(str(item.column))
        + 64
        for item in system.coefficients
    )
    source_bytes += sum(len(value.num) + len(value.den) + 24 for value in system.rhs)
    result_bytes = source_bytes + max(rows, columns) * (2 * result_digits + 32) + 4_096
    if result_bytes > MAX_LINEAR_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded",
            "sparse exact linear algebra exceeds the "
            f"{MAX_LINEAR_RESULT_BYTES:,}-byte transport result bound",
        )


class LinearRationalSolutionFindRequest(StrictModel):
    """Ask for one exact solution of a rational linear system."""

    system: LinearRationalSystem


class LinearRationalSolutionResult(StrictModel):
    """One exact solution outcome bound to its declared source system.

    A solution carries one coordinate per declared variable. The bounded owner
    kernel establishes ``A x = b`` exactly over QQ.
    """

    system: LinearRationalSystem
    status: Literal["SOLUTION", "INCONSISTENT"] = "SOLUTION"
    values: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="after")
    def bind_solution_to_source(self) -> Self:
        produced = self.status == "SOLUTION"
        if produced != (self.values is not None):
            raise _validation_error(
                "invariant_mismatch",
                "solution values must agree with the result status",
            )
        if self.values is None:
            return self
        if len(self.values) != len(self.system.variables):
            raise _validation_error(
                "budget_exceeded",
                "solution length must equal the source variable count",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        system: LinearRationalSystem,
        status: Literal["SOLUTION", "INCONSISTENT"],
        values: tuple[CanonicalRational, ...] | None = None,
    ) -> Self:
        """Construct a result from the owner-local bounded solver output."""

        return cls.model_construct(system=system, status=status, values=values)


class LinearRationalInconsistencyResult(StrictModel):
    """One exact inconsistency outcome bound to its declared source system.

    An admitted separating witness ``y`` carries one coordinate per source
    row. The bounded owner kernel establishes ``y^T A = 0`` and the recorded
    nonzero ``y^T b`` pairing.
    """

    system: LinearRationalSystem
    status: Literal["INCONSISTENT", "CONSISTENT"] = "INCONSISTENT"
    left_witness: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational | None = None

    @model_validator(mode="after")
    def bind_witness_to_source(self) -> Self:
        produced = self.status == "INCONSISTENT"
        if produced != (self.left_witness is not None and self.rhs_pairing is not None):
            raise _validation_error(
                "invariant_mismatch",
                "inconsistency witness must agree with the result status",
            )
        if self.left_witness is None or self.rhs_pairing is None:
            return self
        if len(self.left_witness) != len(self.system.rhs):
            raise _validation_error(
                "shape_mismatch", "witness length must equal the source row count"
            )
        if self.rhs_pairing.as_fraction() == 0:
            raise _validation_error(
                "status_mismatch", "separating witness must have a nonzero pairing"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        system: LinearRationalSystem,
        status: Literal["INCONSISTENT", "CONSISTENT"],
        left_witness: tuple[CanonicalRational, ...] | None = None,
        rhs_pairing: CanonicalRational | None = None,
    ) -> Self:
        """Construct a result from the owner-local bounded solver output."""

        return cls.model_construct(
            system=system,
            status=status,
            left_witness=left_witness,
            rhs_pairing=rhs_pairing,
        )


class LinearRationalInconsistencyFindRequest(StrictModel):
    """Ask whether a rational linear system is inconsistent."""

    system: LinearRationalSystem


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
