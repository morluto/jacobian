"""Exact rational linear-system contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.values import (
    MAX_SPARSE_RATIONAL_MATRIX_AXIS,
    MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
    SparseRationalMatrix,
)

MAX_LINEAR_DIMENSION = MAX_SPARSE_RATIONAL_MATRIX_AXIS
MAX_LINEAR_NONZERO_COUNT = MAX_SPARSE_RATIONAL_MATRIX_NONZEROS
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


class LinearRationalSystem(StrictModel):
    """One declared finite system ``A x = b`` over exact rationals."""

    domain: Literal["QQ"] = "QQ"
    relation: Literal["AX_EQUALS_B"] = "AX_EQUALS_B"
    variables: tuple[LinearVariableName, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    coefficients: SparseRationalMatrix
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
        coefficient_entries = (
            coefficients.get("entries") if isinstance(coefficients, dict) else None
        )
        if (
            isinstance(coefficient_entries, (list, tuple))
            and len(coefficient_entries) > MAX_LINEAR_NONZERO_COUNT
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
        if isinstance(coefficient_entries, (list, tuple)):
            raw_values.extend(
                item.get("value")
                for item in coefficient_entries
                if isinstance(item, dict)
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
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_matching_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "budget_exceeded", "linear-system variable names must be unique"
            )
        if self.coefficients.row_count != len(self.rhs):
            raise _validation_error(
                "budget_exceeded",
                "the right-hand side length must equal the coefficient row count",
            )
        if self.coefficients.column_count != len(self.variables):
            raise _validation_error(
                "budget_exceeded",
                "the coefficient column count must equal the declared variable count",
            )
        _require_bounded_rationals(
            tuple(item.value for item in self.coefficients.entries) + self.rhs
        )
        return self


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
