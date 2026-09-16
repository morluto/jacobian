"""Wire contracts for finite-field Jacobian syzygy checks (#964)."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.finite_fields._algebraic_sets import AlgebraicPolynomial

MAX_JACOBIAN_SYZYGY_VARS = 4
MAX_JACOBIAN_SYZYGY_TERMS = 64
MAX_JACOBIAN_SYZYGY_DEGREE = 32
MAX_JACOBIAN_SYZYGY_ROWS = 16
MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS = 1
MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS = 8_192
MAX_JACOBIAN_SYZYGY_WORK = 1_000_000

JacobianSyzygyStatus = Literal["VERIFIED", "REJECTED"]


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _is_zero_polynomial(polynomial: AlgebraicPolynomial) -> bool:
    return len(polynomial.terms) == 1 and polynomial.terms[0].coefficient.is_zero


class _JacobianSyzygyRequestBase(StrictModel):
    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)


class JacobianSyzygyCheckRequest(_JacobianSyzygyRequestBase):
    """Check candidate syzygy rows of one char-p Jacobian in a quotient ring."""

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None
    rows: tuple[tuple[AlgebraicPolynomial, ...], ...] = Field(
        min_length=1, max_length=MAX_JACOBIAN_SYZYGY_ROWS
    )

    @model_validator(mode="after")
    def require_row_and_regime_shape(self) -> Self:
        nvars = len(self.polynomial.variable_axis.labels)
        if any(len(row) != nvars for row in self.rows):
            raise _error(
                "finite_field.jacobian_syzygy_row_length",
                "every syzygy row must have one entry per declared variable",
            )
        if self.ideal_generators and self.reduction_variable is None:
            raise _error(
                "finite_field.jacobian_syzygy_reduction_variable",
                "a nonempty principal ideal requires a declared reduction variable",
            )
        if not self.ideal_generators and self.reduction_variable is not None:
            raise _error(
                "finite_field.jacobian_syzygy_reduction_variable",
                "an empty ideal must not declare a reduction variable",
            )
        if (
            self.reduction_variable is not None
            and self.reduction_variable not in self.polynomial.variable_axis.labels
        ):
            raise _error(
                "finite_field.jacobian_syzygy_reduction_variable",
                "the reduction variable must be a label of the variable axis",
            )
        return self


class JacobianSyzygyRowReduction(StrictModel):
    """One row's exact normal-form remainder and reduction step count."""

    row_index: int = Field(ge=0)
    remainder: AlgebraicPolynomial
    reduction_steps: int = Field(ge=0)
    verified: bool

    @model_validator(mode="after")
    def require_consistent_verdict(self) -> Self:
        if self.verified != _is_zero_polynomial(self.remainder):
            raise _error(
                "finite_field.jacobian_syzygy_ledger_verdict",
                "a ledger verdict must agree with its retained remainder",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        row_index: int,
        remainder: AlgebraicPolynomial,
        reduction_steps: int,
    ) -> Self:
        return cls.model_construct(
            row_index=row_index,
            remainder=remainder,
            reduction_steps=reduction_steps,
            verified=_is_zero_polynomial(remainder),
        )


class JacobianSyzygyCheckResult(_JacobianSyzygyRequestBase):
    """Source-bound char-p Jacobian syzygy verdict with a per-row ledger."""

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None
    jacobian: tuple[AlgebraicPolynomial, ...] = Field(
        min_length=1, max_length=MAX_JACOBIAN_SYZYGY_VARS
    )
    rows: tuple[tuple[AlgebraicPolynomial, ...], ...] = Field(
        min_length=1, max_length=MAX_JACOBIAN_SYZYGY_ROWS
    )
    ledger: tuple[JacobianSyzygyRowReduction, ...] = Field(
        min_length=1, max_length=MAX_JACOBIAN_SYZYGY_ROWS
    )
    status: JacobianSyzygyStatus
    first_failure: int | None = None

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        nvars = len(self.polynomial.variable_axis.labels)
        if len(self.jacobian) != nvars:
            raise _error(
                "finite_field.jacobian_syzygy_jacobian_shape",
                "the derived Jacobian must have one partial per variable",
            )
        if len(self.ledger) != len(self.rows):
            raise _error(
                "finite_field.jacobian_syzygy_ledger_shape",
                "the reduction ledger must retain one entry per candidate row",
            )
        if tuple(entry.row_index for entry in self.ledger) != tuple(
            range(len(self.rows))
        ):
            raise _error(
                "finite_field.jacobian_syzygy_ledger_order",
                "ledger entries must follow the candidate row order",
            )
        if any(len(row) != nvars for row in self.rows):
            raise _error(
                "finite_field.jacobian_syzygy_row_length",
                "every syzygy row must have one entry per declared variable",
            )
        if any(
            entry.remainder.presentation != self.polynomial.presentation
            or entry.remainder.variable_axis != self.polynomial.variable_axis
            for entry in self.ledger
        ):
            raise _error(
                "finite_field.jacobian_syzygy_remainder_parent",
                "remainders must share the source presentation and axis",
            )
        failures = tuple(
            index for index, entry in enumerate(self.ledger) if not entry.verified
        )
        expected_status: JacobianSyzygyStatus = (
            "VERIFIED" if not failures else "REJECTED"
        )
        if self.status != expected_status:
            raise _error(
                "finite_field.jacobian_syzygy_status_binding",
                "the status must agree with the retained ledger verdicts",
            )
        if self.first_failure != (failures[0] if failures else None):
            raise _error(
                "finite_field.jacobian_syzygy_first_failure_binding",
                "first_failure must name the first rejected ledger row",
            )
        if bool(self.ideal_generators) != (self.reduction_variable is not None):
            raise _error(
                "finite_field.jacobian_syzygy_reduction_variable",
                "the reduction variable binds exactly the principal-ideal regime",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: AlgebraicPolynomial,
        ideal_generators: tuple[AlgebraicPolynomial, ...],
        reduction_variable: str | None,
        jacobian: tuple[AlgebraicPolynomial, ...],
        rows: tuple[tuple[AlgebraicPolynomial, ...], ...],
        ledger: tuple[JacobianSyzygyRowReduction, ...],
    ) -> Self:
        failures = tuple(
            index for index, entry in enumerate(ledger) if not entry.verified
        )
        return cls.model_construct(
            polynomial=polynomial,
            ideal_generators=ideal_generators,
            reduction_variable=reduction_variable,
            jacobian=jacobian,
            rows=rows,
            ledger=ledger,
            status="VERIFIED" if not failures else "REJECTED",
            first_failure=failures[0] if failures else None,
        )


__all__ = [
    "MAX_JACOBIAN_SYZYGY_DEGREE",
    "MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS",
    "MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS",
    "MAX_JACOBIAN_SYZYGY_ROWS",
    "MAX_JACOBIAN_SYZYGY_TERMS",
    "MAX_JACOBIAN_SYZYGY_VARS",
    "MAX_JACOBIAN_SYZYGY_WORK",
    "JacobianSyzygyCheckRequest",
    "JacobianSyzygyCheckResult",
    "JacobianSyzygyRowReduction",
    "JacobianSyzygyStatus",
]
