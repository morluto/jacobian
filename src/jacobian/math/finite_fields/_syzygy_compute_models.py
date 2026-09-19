"""Producer-side contracts for finite-field Jacobian syzygy leaves (#964).

The checker (``finite_field.jacobian_syzygy.check``) derives the
characteristic-p Jacobian itself and verifies supplied rows.  These values
expose the producer side with the same reduction regime and envelope:
formal characteristic-p gradients, quotient normal forms with replayable
reduction ledgers, and bounded-degree syzygy-slice bases whose every row
replays to a zero remainder.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.finite_fields._algebraic_sets import AlgebraicPolynomial
from jacobian.math.finite_fields._jacobian_syzygy_models import (
    MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS,
    MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS,
)

MAX_SYZYGY_SLICE_DEGREE = 8
MAX_SYZYGY_UNKNOWNS = 2048
MAX_SYZYGY_EQUATIONS = 4096
MAX_SYZYGY_RREF_UPDATES = 20_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_field.{reason}", message)


class FiniteFieldJacobianRequest(StrictModel):
    """Derive the formal characteristic-p gradient of one polynomial."""

    polynomial: AlgebraicPolynomial


class FiniteFieldJacobianResult(StrictModel):
    """The source polynomial with its formal partial derivatives.

    ``partials[i]`` is ``d(polynomial)/d(variable[i])`` over GF(p):
    exponents divisible by the characteristic vanish.  Every partial
    shares the source presentation and variable axis.
    """

    polynomial: AlgebraicPolynomial
    partials: tuple[AlgebraicPolynomial, ...]
    characteristic: int = Field(gt=0)

    @model_validator(mode="after")
    def require_jacobian_shape(self) -> Self:
        if len(self.partials) != len(self.polynomial.variable_axis.labels):
            raise _validation_error(
                "jacobian_partial_count",
                "the gradient carries one partial per declared variable",
            )
        for partial in self.partials:
            if (
                partial.presentation != self.polynomial.presentation
                or partial.variable_axis != self.polynomial.variable_axis
            ):
                raise _validation_error(
                    "jacobian_shared_parent",
                    "every partial must share the source presentation and axis",
                )
        if self.characteristic != self.polynomial.presentation.characteristic:
            raise _validation_error(
                "jacobian_characteristic_binding",
                "the result binds the source presentation characteristic",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class ReduceStep(StrictModel):
    """One recorded quotient-reduction step.

    ``target`` is the rewritten monomial and ``coefficient`` its exact
    coefficient in the working polynomial before the step.  The produced
    image is rederived from the retained principal generator during
    replay, so steps record choices, not arithmetic.
    """

    target: tuple[int, ...]
    coefficient: int = Field(gt=0)

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class QuotientReduceRequest(StrictModel):
    """Reduce one polynomial to normal form modulo a principal ideal."""

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None


class QuotientReduceResult(StrictModel):
    """The source polynomial with its normal form and reduction ledger.

    An empty ideal means the plain polynomial ring (the remainder is the
    source and no step ran).  Otherwise the single generator must be monic
    in ``reduction_variable`` and the remainder is the unique normal form.
    Replaying the steps against the retained generator reproduces the
    remainder; equal remainders decide quotient equality.
    """

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None
    remainder: AlgebraicPolynomial
    steps: tuple[ReduceStep, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS
    )
    characteristic: int = Field(gt=0)

    @model_validator(mode="after")
    def require_reduce_shape(self) -> Self:
        presentation = self.polynomial.presentation
        axis = self.polynomial.variable_axis
        for member in (*self.ideal_generators, self.remainder):
            if member.presentation != presentation or member.variable_axis != axis:
                raise _validation_error(
                    "quotient_reduce_shared_parent",
                    "every polynomial must share the source presentation and axis",
                )
        if self.characteristic != presentation.characteristic:
            raise _validation_error(
                "quotient_reduce_characteristic_binding",
                "the result binds the source presentation characteristic",
            )
        if not self.ideal_generators and (
            self.reduction_variable is not None or self.steps
        ):
            raise _validation_error(
                "quotient_reduce_empty_regime",
                "an empty ideal carries no reduction variable or steps",
            )
        if self.ideal_generators and self.reduction_variable is None:
            raise _validation_error(
                "quotient_reduce_regime",
                "a nonempty principal ideal requires a reduction variable",
            )
        nvars = len(axis.labels)
        for step in self.steps:
            if len(step.target) != nvars or any(
                type(exponent) is not int or exponent < 0 for exponent in step.target
            ):
                raise _validation_error(
                    "quotient_reduce_step_shape",
                    "every step targets a nonnegative exponent vector on the axis",
                )
            if not 1 <= step.coefficient < presentation.characteristic:
                raise _validation_error(
                    "quotient_reduce_step_coefficient",
                    "every step coefficient is a nonzero field residue",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SyzygyGeneratorsRequest(StrictModel):
    """Compute the bounded-degree syzygy slice of a characteristic-p Jacobian."""

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None
    max_syzygy_degree: int = Field(
        default=4,
        ge=0,
        le=MAX_SYZYGY_SLICE_DEGREE,
        description=(
            "Return a basis of syzygy rows whose every component has total "
            "degree at most this bound."
        ),
    )


class SyzygyGeneratorsResult(StrictModel):
    """An exact basis of the bounded-degree Jacobian syzygy slice.

    ``rows`` is a basis of the GF(p)-space of rows ``r`` with every
    component of total degree at most ``max_syzygy_degree`` satisfying
    ``normal_form(sum_i r_i * jacobian_i) == 0``.  The dimension theorem
    binds the payload: ``len(rows) == unknowns - rank``.  An empty row
    family is a valid exact claim (the slice is the zero space), never a
    claim about higher degrees or the full syzygy module.
    """

    polynomial: AlgebraicPolynomial
    ideal_generators: tuple[AlgebraicPolynomial, ...] = Field(
        default=(), max_length=MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS
    )
    reduction_variable: str | None = None
    max_syzygy_degree: int = Field(ge=0, le=MAX_SYZYGY_SLICE_DEGREE)
    jacobian: tuple[AlgebraicPolynomial, ...]
    rows: tuple[tuple[AlgebraicPolynomial, ...], ...] = ()
    unknowns: int = Field(ge=0)
    equations: int = Field(ge=0)
    rank: int = Field(ge=0)
    characteristic: int = Field(gt=0)

    @model_validator(mode="after")
    def require_generators_payload(self) -> Self:
        presentation = self.polynomial.presentation
        axis = self.polynomial.variable_axis
        nvars = len(axis.labels)
        if len(self.jacobian) != nvars:
            raise _validation_error(
                "syzygy_generators_jacobian_count",
                "the Jacobian carries one partial per declared variable",
            )
        for member in self.jacobian:
            if member.presentation != presentation or member.variable_axis != axis:
                raise _validation_error(
                    "syzygy_generators_shared_parent",
                    "every polynomial must share the source presentation and axis",
                )
        for row in self.rows:
            if len(row) != nvars:
                raise _validation_error(
                    "syzygy_generators_row_length",
                    "every syzygy row has one entry per declared variable",
                )
            for entry in row:
                if entry.presentation != presentation or entry.variable_axis != axis:
                    raise _validation_error(
                        "syzygy_generators_shared_parent",
                        "every polynomial must share the source presentation and axis",
                    )
                for term in entry.terms:
                    if term.coefficient.is_zero:
                        continue
                    if sum(term.exponents) > self.max_syzygy_degree:
                        raise _validation_error(
                            "syzygy_generators_slice_degree",
                            "every row component stays within the slice degree",
                        )
        if self.characteristic != presentation.characteristic:
            raise _validation_error(
                "syzygy_generators_characteristic_binding",
                "the result binds the source presentation characteristic",
            )
        return self

    @model_validator(mode="after")
    def require_generators_dimension(self) -> Self:
        if self.rank > min(self.unknowns, self.equations):
            raise _validation_error(
                "syzygy_generators_rank_bound",
                "the rank fits inside the assembled linear system",
            )
        if len(self.rows) != self.unknowns - self.rank:
            raise _validation_error(
                "syzygy_generators_nullity",
                "the row family spans the kernel: rows == unknowns - rank",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_SYZYGY_EQUATIONS",
    "MAX_SYZYGY_RREF_UPDATES",
    "MAX_SYZYGY_SLICE_DEGREE",
    "MAX_SYZYGY_UNKNOWNS",
    "FiniteFieldJacobianRequest",
    "FiniteFieldJacobianResult",
    "QuotientReduceRequest",
    "QuotientReduceResult",
    "ReduceStep",
    "SyzygyGeneratorsRequest",
    "SyzygyGeneratorsResult",
]
