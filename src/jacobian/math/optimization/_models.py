"""Private wire models for bounded rational linear optimization."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, NamedTuple, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 128
type RationalLinearProgramStatus = Literal[
    "OPTIMAL",
    "PRIMAL_FEASIBLE",
    "INFEASIBLE",
    "UNBOUNDED",
    "UNKNOWN",
]


class StandardFormRationalLinearProgram(StrictModel):
    """Minimize ``cᵀx`` subject to ``Ax=b`` and ``x>=0``."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=32)
    objective: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)
    coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=64,
    )
    rhs: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("linear-program variable names must be unique")
        if any(
            not name
            or len(name) > 64
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (char.isalnum() or char == "_") for char in name)
            for name in self.variables
        ):
            raise ValueError("linear-program variable names must be identifiers")
        width = len(self.variables)
        if len(self.objective) != width:
            raise ValueError("objective length must equal the variable count")
        if len(self.coefficients) != len(self.rhs):
            raise ValueError("coefficient row count must equal the rhs length")
        if any(len(row) != width for row in self.coefficients):
            raise ValueError("every coefficient row must match the variable count")
        for value in (
            *self.objective,
            *self.rhs,
            *(item for row in self.coefficients for item in row),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="validated-analysis rational",
            )
        return self


class RationalLinearProgramRequest(StrictModel):
    program: StandardFormRationalLinearProgram


class _PrimalReplay(NamedTuple):
    """Exact primal diagnostics recomputed from one source program."""

    objective: CanonicalRational
    residuals: tuple[CanonicalRational, ...]
    feasible: bool


class _DualReplay(NamedTuple):
    """Exact dual diagnostics recomputed from one source program."""

    objective: CanonicalRational
    slacks: tuple[CanonicalRational, ...]
    feasible: bool


def _source_rows(
    program: StandardFormRationalLinearProgram,
) -> tuple[list[Fraction], list[list[Fraction]], list[Fraction]]:
    """Return the exact objective ``c``, coefficient rows ``A``, and rhs ``b``."""

    objective = [value.as_fraction() for value in program.objective]
    coefficients = [
        [value.as_fraction() for value in row] for row in program.coefficients
    ]
    rhs = [value.as_fraction() for value in program.rhs]
    return objective, coefficients, rhs


def _primal_replay(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[CanonicalRational, ...],
) -> _PrimalReplay:
    """Recompute exact primal diagnostics of ``candidate`` against ``program``.

    This is the one domain-owned replay definition shared by the producing
    operation and result validation, so producer-derived diagnostics and
    independently submitted diagnostics can never drift apart. Raises
    ``ValueError`` when the candidate dimension disagrees with the source.
    """

    width = len(program.variables)
    if len(candidate) != width:
        raise ValueError("primal candidate length must equal the source variable count")
    x = [value.as_fraction() for value in candidate]
    objective, coefficients, rhs = _source_rows(program)
    residual_fractions = [
        sum((coefficients[i][j] * x[j] for j in range(width)), Fraction(0)) - rhs[i]
        for i in range(len(rhs))
    ]
    value = sum((objective[j] * x[j] for j in range(width)), Fraction(0))
    return _PrimalReplay(
        objective=CanonicalRational.from_fraction(value),
        residuals=tuple(
            CanonicalRational.from_fraction(residual) for residual in residual_fractions
        ),
        feasible=all(entry >= 0 for entry in x)
        and all(residual == 0 for residual in residual_fractions),
    )


def _dual_replay(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[CanonicalRational, ...],
) -> _DualReplay:
    """Recompute exact dual diagnostics of ``candidate`` against ``program``.

    The minimization dual convention is ``A^T y <= c`` over free ``y``
    with dual objective ``b^T y``.
    """

    rows = len(program.rhs)
    width = len(program.variables)
    if len(candidate) != rows:
        raise ValueError("dual candidate length must equal the source constraint count")
    y = [value.as_fraction() for value in candidate]
    objective, coefficients, rhs = _source_rows(program)
    slack_fractions = [
        objective[j]
        - sum((coefficients[i][j] * y[i] for i in range(rows)), Fraction(0))
        for j in range(width)
    ]
    value = sum((rhs[i] * y[i] for i in range(rows)), Fraction(0))
    return _DualReplay(
        objective=CanonicalRational.from_fraction(value),
        slacks=tuple(
            CanonicalRational.from_fraction(slack) for slack in slack_fractions
        ),
        feasible=all(slack >= 0 for slack in slack_fractions),
    )


def _require_infeasibility_witness(
    program: StandardFormRationalLinearProgram,
    witness: tuple[CanonicalRational, ...],
) -> None:
    """Replay the Farkas certificate ``A^T y >= 0`` and ``b^T y < 0`` exactly.

    This sign convention is part of the public contract: such a ``y``
    certifies that ``{x : Ax=b, x>=0}`` is empty.
    """

    rows = len(program.rhs)
    if len(witness) != rows:
        raise ValueError("farkas witness length must equal the source constraint count")
    y = [value.as_fraction() for value in witness]
    _, coefficients, rhs = _source_rows(program)
    pairings = [
        sum((coefficients[i][j] * y[i] for i in range(rows)), Fraction(0))
        for j in range(len(program.variables))
    ]
    if any(pairing < 0 for pairing in pairings):
        raise ValueError('farkas witness violates the convention "A^T y >= 0"')
    b_pairing = sum((rhs[i] * y[i] for i in range(rows)), Fraction(0))
    if not b_pairing < 0:
        raise ValueError('farkas witness violates the convention "b^T y < 0"')


def _require_unboundedness_witnesses(
    program: StandardFormRationalLinearProgram,
    point: tuple[CanonicalRational, ...],
    direction: tuple[CanonicalRational, ...],
) -> None:
    """Replay the unboundedness evidence ``x0`` and ``d`` against ``program``.

    Requires ``x0 >= 0`` with ``Ax0=b``, and ``d >= 0`` with ``Ad=0`` and
    ``c^T d < 0``, so ``x0 + t*d`` stays feasible while the minimization
    objective tends to ``-infinity``.
    """

    if len(direction) != len(program.variables):
        raise ValueError(
            "recession direction length must equal the source variable count"
        )
    if not _primal_replay(program, point).feasible:
        raise ValueError(
            "feasible point is not primal feasible for the retained source"
        )
    d = [value.as_fraction() for value in direction]
    _, coefficients, _ = _source_rows(program)
    objective = [value.as_fraction() for value in program.objective]
    width = len(program.variables)
    if any(entry < 0 for entry in d):
        raise ValueError("recession direction must be nonnegative")
    images = [
        sum((coefficients[i][j] * d[j] for j in range(width)), Fraction(0))
        for i in range(len(program.rhs))
    ]
    if any(image != 0 for image in images):
        raise ValueError("recession direction must satisfy A d = 0")
    slope = sum((objective[j] * d[j] for j in range(width)), Fraction(0))
    if not slope < 0:
        raise ValueError("recession direction must strictly improve the objective")


class RationalLinearProgramResult(StrictModel):
    """The direct mathematical outcome of one rational linear program.

    Every outcome retains its complete source program, and every claimed
    fact is replayed exactly against that source: primal feasibility
    (``x>=0``, ``Ax=b``), dual feasibility (``A^T y <= c``), both objective
    values, and strong duality (``c^T x = b^T y``). Residuals, slacks, and
    objectives are derived diagnostics that must equal their recomputation.
    ``INFEASIBLE`` carries a Farkas witness under the documented sign
    convention and ``UNBOUNDED`` carries a feasible point plus an improving
    recession direction; ``UNKNOWN`` is an explicitly non-mathematical
    incomplete execution outcome carrying no claim at all.
    """

    status: RationalLinearProgramStatus
    program: StandardFormRationalLinearProgram
    primal_candidate: tuple[CanonicalRational, ...] | None = None
    dual_candidate: tuple[CanonicalRational, ...] | None = None
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = None
    dual_slacks: tuple[CanonicalRational, ...] | None = None
    farkas_witness: tuple[CanonicalRational, ...] | None = None
    feasible_point: tuple[CanonicalRational, ...] | None = None
    recession_direction: tuple[CanonicalRational, ...] | None = None

    def _primal_fields(self) -> tuple[object, ...]:
        return (
            self.primal_candidate,
            self.primal_objective,
            self.primal_residuals,
        )

    def _dual_fields(self) -> tuple[object, ...]:
        return (self.dual_candidate, self.dual_objective, self.dual_slacks)

    def _negative_certificates(self) -> tuple[object, ...]:
        return (self.farkas_witness, self.feasible_point, self.recession_direction)

    def _require_unknown(self) -> None:
        if any(
            value is not None
            for value in (
                *self._primal_fields(),
                *self._dual_fields(),
                *self._negative_certificates(),
            )
        ):
            raise ValueError("an unknown outcome carries no mathematical claim")

    def _require_infeasible(self) -> None:
        if self.farkas_witness is None:
            raise ValueError("an infeasible result requires a Farkas witness")
        if any(
            value is not None
            for value in (
                *self._primal_fields(),
                *self._dual_fields(),
                *self._negative_certificates()[1:],
            )
        ):
            raise ValueError("an infeasible result cannot carry a point or dual data")
        _require_infeasibility_witness(self.program, self.farkas_witness)

    def _require_unbounded(self) -> None:
        if self.feasible_point is None or self.recession_direction is None:
            raise ValueError(
                "an unbounded result requires a feasible point"
                " and a recession direction"
            )
        if any(
            value is not None
            for value in (
                *self._primal_fields(),
                *self._dual_fields(),
                self.farkas_witness,
            )
        ):
            raise ValueError("an unbounded result cannot carry optimal or Farkas data")
        _require_unboundedness_witnesses(
            self.program,
            self.feasible_point,
            self.recession_direction,
        )

    def _require_primal(self) -> bool:
        optimal = self.status == "OPTIMAL"
        has_primal = optimal or self.status == "PRIMAL_FEASIBLE"
        primal_fields = self._primal_fields()
        dual_fields = self._dual_fields()
        if has_primal and not all(value is not None for value in primal_fields):
            raise ValueError(
                "a primal result requires a candidate, objective, and residuals"
            )
        if not has_primal and any(value is not None for value in primal_fields):
            raise ValueError("an infeasible or unbounded result cannot carry a point")
        if optimal and not all(value is not None for value in dual_fields):
            raise ValueError(
                "an optimal result requires a dual candidate, objective, and slacks"
            )
        if not optimal and any(value is not None for value in dual_fields):
            raise ValueError("only an optimal result can carry dual data")
        if any(value is not None for value in self._negative_certificates()):
            raise ValueError(
                "only an infeasible or unbounded result can carry certificates"
            )
        return optimal

    @model_validator(mode="after")
    def bind_outcome_to_source(self) -> Self:
        if self.status == "UNKNOWN":
            self._require_unknown()
            return self
        if self.status == "INFEASIBLE":
            self._require_infeasible()
            return self
        if self.status == "UNBOUNDED":
            self._require_unbounded()
            return self

        assert self.primal_candidate is not None
        assert self.primal_objective is not None
        assert self.primal_residuals is not None
        primal = _primal_replay(self.program, self.primal_candidate)
        if self.primal_residuals != primal.residuals:
            raise ValueError(
                "submitted primal residuals do not equal the recomputed residuals"
            )
        if self.primal_objective != primal.objective:
            raise ValueError(
                "submitted primal objective does not equal the recomputed objective"
            )
        if not primal.feasible:
            raise ValueError(
                "primal candidate is not primal feasible for the retained source"
            )
        if not self._require_primal():
            return self

        assert self.dual_candidate is not None
        assert self.dual_objective is not None
        assert self.dual_slacks is not None
        dual = _dual_replay(self.program, self.dual_candidate)
        if self.dual_slacks != dual.slacks:
            raise ValueError("submitted dual slacks do not equal the recomputed slacks")
        if self.dual_objective != dual.objective:
            raise ValueError(
                "submitted dual objective does not equal the recomputed objective"
            )
        if not dual.feasible:
            raise ValueError(
                "dual candidate is not dual feasible for the retained source"
            )
        if self.primal_objective != self.dual_objective:
            raise ValueError(
                "strong duality fails under the retained source: c^T x != b^T y"
            )
        return self


__all__ = [
    "RationalLinearProgramRequest",
    "RationalLinearProgramResult",
    "RationalLinearProgramStatus",
    "StandardFormRationalLinearProgram",
]
