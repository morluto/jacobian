"""Per-set discrepancy bounds and closed finite-decision outcomes."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem

AbsoluteBound = Annotated[int, Field(ge=0, le=64, strict=True)]
SignedSum = Annotated[int, Field(ge=-64, le=64, strict=True)]
ColorSign = Annotated[
    int, Field(ge=-1, le=1, strict=True, json_schema_extra={"enum": [-1, 1]})
]


class BoundedColoringBudget(StrictModel):
    """One exact decision call's bounded proof work and total elapsed time."""

    solver_work_limit: int = Field(default=1_000_000, ge=1, le=10_000_000)
    wall_seconds: int = Field(default=15, ge=1, le=60)


class BoundedColoringRequest(StrictModel):
    set_system: FiniteSetSystem
    absolute_bounds: tuple[AbsoluteBound, ...] = Field(max_length=1000)
    resource_budget: BoundedColoringBudget = Field(
        default_factory=BoundedColoringBudget
    )

    @model_validator(mode="after")
    def require_bound_axis(self) -> Self:
        _require_bounds(self.set_system, self.absolute_bounds)
        return self


def _require_bounds(source: FiniteSetSystem, bounds: tuple[int, ...]) -> None:
    if len(bounds) != len(source.sets):
        raise ValueError("one absolute bound is required per indexed source set")
    if any(
        type(b) is not int or not 0 <= b <= len(s)
        for s, b in zip(source.sets, bounds, strict=True)
    ):
        raise ValueError(
            "absolute bounds must be integers between zero and their set size"
        )


class SatisfiableBoundedColoring(StrictModel):
    status: Literal["SATISFIABLE"] = "SATISFIABLE"
    coloring: tuple[ColorSign, ...] = Field(max_length=64)
    signed_sums: tuple[SignedSum, ...] = Field(max_length=1000)

    @model_validator(mode="after")
    def require_signed_colors(self) -> Self:
        if any(value == 0 for value in self.coloring):
            raise ValueError("coloring values must be -1 or +1")
        return self


class UnsatisfiableBoundedColoring(StrictModel):
    status: Literal["UNSATISFIABLE"] = "UNSATISFIABLE"


class BoundedColoringBudgetExceeded(StrictModel):
    """No mathematical conclusion or partial witness was established."""

    status: Literal["BUDGET_EXCEEDED"] = "BUDGET_EXCEEDED"


class BoundedColoringExecutionFailed(StrictModel):
    """The exact backend did not establish a decision; no witness is returned."""

    status: Literal["EXECUTION_FAILED"] = "EXECUTION_FAILED"


BoundedColoringOutcome = Annotated[
    SatisfiableBoundedColoring
    | UnsatisfiableBoundedColoring
    | BoundedColoringBudgetExceeded
    | BoundedColoringExecutionFailed,
    Field(discriminator="status"),
]


class BoundedColoringResult(StrictModel):
    """Source-bound per-set feasibility, with a complete ledger only on SAT.

    Parsing checks domains and axes. The producer establishes each signed sum
    and bound relation; the exact solver alone establishes unsatisfiability.
    """

    set_system: FiniteSetSystem
    absolute_bounds: tuple[AbsoluteBound, ...] = Field(max_length=1000)
    outcome: BoundedColoringOutcome

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_bounds(self.set_system, self.absolute_bounds)
        if isinstance(self.outcome, SatisfiableBoundedColoring):
            if len(self.outcome.coloring) != self.set_system.ground_set_size:
                raise ValueError("coloring must cover the complete ground set")
            if len(self.outcome.signed_sums) != len(self.set_system.sets):
                raise ValueError(
                    "signed-sum ledger must cover every indexed source set"
                )
        return self
