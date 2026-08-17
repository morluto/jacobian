"""Provider-independent values for bounded independence-number search."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

IndependenceSearchStatus = Literal["EXACT", "UNKNOWN"]
IndependenceTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "WALL_TIME",
    "SOLVER_UNKNOWN",
    "SOLVER_UNSAT",
    "SPECIAL_CASE",
]


class IndependenceNumberBudget(StrictModel):
    """Explicit public limits for one bounded independence-number search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(
        default=1,
        ge=1,
        le=33,
        description=(
            "Compatibility budget retained from the threshold-search contract; "
            "the version-2 optimizer uses one solver call."
        ),
    )
    max_order: StrictInt = Field(default=128, ge=0, le=128)


class IndependenceNumberRequest(StrictModel):
    """One finite simple graph and its operation-owned search budget."""

    graph: SimpleUndirectedGraph
    resource_budget: IndependenceNumberBudget = Field(
        default_factory=IndependenceNumberBudget
    )

    @model_validator(mode="after")
    def require_supported_order(self) -> Self:
        order = len(self.graph.vertices)
        if order > self.resource_budget.max_order:
            raise ValueError("graph order exceeds the declared max_order budget")
        if order > 128:
            raise ValueError("independence-number search supports order at most 128")
        return self


class IndependenceNumberResult(StrictModel):
    """Exact optimum or bounded incumbent and bounds for one supplied graph."""

    result_schema_version: Literal["1"] = "1"
    status: IndependenceSearchStatus
    order: StrictInt = Field(ge=0, le=128)
    optimum_value: StrictInt | None = Field(default=None, ge=0, le=128)
    incumbent_value: StrictInt = Field(ge=0, le=128)
    lower_bound: StrictInt = Field(ge=0, le=128)
    upper_bound: StrictInt = Field(ge=0, le=128)
    witness_vertices: tuple[str, ...] = Field(max_length=128)
    termination_reason: IndependenceTermination
    detail: str = Field(min_length=1, max_length=1024)
    convention: Literal["MAXIMUM_EDGE_FREE_VERTEX_SUBSET"] = (
        "MAXIMUM_EDGE_FREE_VERTEX_SUBSET"
    )

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        if self.witness_vertices != tuple(sorted(self.witness_vertices)) or len(
            set(self.witness_vertices)
        ) != len(self.witness_vertices):
            raise ValueError("witness vertices must be unique and canonically sorted")
        if self.incumbent_value != len(self.witness_vertices):
            raise ValueError("witness cardinality must match the incumbent")
        if self.lower_bound != self.incumbent_value:
            raise ValueError("a maximum-search incumbent is the lower bound")
        if not self.lower_bound <= self.upper_bound <= self.order:
            raise ValueError("independence-number bounds must lie inside graph order")
        if self.status == "EXACT":
            if (
                self.optimum_value is None
                or self.optimum_value != self.incumbent_value
                or self.optimum_value != self.upper_bound
                or self.termination_reason
                not in {"OPTIMUM_ESTABLISHED", "SPECIAL_CASE"}
            ):
                raise ValueError("exact result must bind one coincident optimum")
        elif self.optimum_value is not None:
            raise ValueError("incomplete search cannot claim an optimum")
        return self


def independence_number(request: IndependenceNumberRequest) -> IndependenceNumberResult:
    """Return an exact optimum when bounded Z3 optimization establishes it."""

    from jacobian.math.graphs import _independence_z3

    return _independence_z3.solve_independence_number(request)


__all__ = [
    "IndependenceNumberBudget",
    "IndependenceNumberRequest",
    "IndependenceNumberResult",
    "IndependenceSearchStatus",
    "IndependenceTermination",
    "independence_number",
]
