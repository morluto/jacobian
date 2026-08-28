"""Provider-independent values for bounded independence-number search."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import (
    SimpleUndirectedGraph,
    simple_undirected_graph_wire_bytes,
)

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
    max_order: StrictInt = Field(default=128, ge=0, le=128)


# The result retains its source graph and echoes every witness identifier,
# so a request near the canonical input limit can serialize a response past
# the identical output limit.  Admission reserves this much for the fixed
# scalar fields, the bounded detail string, and the result envelope beyond
# the echoed graph and worst-case witness labels.
_RESULT_ENVELOPE_RESERVE_BYTES = 2_048


def _label_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    return sum(len(encode_strict_json(label) + b",") for label in graph.vertices)


def _require_output_headroom(source_bytes: int, witness_label_bytes: int) -> None:
    estimated_result_bytes = (
        source_bytes + witness_label_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.independence_number.output_budget",
            message=(
                "the independence-number result retains its source graph and "
                "witness labels and would exceed the "
                f"{output_limit}-byte canonical output limit; "
                "shorten vertex labels or shrink the graph"
            ),
        )


class IndependenceNumberRequest(StrictModel):
    """One finite simple graph and its operation-owned search budget."""

    graph: SimpleUndirectedGraph
    resource_budget: IndependenceNumberBudget = Field(
        default_factory=IndependenceNumberBudget
    )


def _require_supported_order(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
) -> None:
    """Admit the mathematical graph order for one bounded kernel call."""

    order = len(graph.vertices)
    if order > resource_budget.max_order:
        raise OperationDomainValidationError(
            location=("resource_budget", "max_order"),
            code="graph.independence_number.max_order_budget",
            message="graph order exceeds the declared max_order budget",
        )
    if order > 128:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.independence_number.order_bound",
            message="independence-number search supports order at most 128",
        )


def _require_admitted_request(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
) -> None:
    _require_supported_order(graph, resource_budget)
    _require_output_headroom(
        simple_undirected_graph_wire_bytes(graph), _label_wire_bytes(graph)
    )


class IndependenceNumberResult(StrictModel):
    """Exact optimum or bounded incumbent and bounds for one supplied graph.

    Retains the canonical source graph and checks structural source binding:
    every witness identifier belongs to the source, no source edge has both
    endpoints in the witness, the incumbent equals the witness cardinality,
    and the reported order matches the source.  Exact optimality is a
    semantic claim, not structural JSON validation; callers accepting a
    separately supplied ``EXACT`` result use the owner-local bounded verifier.
    Operational ``UNKNOWN`` stays distinct from a mathematical optimum. An
    incomplete outcome reports the graph order as its independently safe
    upper bound, so no unauthenticated incumbent gap survives validation.
    """

    graph: SimpleUndirectedGraph
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        status: IndependenceSearchStatus,
        optimum_value: int | None,
        incumbent_vertices: tuple[str, ...],
        upper_bound: int,
        termination_reason: IndependenceTermination,
        detail: str,
    ) -> Self:
        """Construct one structurally checked outcome from the trusted kernel."""

        incumbent_value = len(incumbent_vertices)
        return cls(
            graph=graph,
            status=status,
            order=len(graph.vertices),
            optimum_value=optimum_value,
            incumbent_value=incumbent_value,
            lower_bound=incumbent_value,
            upper_bound=upper_bound,
            witness_vertices=incumbent_vertices,
            termination_reason=termination_reason,
            detail=detail,
        )

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        if self.order != len(self.graph.vertices):
            raise ValueError("reported order must match the retained source graph")
        vertices = set(self.graph.vertices)
        if any(vertex not in vertices for vertex in self.witness_vertices):
            raise ValueError("every witness vertex must belong to the source graph")
        witness = set(self.witness_vertices)
        if any(
            left in witness and right in witness for left, right in self.graph.edges
        ):
            raise ValueError("witness must not contain both endpoints of a source edge")
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
        elif self.upper_bound != self.order:
            raise ValueError(
                "an incomplete result must report the graph order as its "
                "independently safe upper bound"
            )
        return self


def _compute_independence_number(
    request: IndependenceNumberRequest,
) -> IndependenceNumberResult:
    """Run the wire-request adapter retained for the catalog and MCP path."""

    return independence_number(request.graph, resource_budget=request.resource_budget)


def independence_number(
    graph: SimpleUndirectedGraph,
    *,
    resource_budget: IndependenceNumberBudget | None = None,
) -> IndependenceNumberResult:
    """Return the bounded independence-number outcome of ``graph``.

    Native callers supply the canonical graph value directly.  The public
    default execution envelope is the same order-128, five-second envelope
    advertised by the catalog; MCP-specific request parsing and transport
    headroom checks remain in the private wire adapter.
    """

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("independence_number expects a SimpleUndirectedGraph")
    resource_budget = resource_budget or IndependenceNumberBudget()
    _require_admitted_request(graph, resource_budget)

    from jacobian.math.graphs import _independence_z3

    return _independence_z3.solve_independence_number_values(graph, resource_budget)


__all__ = [
    "IndependenceNumberResult",
    "IndependenceSearchStatus",
    "IndependenceTermination",
    "independence_number",
]
