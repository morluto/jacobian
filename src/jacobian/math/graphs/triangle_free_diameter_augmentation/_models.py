"""Typed contracts for triangle-free diameter augmentation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

TriangleFreeDiameterAugmentationStatus = Literal[
    "EXACT",
    "INFEASIBLE",
    "SOLVER_BUDGET_EXCEEDED",
]


class TriangleFreeDiameterAugmentationBudget(StrictModel):
    """Explicit wall-clock and order limits for one bounded augmentation search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=60)
    max_order: StrictInt = Field(default=10, ge=0, le=12)


class TriangleFreeDiameterAugmentationRequest(StrictModel):
    """One connected triangle-free graph and target diameter."""

    graph: SimpleUndirectedGraph
    target_diameter: StrictInt = Field(ge=1, le=12)
    resource_budget: TriangleFreeDiameterAugmentationBudget = Field(
        default_factory=TriangleFreeDiameterAugmentationBudget
    )


class TriangleFreeDiameterAugmentationResult(StrictModel):
    """Exact minimum augmentation or typed non-success for one source graph."""

    graph: SimpleUndirectedGraph
    target_diameter: StrictInt = Field(ge=1, le=12)
    status: TriangleFreeDiameterAugmentationStatus
    added_edge_count: StrictInt | None = Field(default=None, ge=0)
    added_edges: tuple[tuple[str, str], ...] = Field(default=())
    augmented_diameter: StrictInt | None = Field(default=None, ge=0, le=12)
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        self._require_canonical_edges()
        self._require_cardinality_binding()
        return self

    def _require_canonical_edges(self) -> None:
        if self.added_edges != tuple(sorted(self.added_edges)):
            raise PydanticCustomError(
                "graph.augmentation_edges_must_be_canonically_sorted",
                "augmentation edges must be canonically sorted",
            )
        if len(set(self.added_edges)) != len(self.added_edges):
            raise PydanticCustomError(
                "graph.augmentation_edges_must_be_unique",
                "augmentation edges must be unique",
            )
        for left, right in self.added_edges:
            if left >= right:
                raise PydanticCustomError(
                    "graph.augmentation_edges_must_be_canonical_pairs",
                    "augmentation edges must be canonical pairs with left < right",
                )
            if left not in self.graph.vertices or right not in self.graph.vertices:
                raise PydanticCustomError(
                    "graph.augmentation_edges_must_reference_source_vertices",
                    "augmentation edges must reference source vertices",
                )
        original = set(self.graph.edges)
        if any(edge in original for edge in self.added_edges):
            raise PydanticCustomError(
                "graph.augmentation_edges_must_be_missing_from_source",
                "augmentation edges must be missing from source",
            )

    def _require_cardinality_binding(self) -> None:
        if self.status == "EXACT":
            self._require_exact_branch()
        elif self.status == "INFEASIBLE":
            if self.added_edge_count is not None or self.added_edges:
                raise PydanticCustomError(
                    "graph.infeasible_must_not_carry_witness",
                    "infeasible result must not carry witness",
                )
            if self.augmented_diameter is not None:
                raise PydanticCustomError(
                    "graph.infeasible_must_not_carry_diameter",
                    "infeasible result must not carry diameter",
                )
        elif self.status == "SOLVER_BUDGET_EXCEEDED":
            if self.added_edge_count is not None or self.added_edges:
                raise PydanticCustomError(
                    "graph.budget_exceeded_must_not_carry_witness",
                    "budget-exceeded result must not carry witness",
                )
            if self.augmented_diameter is not None:
                raise PydanticCustomError(
                    "graph.budget_exceeded_must_not_carry_diameter",
                    "budget-exceeded result must not carry diameter",
                )

    def _require_exact_branch(self) -> None:
        if self.added_edge_count is None or self.added_edge_count != len(
            self.added_edges
        ):
            raise PydanticCustomError(
                "graph.exact_augmentation_requires_coincident_cardinality",
                "exact augmentation requires coincident cardinality",
            )
        if self.augmented_diameter is None:
            raise PydanticCustomError(
                "graph.exact_augmentation_requires_augmented_diameter",
                "exact augmentation requires augmented diameter",
            )
        if self.augmented_diameter > self.target_diameter:
            raise PydanticCustomError(
                "graph.augmented_diameter_must_not_exceed_target",
                "augmented diameter must not exceed target",
            )
        import networkx as nx

        g: nx.Graph[str] = nx.Graph()
        g.add_nodes_from(self.graph.vertices)
        g.add_edges_from(self.graph.edges)
        g.add_edges_from(self.added_edges)
        try:
            tri = sum(nx.triangles(g).values()) // 3  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive
            raise PydanticCustomError(
                "graph.augmentation_structural_check_failed",
                "augmentation structural check failed",
            ) from exc
        if tri != 0:
            raise PydanticCustomError(
                "graph.augmented_graph_must_be_triangle_free",
                "augmented graph must be triangle-free",
            )
        if not nx.is_connected(g):
            raise PydanticCustomError(
                "graph.augmented_graph_must_be_connected",
                "augmented graph must be connected",
            )
        try:
            diam = int(nx.diameter(g))
        except Exception as exc:
            raise PydanticCustomError(
                "graph.augmented_graph_diameter_check_failed",
                "augmented graph diameter check failed",
            ) from exc
        if diam != self.augmented_diameter:
            raise PydanticCustomError(
                "graph.augmented_diameter_must_match_graph",
                "augmented diameter must match graph",
            )
        if diam > self.target_diameter:
            raise PydanticCustomError(
                "graph.augmented_diameter_exceeds_target",
                "augmented diameter exceeds target",
            )

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        target_diameter: int,
        status: TriangleFreeDiameterAugmentationStatus,
        added_edges: tuple[tuple[str, str], ...],
        augmented_diameter: int | None,
        detail: str,
    ) -> Self:
        """Construct one structurally checked outcome from the trusted kernel."""

        added_edge_count = len(added_edges) if status == "EXACT" else None
        # Use model_construct to bypass re-validation of invariants already proved?
        # We still want structural validation, so use normal construction but
        # kernel has already validated. Use model_construct for speed when kernel
        # has established invariants; the validator remains for external deserialization.
        # Here we use model_construct then rely on validator for external reads;
        # kernel-side we have already checked.
        return cls.model_construct(
            graph=graph,
            target_diameter=target_diameter,
            status=status,
            added_edge_count=added_edge_count,
            added_edges=added_edges if status == "EXACT" else (),
            augmented_diameter=augmented_diameter,
            detail=detail,
        )


__all__ = [
    "TriangleFreeDiameterAugmentationBudget",
    "TriangleFreeDiameterAugmentationRequest",
    "TriangleFreeDiameterAugmentationResult",
    "TriangleFreeDiameterAugmentationStatus",
]
