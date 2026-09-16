"""Typed wire contracts for algebraic topology operations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
    VertexLabel,
)

MAX_EDGES = 64
MAX_WORD = 128
# The abelianization is reduced through the shared exact integer Smith kernel,
# whose owner-envelope admits at most 64 rows and columns.
MAX_PRESENTATION_GENERATORS = 64
MAX_PRESENTATION_RELATORS = 64
# Each two-simplex contributes at most three edge words after tree collapse.
MAX_PRESENTATION_RELATOR_LETTERS = 3 * MAX_PRESENTATION_RELATORS


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(reason, message)


class OrientedEdge(StrictModel):
    edge_index: int = Field(ge=0)
    orientation: Literal[-1, 1]


class EdgePathWordRequest(StrictModel):
    """Compute the free group word for an edge path."""

    vertex_count: int = Field(ge=1)
    edges: tuple[tuple[int, int], ...] = Field(max_length=MAX_EDGES)
    start_vertex: int = Field(ge=0)
    path: tuple[OrientedEdge, ...] = Field(max_length=MAX_WORD)


class EdgePathConcatenateRequest(StrictModel):
    """Concatenate two edge paths."""

    vertex_count: int = Field(ge=1)
    path_a: tuple[int, ...] = Field(min_length=1, max_length=MAX_WORD)
    path_b: tuple[int, ...] = Field(min_length=1, max_length=MAX_WORD)


# Results


class EdgeGraph(StrictModel):
    """A bounded graph with an explicit vertex axis for edge paths."""

    vertex_count: int = Field(ge=1)
    edges: tuple[tuple[int, int], ...] = Field(max_length=MAX_EDGES)

    @classmethod
    def from_request(
        cls, vertex_count: int, edges: tuple[tuple[int, int], ...]
    ) -> EdgeGraph:
        return cls.model_construct(vertex_count=vertex_count, edges=edges)


class EdgePathWordResult(StrictModel):
    graph: EdgeGraph
    start_vertex: int
    path: tuple[OrientedEdge, ...] = Field(max_length=MAX_WORD)
    word: tuple[str, ...]
    length: int = Field(ge=0)


class EdgePathConcatenateResult(StrictModel):
    vertex_count: int = Field(ge=1)
    path_a: tuple[int, ...] = Field(min_length=1, max_length=MAX_WORD)
    path_b: tuple[int, ...] = Field(min_length=1, max_length=MAX_WORD)
    path: tuple[int, ...]
    length: int = Field(ge=0)


# Fundamental-group edge-path presentation


class WordLetter(StrictModel):
    """One signed occurrence of a presentation generator."""

    generator: int = Field(ge=0, le=MAX_PRESENTATION_GENERATORS - 1)
    exponent: Literal[-1, 1]


class FiniteGroupWord(StrictModel):
    """A freely reduced finite word over an ordered generator family.

    ``letters`` is the reduced word; the empty tuple is the identity word.
    """

    letters: tuple[WordLetter, ...] = Field(default=(), max_length=MAX_WORD)


class FiniteGroupPresentation(StrictModel):
    """Ordered generator IDs and their freely reduced relators."""

    generators: tuple[str, ...] = Field(
        default=(), max_length=MAX_PRESENTATION_GENERATORS
    )
    relators: tuple[FiniteGroupWord, ...] = Field(
        default=(), max_length=MAX_PRESENTATION_RELATORS
    )

    @model_validator(mode="after")
    def require_word_generator_domain(self) -> Self:
        if len(set(self.generators)) != len(self.generators):
            raise _validation_error(
                "fundamental_group.generator_ids",
                "presentation generator IDs must be unique",
            )
        for relator in self.relators:
            if any(
                letter.generator >= len(self.generators) for letter in relator.letters
            ):
                raise _validation_error(
                    "fundamental_group.relator_generator",
                    "every relator letter must name a declared generator",
                )
        return self


class EdgeWordEntry(StrictModel):
    """Translations of one canonical undirected edge into generator words.

    ``edge`` is the canonical ordered vertex pair with ``edge[0] < edge[1]``.
    ``forward`` traverses ``edge[0] -> edge[1]`` and ``backward`` traverses
    ``edge[1] -> edge[0]``; tree edges carry the identity word.
    """

    edge: tuple[VertexLabel, VertexLabel]
    is_tree_edge: bool
    forward: FiniteGroupWord
    backward: FiniteGroupWord

    @model_validator(mode="after")
    def require_canonical_orientation(self) -> Self:
        if not self.edge[0] < self.edge[1]:
            raise _validation_error(
                "fundamental_group.edge_order",
                "edge words must retain the canonical increasing orientation",
            )
        if (self.forward.letters == ()) != (self.backward.letters == ()):
            raise _validation_error(
                "fundamental_group.edge_tree_identity",
                "a tree edge must carry the identity word in both directions",
            )
        return self


class TriangleRelator(StrictModel):
    """One oriented two-simplex and its freely reduced boundary relator."""

    simplex: tuple[VertexLabel, VertexLabel, VertexLabel]
    word: FiniteGroupWord

    @model_validator(mode="after")
    def require_canonical_simplex(self) -> Self:
        if not (self.simplex[0] < self.simplex[1] < self.simplex[2]):
            raise _validation_error(
                "fundamental_group.triangle_order",
                "triangle relators must retain the canonical increasing simplex",
            )
        return self


class AbelianizationResult(StrictModel):
    """Exact cokernel data of the presentation's exponent-sum relations."""

    relation_matrix: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_PRESENTATION_GENERATORS)
    free_rank: int = Field(ge=0, le=MAX_PRESENTATION_GENERATORS)
    torsion_invariant_factors: tuple[int, ...] = Field(
        default=(), max_length=MAX_PRESENTATION_GENERATORS
    )

    @model_validator(mode="after")
    def require_cokernel_ledger(self) -> Self:
        if self.free_rank + self.rank != self.relation_matrix.column_count:
            raise _validation_error(
                "fundamental_group.abelianization_rank",
                "free rank plus Smith rank must equal the generator count",
            )
        if len(self.torsion_invariant_factors) > self.rank or any(
            factor <= 1 for factor in self.torsion_invariant_factors
        ):
            raise _validation_error(
                "fundamental_group.abelianization_factors",
                "torsion factors must be nontrivial Smith factors within the rank",
            )
        return self


class FundamentalGroupPresentationRequest(StrictModel):
    """Compute the edge-path presentation of one complex near a base vertex."""

    complex: SimplicialComplexRequest
    base_vertex: VertexLabel


class FundamentalGroupPresentationResult(StrictModel):
    """Deterministic edge-path presentation of the selected component."""

    complex: FiniteSimplicialComplex
    base_vertex: VertexLabel
    component_vertices: tuple[VertexLabel, ...] = Field(min_length=1)
    spanning_tree_edges: tuple[tuple[VertexLabel, VertexLabel], ...] = Field(default=())
    non_tree_edges: tuple[tuple[VertexLabel, VertexLabel], ...] = Field(default=())
    edge_words: tuple[EdgeWordEntry, ...] = Field(default=())
    triangle_relators: tuple[TriangleRelator, ...] = Field(
        default=(), max_length=MAX_PRESENTATION_RELATORS
    )
    presentation: FiniteGroupPresentation
    abelianization: AbelianizationResult

    @model_validator(mode="after")
    def require_presentation_coherence(self) -> Self:
        if self.base_vertex not in set(self.component_vertices):
            raise _validation_error(
                "fundamental_group.base_vertex",
                "the base vertex must lie in the selected component",
            )
        tree_edges = set(self.spanning_tree_edges)
        non_tree_edges = set(self.non_tree_edges)
        if tree_edges & non_tree_edges:
            raise _validation_error(
                "fundamental_group.edge_partition",
                "spanning tree and non-tree edges must be disjoint",
            )
        if {entry.edge for entry in self.edge_words} != tree_edges | non_tree_edges:
            raise _validation_error(
                "fundamental_group.edge_word_coverage",
                "edge words must cover exactly the component edges",
            )
        if any(
            entry.is_tree_edge != (entry.edge in tree_edges)
            for entry in self.edge_words
        ):
            raise _validation_error(
                "fundamental_group.edge_word_tree_flag",
                "edge words must flag exactly the spanning tree edges",
            )
        if tuple(sorted(self.spanning_tree_edges)) != self.spanning_tree_edges:
            raise _validation_error(
                "fundamental_group.tree_order",
                "spanning tree edges must be sorted",
            )
        if tuple(sorted(self.non_tree_edges)) != self.non_tree_edges:
            raise _validation_error(
                "fundamental_group.non_tree_order",
                "non-tree edges must be sorted",
            )
        if len(self.non_tree_edges) != len(self.presentation.generators):
            raise _validation_error(
                "fundamental_group.generator_count",
                "each non-tree edge must receive exactly one generator",
            )
        if len(self.triangle_relators) != len(self.presentation.relators):
            raise _validation_error(
                "fundamental_group.relator_count",
                "each two-simplex must receive exactly one relator",
            )
        for entry, relator in zip(
            self.triangle_relators, self.presentation.relators, strict=True
        ):
            if entry.word != relator:
                raise _validation_error(
                    "fundamental_group.relator_binding",
                    "triangle relators must bind to the presentation relators",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)
