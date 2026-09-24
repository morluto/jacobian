"""Typed contracts for exact permutation-valued lattice-gauge holonomy."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lattice_gauge.{reason}", message)


MAX_GAUGE_VERTICES = 64
"""Maximum vertices in one admitted gauge lattice."""

MAX_GAUGE_EDGES = 128
"""Maximum canonically oriented edges in one admitted gauge lattice."""

MAX_GAUGE_DEGREE = 8
"""Maximum permutation degree of the structure group S_d."""

MIN_GAUGE_DEGREE = 1
"""Minimum permutation degree; degree one is the trivial structure group."""

MAX_GAUGE_PATH_LENGTH = 256
"""Maximum oriented steps in one admitted lattice path."""

MAX_GAUGE_LABEL_LENGTH = 64
"""Maximum length of a vertex or edge identifier."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


GaugeLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_GAUGE_LABEL_LENGTH, strict=True),
    AfterValidator(_require_scalar_label),
]
"""One vertex or edge identifier in a finite gauge lattice."""


class GaugeEdge(StrictModel):
    """One canonically oriented lattice edge with stable identity."""

    edge_id: GaugeLabel
    tail: GaugeLabel
    head: GaugeLabel


class GaugeLattice(StrictModel):
    """A finite vertex domain with canonically oriented edge IDs."""

    vertices: tuple[GaugeLabel, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )
    edges: tuple[GaugeEdge, ...] = Field(min_length=1, max_length=MAX_GAUGE_EDGES)

    @model_validator(mode="after")
    def require_canonical_lattice(self) -> Self:
        if len(set(self.vertices)) != len(self.vertices):
            raise _validation_error(
                "lattice_vertices_unique", "lattice vertices must be unique"
            )
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if tuple(sorted(edge_ids)) != edge_ids or len(set(edge_ids)) != len(edge_ids):
            raise _validation_error(
                "lattice_edge_ids",
                "lattice edge IDs must be unique and strictly ordered",
            )
        known = set(self.vertices)
        if any(edge.tail not in known or edge.head not in known for edge in self.edges):
            raise _validation_error(
                "lattice_endpoints",
                "every edge endpoint must be a lattice vertex",
            )
        return self


class PermutationLabel(StrictModel):
    """One exact permutation of ``0..degree-1`` acting on the right.

    The image tuple sends each point to its image; composition applies
    left-to-right along traversal order. Degree one represents the trivial
    group and has the unique identity permutation ``(0,)``.
    """

    degree: StrictInt = Field(ge=MIN_GAUGE_DEGREE, le=MAX_GAUGE_DEGREE)
    image: tuple[StrictInt, ...] = Field(
        min_length=MIN_GAUGE_DEGREE, max_length=MAX_GAUGE_DEGREE
    )

    @model_validator(mode="after")
    def require_permutation_shape(self) -> Self:
        if len(self.image) != self.degree:
            raise _validation_error(
                "permutation_shape",
                "permutation image must carry exactly degree entries",
            )
        if sorted(self.image) != list(range(self.degree)):
            raise _validation_error(
                "permutation_bijective",
                "permutation image must be a bijection of 0..degree-1",
            )
        return self


class GaugeFieldEdgeLabel(StrictModel):
    """The exact group element on one canonically oriented edge."""

    edge_id: GaugeLabel
    label: PermutationLabel


class GaugeField(StrictModel):
    """One exact permutation-valued edge field over a lattice.

    Every canonically oriented edge carries exactly one label; reverse
    traversal uses the exact inverse and is never submitted independently.
    """

    lattice: GaugeLattice
    degree: StrictInt = Field(ge=MIN_GAUGE_DEGREE, le=MAX_GAUGE_DEGREE)
    edge_labels: tuple[GaugeFieldEdgeLabel, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_EDGES
    )

    @model_validator(mode="after")
    def require_complete_edge_field(self) -> Self:
        if any(label.label.degree != self.degree for label in self.edge_labels):
            raise _validation_error(
                "field_degree",
                "every edge label must use the field structure degree",
            )
        have = tuple(label.edge_id for label in self.edge_labels)
        want = tuple(edge.edge_id for edge in self.lattice.edges)
        if tuple(sorted(have)) != tuple(sorted(set(have))) or set(have) != set(want):
            raise _validation_error(
                "field_coverage",
                "edge labels must cover every lattice edge exactly once",
            )
        return self


class GaugePathStep(StrictModel):
    """One oriented traversal of a lattice edge."""

    edge_id: GaugeLabel
    forward: bool


class OrientedGaugePath(StrictModel):
    """An ordered edge walk, or a based zero-length identity path."""

    steps: tuple[GaugePathStep, ...] = Field(max_length=MAX_GAUGE_PATH_LENGTH)
    basepoint: GaugeLabel | None = None

    @model_validator(mode="after")
    def require_basepoint_for_empty_path(self) -> Self:
        if not self.steps and self.basepoint is None:
            raise _validation_error(
                "empty_path_basepoint",
                "a zero-length path must name its identity-path basepoint",
            )
        return self


class EdgeContribution(StrictModel):
    """The resolved oriented group value of one path step."""

    edge_id: GaugeLabel
    forward: bool
    value: PermutationLabel


class GaugeVertexValue(StrictModel):
    """One exact gauge-frame element at a lattice vertex."""

    vertex: GaugeLabel
    value: PermutationLabel


class GaugeTransformRequest(StrictModel):
    """A finite gauge transform bound to one lattice and group degree."""

    field: GaugeField
    vertex_values: tuple[GaugeVertexValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )

    @model_validator(mode="after")
    def require_complete_vertex_values(self) -> Self:
        vertices = self.field.lattice.vertices
        labels = {entry.vertex: entry.value for entry in self.vertex_values}
        if set(labels) != set(vertices) or len(labels) != len(self.vertex_values):
            raise _validation_error(
                "transform_vertices",
                "gauge transform must label every lattice vertex exactly once",
            )
        if any(value.degree != self.field.degree for value in labels.values()):
            raise _validation_error(
                "transform_degree", "gauge-frame elements must use the field degree"
            )
        return self


class GaugeTransformResult(StrictModel):
    """The transformed edge field and retained vertex-frame map."""

    source: GaugeField
    transformed: GaugeField
    vertex_values: tuple[GaugeVertexValue, ...]

    @model_validator(mode="after")
    def require_parent_binding(self) -> Self:
        if (
            not isinstance(self.source, GaugeField)
            or not isinstance(self.transformed, GaugeField)
            or not isinstance(self.source.lattice, GaugeLattice)
            or not isinstance(self.transformed.lattice, GaugeLattice)
        ):
            raise _validation_error(
                "transform_parent", "source and transformed values must be gauge fields"
            )
        if self.source.lattice != self.transformed.lattice:
            raise _validation_error(
                "transform_parent",
                "source and transformed fields must share one lattice",
            )
        if self.source.degree != self.transformed.degree:
            raise _validation_error(
                "transform_degree",
                "source and transformed fields must share one group degree",
            )
        vertices = self.source.lattice.vertices
        if not isinstance(self.vertex_values, tuple) or any(
            not isinstance(entry, GaugeVertexValue) for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_vertices", "vertex frames must be typed gauge values"
            )
        if any(
            not isinstance(entry.value, PermutationLabel)
            or type(entry.value.degree) is not int
            or not isinstance(entry.value.image, tuple)
            or len(entry.value.image) != entry.value.degree
            or any(type(value) is not int for value in entry.value.image)
            or sorted(entry.value.image) != list(range(entry.value.degree))
            for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_vertices", "vertex frames must be valid permutation values"
            )
        labels = tuple(entry.vertex for entry in self.vertex_values)
        if tuple(sorted(labels)) != tuple(sorted(vertices)) or len(set(labels)) != len(
            labels
        ):
            raise _validation_error(
                "transform_vertices",
                "vertex frames must cover the source lattice exactly once",
            )
        if any(
            entry.value.degree != self.source.degree for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_degree",
                "every vertex frame must use the source group degree",
            )
        return self


class PlaquetteRequest(StrictModel):
    """A closed oriented lattice path whose holonomy is curvature."""

    field: GaugeField
    path: OrientedGaugePath


class PlaquetteResult(StrictModel):
    """Exact oriented plaquette curvature bound to field and path."""

    field: GaugeField
    path: OrientedGaugePath
    curvature: PermutationLabel
    start: GaugeLabel

    @model_validator(mode="after")
    def require_path_and_field_binding(self) -> Self:
        if (
            not isinstance(self.field, GaugeField)
            or not isinstance(self.field.lattice, GaugeLattice)
            or not isinstance(self.path, OrientedGaugePath)
            or not isinstance(self.curvature, PermutationLabel)
            or type(self.field.degree) is not int
            or type(self.curvature.degree) is not int
            or any(not isinstance(edge, GaugeEdge) for edge in self.field.lattice.edges)
        ):
            raise _validation_error(
                "plaquette_parent", "plaquette source and path are malformed"
            )
        if (
            not isinstance(self.curvature.image, tuple)
            or len(self.curvature.image) != self.curvature.degree
            or any(type(value) is not int for value in self.curvature.image)
            or sorted(self.curvature.image) != list(range(self.curvature.degree))
        ):
            raise _validation_error(
                "plaquette_parent", "plaquette curvature is not a permutation value"
            )
        if self.curvature.degree != self.field.degree:
            raise _validation_error(
                "plaquette_degree", "curvature must use the field's group degree"
            )
        by_id = {edge.edge_id: edge for edge in self.field.lattice.edges}
        if not isinstance(self.path.steps, tuple) or not self.path.steps:
            raise _validation_error("plaquette_path", "plaquette path must be nonempty")
        cursor: str | None = None
        first: str | None = None
        for step in self.path.steps:
            if not isinstance(step, GaugePathStep) or type(step.forward) is not bool:
                raise _validation_error(
                    "plaquette_path", "plaquette steps must be typed lattice traversals"
                )
            edge = by_id.get(step.edge_id)
            if edge is None:
                raise _validation_error(
                    "plaquette_path", "plaquette path must use source-lattice edges"
                )
            tail, head = (
                (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
            )
            if cursor is not None and tail != cursor:
                raise _validation_error(
                    "plaquette_path", "plaquette steps must chain head-to-tail"
                )
            if first is None:
                first = tail
            cursor = head
        if first != cursor or self.start != first:
            raise _validation_error(
                "plaquette_path",
                "plaquette start must bind to a closed source-lattice path",
            )
        return self


class HolonomyRequest(StrictModel):
    """Compute the ordered exact group product along an oriented edge path.

    Path steps must chain head-to-tail over the field's lattice, and every
    step must reference a labelled lattice edge. Reverse traversal resolves
    to the exact group inverse of the forward label.
    """

    field: GaugeField
    path: OrientedGaugePath


class PermutationWilsonTraceRequest(StrictModel):
    """Evaluate the natural permutation-character Wilson loop over ``S_d``."""

    field: GaugeField
    path: OrientedGaugePath


class PermutationWilsonTraceResult(StrictModel):
    """Exact trace in the natural degree-``d`` permutation representation."""

    field: GaugeField
    path: OrientedGaugePath
    holonomy: PermutationLabel
    trace: StrictInt = Field(ge=0, le=MAX_GAUGE_DEGREE)


class HolonomyResult(StrictModel):
    """Ordered holonomy bound to its source field and oriented path."""

    field: GaugeField
    path: OrientedGaugePath

    holonomy: PermutationLabel
    contributions: tuple[EdgeContribution, ...] = Field(
        max_length=MAX_GAUGE_PATH_LENGTH
    )
    start: GaugeLabel
    end: GaugeLabel

    @model_validator(mode="after")
    def require_holonomy_shape(self) -> Self:
        if any(
            contribution.value.degree != self.holonomy.degree
            for contribution in self.contributions
        ):
            raise _validation_error(
                "holonomy_degree",
                "every contribution must use the holonomy structure degree",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: GaugeField,
        path: OrientedGaugePath,
        holonomy: PermutationLabel,
        contributions: tuple[EdgeContribution, ...],
        start: str,
        end: str,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its product."""

        return cls.model_construct(
            field=field,
            path=path,
            holonomy=holonomy,
            contributions=contributions,
            start=start,
            end=end,
        )


__all__ = [
    "MAX_GAUGE_DEGREE",
    "MAX_GAUGE_EDGES",
    "MAX_GAUGE_LABEL_LENGTH",
    "MAX_GAUGE_PATH_LENGTH",
    "MAX_GAUGE_VERTICES",
    "MIN_GAUGE_DEGREE",
    "EdgeContribution",
    "GaugeEdge",
    "GaugeField",
    "GaugeFieldEdgeLabel",
    "GaugeLabel",
    "GaugeLattice",
    "GaugePathStep",
    "GaugeTransformRequest",
    "GaugeTransformResult",
    "GaugeVertexValue",
    "HolonomyRequest",
    "HolonomyResult",
    "OrientedGaugePath",
    "PermutationLabel",
    "PlaquetteRequest",
    "PlaquetteResult",
]
