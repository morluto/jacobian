"""Typed contracts for exact planar framework rigidity profiles."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.geometry.exact._models import (
    MAX_PAIRS,
    MAX_POINTS,
    PointConfiguration,
)
from jacobian.math.geometry.framework._bounds import (
    MAX_FRAMEWORK_COORDINATE_WORK,
    difference_work,
    rational_parse_work,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices._operation_models import MatrixRankResult
from jacobian.math.matrices.values import SparseRationalMatrix


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"geometry.framework.{reason}", message)


def _require_planar_framework_shape(
    configuration: PointConfiguration,
    graph: SimpleUndirectedGraph,
) -> None:
    """Require one planar realization of exactly the graph's labelled vertices."""

    if len(configuration.points[0].coordinates) != 2:
        raise _validation_error(
            "configuration_must_be_planar",
            "framework configuration points must have exactly two coordinates",
        )
    point_labels = tuple(point.label for point in configuration.points)
    if len(graph.vertices) != len(point_labels) or set(graph.vertices) != set(
        point_labels
    ):
        raise _validation_error(
            "graph_vertices_must_match_point_labels",
            "graph vertices must equal the configuration point-label set",
        )


def _raw_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _raw_rational_components(value: object) -> tuple[str, str] | None:
    numerator = _raw_field(value, "num")
    denominator = _raw_field(value, "den")
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        return None
    return numerator, denominator


def _raw_coordinate_map(
    raw_points: list[object] | tuple[object, ...],
) -> tuple[
    dict[str, tuple[tuple[str, str], tuple[str, str]]],
    int,
]:
    coordinates: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    source_parse_work = 0
    for raw_point in raw_points:
        label = _raw_field(raw_point, "label")
        raw_coordinates = _raw_field(raw_point, "coordinates")
        if not isinstance(label, str) or not isinstance(raw_coordinates, (list, tuple)):
            continue
        if len(raw_coordinates) != 2:
            raise _validation_error(
                "configuration_must_be_planar",
                "framework configuration points must have exactly two coordinates",
            )
        component_pairs = tuple(
            components
            for raw_coordinate in raw_coordinates
            if (components := _raw_rational_components(raw_coordinate)) is not None
        )
        source_parse_work += sum(map(rational_parse_work, component_pairs))
        if source_parse_work > MAX_FRAMEWORK_COORDINATE_WORK:
            raise _coordinate_work_error()
        if len(component_pairs) == 2:
            coordinates[label] = (component_pairs[0], component_pairs[1])
    return coordinates, source_parse_work


def _coordinate_work_error() -> PydanticCustomError:
    return _validation_error(
        "coordinate_work_exceeds_bound",
        "framework coordinate normalization and edge differences exceed "
        f"the {MAX_FRAMEWORK_COORDINATE_WORK:,}-unit work bound",
    )


def _require_raw_execution_envelope(data: object) -> None:
    if not isinstance(data, dict):
        return
    raw_points = _raw_field(data.get("configuration"), "points")
    raw_edges = _raw_field(data.get("graph"), "edges")
    if not isinstance(raw_points, (list, tuple)):
        return
    if len(raw_points) > MAX_POINTS:
        raise _validation_error(
            "point_count_exceeds_bound",
            f"framework configurations contain at most {MAX_POINTS} points",
        )
    coordinates, total_work = _raw_coordinate_map(raw_points)
    if not isinstance(raw_edges, (list, tuple)):
        return
    if len(raw_edges) > MAX_PAIRS:
        raise _validation_error(
            "edge_count_exceeds_bound",
            f"a graph on a framework configuration contains at most {MAX_PAIRS} edges",
        )
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            continue
        left_label, right_label = raw_edge
        if left_label not in coordinates or right_label not in coordinates:
            continue
        total_work += sum(
            difference_work(left_coordinate, right_coordinate)
            for left_coordinate, right_coordinate in zip(
                coordinates[left_label], coordinates[right_label], strict=True
            )
        )
        if total_work > MAX_FRAMEWORK_COORDINATE_WORK:
            raise _coordinate_work_error()


class PlanarRigidityProfileRequest(StrictModel):
    """One labelled rational planar framework.

    The configuration's point order defines the vertex and coordinate-column
    axis. The graph must contain exactly the same labels. Graph edge tuple order
    is not authoritative: the result derives a lexicographically sorted edge
    axis.
    """

    configuration: PointConfiguration = Field(
        description=(
            "A labelled rational point configuration in exactly two dimensions; "
            "its declared point order is the rigidity-matrix vertex axis."
        )
    )
    graph: SimpleUndirectedGraph = Field(
        description=(
            "A simple undirected graph whose vertex set exactly equals the "
            "configuration point-label set. Edge tuple order is ignored when "
            "deriving the sorted result edge axis."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_execution_envelope(cls, data: Any) -> Any:
        """Reject excessive coordinate work before nested rational parsing."""

        _require_raw_execution_envelope(data)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_planar_framework(self) -> Self:
        _require_planar_framework_shape(self.configuration, self.graph)
        return self


class PlanarRigidityProfile(StrictModel):
    """Exact source-bound rigidity matrix and rational rank of one framework.

    ``matrix_rank.matrix`` has one row per ``edge_axis`` entry and two columns
    per ``vertex_axis`` entry. Columns ``2*i`` and ``2*i + 1`` are respectively
    the x and y coordinates of vertex ``vertex_axis[i]``. A false
    ``is_infinitesimally_rigid`` value says only that the supplied realization
    fails the infinitesimal rank criterion; it is not a local- or global-
    non-rigidity conclusion.
    """

    configuration: PointConfiguration
    graph: SimpleUndirectedGraph
    vertex_axis: tuple[str, ...] = Field(min_length=2, max_length=MAX_POINTS)
    edge_axis: tuple[tuple[str, str], ...] = Field(max_length=MAX_PAIRS)
    matrix_rank: MatrixRankResult
    maximal_infinitesimal_rigidity_rank: int = Field(ge=1, le=2 * MAX_POINTS - 3)
    is_infinitesimally_rigid: bool

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)

    @model_validator(mode="after")
    def require_structural_source_binding(self) -> Self:
        _require_planar_framework_shape(self.configuration, self.graph)
        source_vertex_axis = tuple(point.label for point in self.configuration.points)
        if self.vertex_axis != source_vertex_axis:
            raise _validation_error(
                "vertex_axis_must_follow_configuration_order",
                "vertex axis must equal the configuration's declared point order",
            )
        if self.edge_axis != tuple(sorted(self.graph.edges)):
            raise _validation_error(
                "edge_axis_must_be_sorted_graph_edges",
                "edge axis must be the lexicographically sorted graph edges",
            )
        matrix = self.matrix_rank.matrix
        if not isinstance(matrix, SparseRationalMatrix):
            raise _validation_error(
                "rigidity_matrix_must_be_coordinate_sparse",
                "rigidity matrix must use the coordinate-sparse rational carrier",
            )
        if matrix.row_count != len(self.edge_axis) or matrix.column_count != 2 * len(
            self.vertex_axis
        ):
            raise _validation_error(
                "rigidity_matrix_axes_must_match_framework_axes",
                "rigidity matrix axes must be |edge_axis| by 2|vertex_axis|",
            )
        maximal_rank = 2 * len(self.vertex_axis) - 3
        if self.maximal_infinitesimal_rigidity_rank != maximal_rank:
            raise _validation_error(
                "maximal_rank_must_equal_two_vertices_minus_three",
                "maximal infinitesimal-rigidity rank must equal 2|V|-3",
            )
        if self.is_infinitesimally_rigid != (self.matrix_rank.rank == maximal_rank):
            raise _validation_error(
                "rigidity_decision_must_match_rank",
                "infinitesimal-rigidity decision must equal rank == 2|V|-3",
            )
        return self


__all__ = [
    "PlanarRigidityProfile",
    "PlanarRigidityProfileRequest",
]
