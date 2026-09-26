"""Exact SU(2) gauge transforms and ordered holonomy over QQ quaternions."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_LABEL_LENGTH,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
)
from jacobian.math.gauge._su2_models import (
    SU2GaugeEdgeValue,
    SU2GaugeField,
    SU2GaugeTransformResult,
    SU2GaugeVertexValue,
    SU2HolonomyResult,
    SU2WilsonTraceResult,
)
from jacobian.math.quaternions import (
    RationalUnitQuaternion,
    inverse_rational_unit_quaternion,
    multiply_rational_unit_quaternions,
)

MAX_SU2_GAUGE_AGGREGATE_WORK = 100_000_000


def _value_digits(value: RationalUnitQuaternion) -> int:
    return max(
        canonical_rational_component_digits(coordinate)
        for coordinate in value.coordinates
    )


def _admit_aggregate_work(work: int, location: str) -> None:
    if work > MAX_SU2_GAUGE_AGGREGATE_WORK:
        raise OperationResourceAdmissionError(
            location=(location,),
            code="lattice_gauge.su2.aggregate_work_bound",
            message="SU(2) gauge operation exceeds its aggregate exact-work envelope",
        )


def _admit_su2_path(
    field: SU2GaugeField,
    path: OrientedGaugePath,
    labels: dict[str, RationalUnitQuaternion],
) -> tuple[str, str]:
    if not isinstance(path, OrientedGaugePath) or not isinstance(path.steps, tuple):
        _reject("path_shape", "path must be a typed oriented gauge path")
    if len(path.steps) > 256:
        _reject("path_bound", "path exceeds the 256-edge envelope")
    edges = {edge.edge_id: edge for edge in field.lattice.edges}
    aggregate_work = 0
    accumulated_digits = 1
    previous: tuple[str, bool] | None = None
    cursor: str | None = None
    start: str | None = None
    for step in path.steps:
        if (
            not isinstance(step, GaugePathStep)
            or type(getattr(step, "edge_id", None)) is not str
            or type(getattr(step, "forward", None)) is not bool
        ):
            _reject("path_shape", "path steps require a strict edge ID and orientation")
        edge = edges.get(step.edge_id)
        if edge is None or step.edge_id not in labels:
            _reject("unknown_edge", "path references an edge outside the field lattice")
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and tail != cursor:
            _reject("disconnected_path", "oriented path steps must chain head-to-tail")
        if start is None:
            start = tail
        cursor = head
        step_digits = _value_digits(labels[step.edge_id])
        traversal = (step.edge_id, step.forward)
        if (
            previous is not None
            and previous[0] == step.edge_id
            and previous[1] != step.forward
        ):
            accumulated_digits = 1
        aggregate_work += 64 * accumulated_digits * step_digits
        accumulated_digits += step_digits + 1
        previous = traversal
    _admit_aggregate_work(aggregate_work, "path")
    if not path.steps:
        basepoint: object = getattr(path, "basepoint", None)
        if not isinstance(basepoint, str) or type(basepoint) is not str:
            _reject("empty_path_basepoint", "empty path needs a lattice basepoint")
        if basepoint not in field.lattice.vertices:
            _reject("empty_path_basepoint", "empty path needs a lattice basepoint")
        return basepoint, basepoint
    if start is None or cursor is None:
        _reject("disconnected_path", "admitted path steps produced no endpoints")
    if path.basepoint is not None and path.basepoint != start:
        _reject("basepoint_mismatch", "path basepoint must equal its first vertex")
    return start, cursor


def _identity_quaternion() -> RationalUnitQuaternion:
    zero = CanonicalRational(num=0, den=1)
    return RationalUnitQuaternion(
        coordinates=(CanonicalRational(num=1, den=1), zero, zero, zero)
    )


def _compose_path(
    path: OrientedGaugePath,
    labels: dict[str, RationalUnitQuaternion],
) -> RationalUnitQuaternion:
    """Compose the exact ordered product along one admitted oriented path."""
    result = _identity_quaternion()
    for step in path.steps:
        factor = labels[step.edge_id]
        if not step.forward:
            factor = inverse_rational_unit_quaternion(factor)
        result = multiply_rational_unit_quaternions(result, factor)
    return result


def _valid_label(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= MAX_GAUGE_LABEL_LENGTH
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("field",), code=f"lattice_gauge.su2.{code}", message=message
    )


def _admit_field(field: object) -> SU2GaugeField:
    if not isinstance(field, SU2GaugeField):
        _reject("field_type", "expected an exact rational SU(2) gauge field")
    lattice = getattr(field, "lattice", None)
    values = getattr(field, "edge_values", None)
    if not isinstance(lattice, GaugeLattice) or not isinstance(values, tuple):
        _reject("field_shape", "SU(2) gauge field is malformed")
    vertices = getattr(lattice, "vertices", None)
    edges = getattr(lattice, "edges", None)
    if (
        not isinstance(vertices, tuple)
        or not 1 <= len(vertices) <= 64
        or any(not _valid_label(vertex) for vertex in vertices)
        or len(set(vertices)) != len(vertices)
        or not isinstance(edges, tuple)
        or not 1 <= len(edges) <= 128
    ):
        _reject("field_shape", "SU(2) gauge lattice is malformed")
    for edge in edges:
        if (
            not _valid_label(getattr(edge, "edge_id", None))
            or not _valid_label(getattr(edge, "tail", None))
            or not _valid_label(getattr(edge, "head", None))
            or edge.tail not in vertices
            or edge.head not in vertices
        ):
            _reject("field_shape", "SU(2) lattice edge is malformed")
    if tuple(edge.edge_id for edge in edges) != tuple(
        sorted({edge.edge_id for edge in edges})
    ):
        _reject("field_shape", "SU(2) lattice edge IDs must be unique and sorted")
    if len(values) != len(lattice.edges) or tuple(
        getattr(entry, "edge_id", None) for entry in values
    ) != tuple(edge.edge_id for edge in lattice.edges):
        _reject("field_coverage", "SU(2) values must cover the lattice edges in order")
    for entry in values:
        if not isinstance(entry, SU2GaugeEdgeValue):
            _reject("field_shape", "every edge value must be typed")
        # The quaternion kernel re-admits the exact norm and rational coordinates.
        inverse_rational_unit_quaternion(entry.value)
    return field


def su2_gauge_transform(
    field: SU2GaugeField,
    vertex_values: tuple[SU2GaugeVertexValue, ...],
) -> SU2GaugeTransformResult:
    """Apply ``U'_(u->v) = g_u U_(u->v) g_v^-1`` edgewise."""
    field = _admit_field(field)
    if (
        not isinstance(vertex_values, tuple)
        or len(vertex_values) > len(field.lattice.vertices)
        or any(not isinstance(entry, SU2GaugeVertexValue) for entry in vertex_values)
    ):
        _reject("transform_vertices", "gauge frames are malformed")
    frames: dict[str, RationalUnitQuaternion] = {}
    for entry in vertex_values:
        frames[entry.vertex] = entry.value
    vertices = field.lattice.vertices
    if set(frames) != set(vertices) or len(frames) != len(vertex_values):
        _reject("transform_vertices", "gauge frames must cover every lattice vertex")
    for value in frames.values():
        inverse_rational_unit_quaternion(value)
    transformed = []
    source_by_id: dict[str, RationalUnitQuaternion] = {}
    for link in field.edge_values:
        source_by_id[link.edge_id] = link.value
    aggregate_work = 0
    for edge in field.lattice.edges:
        left_digits = _value_digits(frames[edge.tail])
        link_digits = _value_digits(source_by_id[edge.edge_id])
        right_digits = _value_digits(frames[edge.head])
        aggregate_work += 64 * left_digits * link_digits
        aggregate_work += 64 * (left_digits + link_digits + 1) * right_digits
    _admit_aggregate_work(aggregate_work, "field")
    for edge in field.lattice.edges:
        left = multiply_rational_unit_quaternions(
            frames[edge.tail], source_by_id[edge.edge_id]
        )
        value = multiply_rational_unit_quaternions(
            left, inverse_rational_unit_quaternion(frames[edge.head])
        )
        transformed.append(SU2GaugeEdgeValue(edge_id=edge.edge_id, value=value))
    output = SU2GaugeField(lattice=field.lattice, edge_values=tuple(transformed))
    canonical_frames = tuple(
        SU2GaugeVertexValue(vertex=vertex, value=frames[vertex])
        for vertex in field.lattice.vertices
    )
    return SU2GaugeTransformResult(
        source=field, transformed=output, vertex_values=canonical_frames
    )


def su2_path_holonomy(
    field: SU2GaugeField, path: OrientedGaugePath
) -> SU2HolonomyResult:
    """Compose exact SU(2) edge labels along one oriented lattice path."""
    field = _admit_field(field)
    labels: dict[str, RationalUnitQuaternion] = {}
    for entry in field.edge_values:
        labels[entry.edge_id] = entry.value
    start, end = _admit_su2_path(field, path, labels)
    result = _compose_path(path, labels)
    return SU2HolonomyResult(
        field=field, path=path, holonomy=result, start=start, end=end
    )


def su2_wilson_trace(holonomy: SU2HolonomyResult) -> SU2WilsonTraceResult:
    """Return the exact fundamental-representation trace ``2 Re(U)``.

    The caller-authored holonomy quaternion is never trusted: the ordered
    product is recomputed from the source-bound field and path and the trace
    is taken of that exact composition.
    """
    if not isinstance(holonomy, SU2HolonomyResult):
        _reject("wilson_parent", "Wilson trace requires a typed SU(2) holonomy")
    field = _admit_field(holonomy.field)
    labels: dict[str, RationalUnitQuaternion] = {}
    for entry in field.edge_values:
        labels[entry.edge_id] = entry.value
    start, end = _admit_su2_path(
        field,
        holonomy.path,
        labels,
    )
    if holonomy.start != start or holonomy.end != end:
        _reject(
            "holonomy_endpoint_mismatch",
            "holonomy endpoints must match the source path",
        )
    if start != end:
        _reject("wilson_open_path", "Wilson trace requires a closed path")
    authored = holonomy.holonomy
    if not isinstance(authored, RationalUnitQuaternion) or not isinstance(
        authored.coordinates, tuple
    ):
        _reject("wilson_parent", "holonomy carries a malformed quaternion")
    recomputed = _compose_path(holonomy.path, labels)
    if recomputed.coordinates != authored.coordinates:
        _reject(
            "holonomy_composition_mismatch",
            "holonomy quaternion must equal the ordered product of its source path",
        )
    real_part = recomputed.coordinates[0]
    numerator = real_part.num * 2
    if len(str(abs(numerator))) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("holonomy", "holonomy", "coordinates", 0),
            code="lattice_gauge.su2.wilson_trace_output_bound",
            message="Wilson trace exceeds the canonical rational digit bound",
        )
    scalar = Fraction(numerator, real_part.den)
    trace = CanonicalRational.from_fraction(scalar)
    return SU2WilsonTraceResult(holonomy=holonomy, trace=trace)


__all__ = ["su2_gauge_transform", "su2_path_holonomy", "su2_wilson_trace"]
