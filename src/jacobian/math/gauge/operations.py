"""Native exact permutation-valued lattice-gauge path holonomy."""

from __future__ import annotations

from typing import NoReturn, cast

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_LABEL_LENGTH,
    MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS,
    MAX_GAUGE_LOOP_FAMILY_SIZE,
    MAX_GAUGE_LOOP_FAMILY_STEPS,
    MAX_GAUGE_LOOP_FAMILY_WORK,
    MAX_GAUGE_PATH_LENGTH,
    MIN_GAUGE_DEGREE,
    EdgeContribution,
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugeLoopFamilyHolonomies,
    GaugeLoopHolonomy,
    GaugePathStep,
    GaugeTransformResult,
    GaugeVertexValue,
    HolonomyResult,
    OrientedGaugePath,
    PermutationLabel,
    PlaquetteResult,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Left-to-right permutation composition: apply ``first``, then ``second``."""

    return tuple(second[first[i]] for i in range(len(first)))


def _inverse(image: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(image)
    for i, j in enumerate(image):
        result[j] = i
    return tuple(result)


def _admit_holonomy(field: GaugeField, path: OrientedGaugePath) -> None:
    """Enforce the shared envelope for native and catalog calls."""

    # Re-admit the complete nested field before path lookup or permutation
    # arithmetic. ``model_construct`` is a supported native escape hatch for
    # trusted producers, so merely checking the outer GaugeField type is not a
    # sufficient public boundary.
    lattice, labels = _admit_transform_field(field)
    if type(path) is not OrientedGaugePath:
        _reject(
            "path",
            "lattice_gauge.holonomy.path_not_a_gauge_path",
            "holonomy path must be an oriented lattice edge path",
        )
    steps = getattr(path, "steps", _MISSING)
    if type(steps) is not tuple:
        _reject(
            "path",
            "lattice_gauge.holonomy.path_shape",
            "holonomy path steps must be a tuple of oriented edge steps",
        )
    if len(steps) > 256:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="lattice_gauge.holonomy.path_over_envelope",
            message="oriented path exceeds the 256-step envelope",
        )
    by_id = {edge.edge_id: edge for edge in lattice.edges}
    if not steps:
        basepoint = getattr(path, "basepoint", _MISSING)
        if not _gauge_label_is_valid(basepoint) or basepoint not in lattice.vertices:
            _reject(
                "path",
                "lattice_gauge.holonomy.empty_path_basepoint",
                "a zero-length path basepoint must name a vertex of the source lattice",
            )
        return
    cursor: str | None = None
    first: str | None = None
    for step in steps:
        if not isinstance(step, GaugePathStep):
            _reject(
                "path",
                "lattice_gauge.holonomy.step_shape",
                "every path step must be a typed lattice traversal",
            )
        edge_id = getattr(step, "edge_id", _MISSING)
        forward = getattr(step, "forward", _MISSING)
        if not _gauge_label_is_valid(edge_id) or type(forward) is not bool:
            _reject(
                "path",
                "lattice_gauge.holonomy.step_shape",
                "path steps require a strict edge label and boolean orientation",
            )
        edge = by_id.get(cast(str, edge_id))
        if edge is None or edge_id not in labels:
            _reject(
                "path",
                "lattice_gauge.holonomy.unknown_edge_step",
                f"path step references unknown lattice edge {edge_id!r}",
            )
        tail, head = (edge.tail, edge.head) if forward else (edge.head, edge.tail)
        if cursor is not None and tail != cursor:
            _reject(
                "path",
                "lattice_gauge.holonomy.disconnected_path",
                "oriented path steps must chain head-to-tail",
            )
        if first is None:
            first = tail
        cursor = head
    basepoint = getattr(path, "basepoint", None)
    if basepoint is not None and basepoint != first:
        _reject(
            "path",
            "lattice_gauge.holonomy.basepoint_mismatch",
            "a supplied path basepoint must equal its first vertex",
        )


_MISSING = object()


def _gauge_label_is_valid(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 64
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _permutation_is_valid(value: object, degree: int) -> bool:
    if not isinstance(value, PermutationLabel):
        return False
    value_degree = getattr(value, "degree", _MISSING)
    image = getattr(value, "image", _MISSING)
    return (
        type(value_degree) is int
        and value_degree == degree
        and MIN_GAUGE_DEGREE <= value_degree <= 8
        and type(image) is tuple
        and len(image) == degree
        and all(type(entry) is int for entry in image)
        and sorted(image) == list(range(degree))
    )


def _admit_transform_field(
    field: object,
) -> tuple[GaugeLattice, dict[str, PermutationLabel]]:
    """Re-admit every nested carrier before the transform indexes it.

    Native callers can use ``model_construct`` and therefore bypass all model
    validators.  Keep this admission local to the owner so malformed nested
    carriers cannot turn dictionary lookups or permutation arithmetic into
    incidental ``AttributeError``/``KeyError`` failures.
    """
    if not isinstance(field, GaugeField):
        _reject(
            "field",
            "lattice_gauge.transform.field_not_a_gauge_field",
            "transform source must be a gauge field",
        )
    lattice = getattr(field, "lattice", _MISSING)
    degree = getattr(field, "degree", _MISSING)
    edge_labels = getattr(field, "edge_labels", _MISSING)
    if not isinstance(lattice, GaugeLattice):
        _reject(
            "field", "lattice_gauge.transform.field_shape", "field lattice is malformed"
        )
    if type(degree) is not int or not MIN_GAUGE_DEGREE <= degree <= 8:
        _reject(
            "field", "lattice_gauge.transform.field_degree", "field degree is malformed"
        )
    vertices = getattr(lattice, "vertices", _MISSING)
    edges = getattr(lattice, "edges", _MISSING)
    if (
        type(vertices) is not tuple
        or not 1 <= len(vertices) <= 64
        or any(not _gauge_label_is_valid(vertex) for vertex in vertices)
        or len(set(vertices)) != len(vertices)
    ):
        _reject(
            "field",
            "lattice_gauge.transform.lattice_shape",
            "lattice vertices are malformed",
        )
    if type(edges) is not tuple or not 1 <= len(edges) <= 128:
        _reject(
            "field",
            "lattice_gauge.transform.lattice_shape",
            "lattice edges are malformed",
        )
    edge_ids: list[str] = []
    for edge in edges:
        if not isinstance(edge, GaugeEdge):
            _reject(
                "field",
                "lattice_gauge.transform.lattice_shape",
                "lattice edge is malformed",
            )
        edge_id = cast(str, getattr(edge, "edge_id", _MISSING))
        tail = cast(str, getattr(edge, "tail", _MISSING))
        head = cast(str, getattr(edge, "head", _MISSING))
        if (
            not _gauge_label_is_valid(edge_id)
            or not _gauge_label_is_valid(tail)
            or not _gauge_label_is_valid(head)
            or tail not in vertices
            or head not in vertices
        ):
            _reject(
                "field",
                "lattice_gauge.transform.lattice_shape",
                "lattice edge is malformed",
            )
        edge_ids.append(edge_id)
    if tuple(sorted(edge_ids)) != tuple(edge_ids) or len(set(edge_ids)) != len(
        edge_ids
    ):
        _reject(
            "field",
            "lattice_gauge.transform.lattice_shape",
            "lattice edge IDs are malformed",
        )
    if type(edge_labels) is not tuple or len(edge_labels) != len(edges):
        _reject(
            "field",
            "lattice_gauge.transform.field_coverage",
            "edge labels are incomplete",
        )
    labels: dict[str, PermutationLabel] = {}
    for entry in edge_labels:
        if not isinstance(entry, GaugeFieldEdgeLabel):
            _reject(
                "field",
                "lattice_gauge.transform.field_shape",
                "edge label is malformed",
            )
        edge_id = cast(str, getattr(entry, "edge_id", _MISSING))
        label = getattr(entry, "label", _MISSING)
        if (
            not _gauge_label_is_valid(edge_id)
            or edge_id in labels
            or edge_id not in edge_ids
            or not _permutation_is_valid(label, degree)
        ):
            _reject(
                "field",
                "lattice_gauge.transform.field_coverage",
                "edge labels are malformed",
            )
        labels[edge_id] = cast(PermutationLabel, label)
    if set(labels) != set(edge_ids):
        _reject(
            "field",
            "lattice_gauge.transform.field_coverage",
            "edge labels are incomplete",
        )
    return lattice, labels


def gauge_transform(
    field: GaugeField,
    vertex_values: tuple[GaugeVertexValue, ...] | list[GaugeVertexValue],
) -> GaugeTransformResult:
    """Apply ``U'_e = g_tail U_e g_head^-1`` on one finite lattice."""
    lattice, labels = _admit_transform_field(field)
    if not isinstance(vertex_values, (tuple, list)) or len(vertex_values) > 64:
        _reject(
            "vertex_values",
            "lattice_gauge.transform.vertex_values_type",
            "vertex values must be a finite labelled family",
        )
    by_vertex: dict[str, PermutationLabel] = {}
    for entry in vertex_values:
        if not isinstance(entry, GaugeVertexValue):
            _reject(
                "vertex_values",
                "lattice_gauge.transform.vertex_values_type",
                "vertex value is malformed",
            )
        vertex = cast(str, getattr(entry, "vertex", _MISSING))
        frame_value = cast(PermutationLabel, getattr(entry, "value", _MISSING))
        if not _gauge_label_is_valid(vertex) or vertex in by_vertex:
            _reject(
                "vertex_values",
                "lattice_gauge.transform.vertex_coverage",
                "transform vertex coverage is malformed",
            )
        if not _permutation_is_valid(frame_value, field.degree):
            _reject(
                "vertex_values",
                "lattice_gauge.transform.degree_mismatch",
                "frame element is malformed",
            )
        by_vertex[vertex] = frame_value
    if set(by_vertex) != set(lattice.vertices):
        _reject(
            "vertex_values",
            "lattice_gauge.transform.vertex_coverage",
            "transform must label every lattice vertex exactly once",
        )
    transformed_labels = []
    for edge in lattice.edges:
        value = _compose(
            _compose(by_vertex[edge.tail].image, labels[edge.edge_id].image),
            _inverse(by_vertex[edge.head].image),
        )
        transformed_labels.append(
            GaugeFieldEdgeLabel(
                edge_id=edge.edge_id,
                label=PermutationLabel(degree=field.degree, image=value),
            )
        )
    transformed = GaugeField(
        lattice=lattice,
        degree=field.degree,
        edge_labels=tuple(transformed_labels),
    )
    canonical_values = tuple(
        GaugeVertexValue(vertex=vertex, value=by_vertex[vertex])
        for vertex in lattice.vertices
    )
    return GaugeTransformResult(
        source=field, transformed=transformed, vertex_values=canonical_values
    )


def plaquette_curvature(field: GaugeField, path: OrientedGaugePath) -> PlaquetteResult:
    """Return exact curvature for a closed oriented plaquette path."""
    if isinstance(path, OrientedGaugePath) and not path.steps:
        _reject(
            "path",
            "lattice_gauge.plaquette.empty_boundary",
            "plaquette curvature requires a nonempty oriented face boundary",
        )
    result = path_holonomy(field, path)
    if result.start != result.end:
        _reject(
            "path",
            "lattice_gauge.plaquette.open_path",
            "a plaquette path must be closed",
        )
    return PlaquetteResult(
        field=field, path=path, curvature=result.holonomy, start=result.start
    )


def path_holonomy(field: GaugeField, path: OrientedGaugePath) -> HolonomyResult:
    """Compute the ordered exact group product along an oriented edge path.

    Under the published left-to-right convention
    ``Hol(gamma) = U_{e_1} ... U_{e_m}`` where a backward step resolves to
    the exact inverse of its forward label. Endpoint chaining and the ordered
    product are established as the contribution ledger is built.
    """

    _admit_holonomy(field, path)
    by_edge = {edge.edge_id: edge for edge in field.lattice.edges}
    labels = {label.edge_id: label.label for label in field.edge_labels}
    degree = field.degree
    if not path.steps:
        basepoint = cast(str, path.basepoint)
        return HolonomyResult._from_kernel(
            field=field,
            path=path,
            holonomy=PermutationLabel(degree=degree, image=tuple(range(degree))),
            contributions=(),
            start=basepoint,
            end=basepoint,
        )
    contributions: list[EdgeContribution] = []
    cursor: str | None = None
    start: str | None = None
    for step in path.steps:
        edge = by_edge[step.edge_id]
        image = tuple(labels[step.edge_id].image)
        value = image if step.forward else _inverse(image)
        contributions.append(
            EdgeContribution(
                edge_id=step.edge_id,
                forward=step.forward,
                value=PermutationLabel(degree=degree, image=value),
            )
        )
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if start is None:
            start = tail
        cursor = head
    if start is None or cursor is None:
        raise RuntimeError("admitted holonomy path produced no endpoints")
    product = tuple(range(degree))
    for contribution in contributions:
        product = _compose(product, tuple(contribution.value.image))
    return HolonomyResult._from_kernel(
        field=field,
        path=path,
        holonomy=PermutationLabel(degree=degree, image=product),
        contributions=tuple(contributions),
        start=start,
        end=cursor,
    )


def _admit_loop_family_size(
    loops: tuple[OrientedGaugePath, ...] | list[OrientedGaugePath],
) -> int:
    if type(loops) not in (tuple, list):
        _reject(
            "loops",
            "lattice_gauge.loop_family.loop_collection",
            "loop family must be a finite tuple or list of oriented paths",
        )
    if len(loops) > MAX_GAUGE_LOOP_FAMILY_SIZE:
        raise OperationResourceAdmissionError(
            location=("loops",),
            code="lattice_gauge.loop_family.count_over_envelope",
            message="loop family exceeds the 128-loop envelope",
        )
    total_steps = 0
    for path in loops:
        if not isinstance(path, OrientedGaugePath):
            _reject(
                "loops",
                "lattice_gauge.loop_family.path_shape",
                "every family member must be an oriented gauge path",
            )
        steps = getattr(path, "steps", _MISSING)
        if not isinstance(steps, tuple):
            _reject(
                "loops",
                "lattice_gauge.loop_family.path_shape",
                "loop path steps must be a tuple",
            )
        if len(steps) > MAX_GAUGE_PATH_LENGTH:
            raise OperationResourceAdmissionError(
                location=("loops",),
                code="lattice_gauge.loop_family.path_length_over_envelope",
                message="a loop exceeds the 256-step path envelope",
            )
        total_steps += len(steps)
        if total_steps > MAX_GAUGE_LOOP_FAMILY_STEPS:
            raise OperationResourceAdmissionError(
                location=("loops",),
                code="lattice_gauge.loop_family.steps_over_envelope",
                message="aggregate loop-family paths exceed 4096 steps",
            )
    return total_steps


def _admit_closed_loop(
    path: OrientedGaugePath, lattice: GaugeLattice, by_edge: dict[str, GaugeEdge]
) -> str:
    first: str | None = None
    cursor: str | None = None
    for step in path.steps:
        edge = by_edge.get(step.edge_id)
        if edge is None:
            _reject(
                "loops",
                "lattice_gauge.loop_family.unknown_edge",
                "loop paths must use edges of the source field",
            )
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and cursor != tail:
            _reject(
                "loops",
                "lattice_gauge.loop_family.disconnected_path",
                "each loop must chain head-to-tail",
            )
        if first is None:
            first = tail
        cursor = head
    if first is None:
        basepoint = getattr(path, "basepoint", _MISSING)
        if not _gauge_label_is_valid(basepoint) or basepoint not in lattice.vertices:
            _reject(
                "loops",
                "lattice_gauge.loop_family.empty_path_basepoint",
                "an empty loop must name a source-lattice basepoint",
            )
        return basepoint
    if cursor != first:
        _reject(
            "loops",
            "lattice_gauge.loop_family.open_path",
            "every member of a loop family must be closed",
        )
    basepoint = getattr(path, "basepoint", None)
    if basepoint is not None and basepoint != first:
        _reject(
            "loops",
            "lattice_gauge.loop_family.basepoint_mismatch",
            "a supplied loop basepoint must equal its first vertex",
        )
    return first


def _admit_loop_family_path_shapes(
    loops: tuple[OrientedGaugePath, ...] | list[OrientedGaugePath],
) -> int:
    """Check bounded scalar shapes and count text work before path replay."""

    text_units = 0
    for path in loops:
        for step in path.steps:
            if not isinstance(step, GaugePathStep):
                _reject(
                    "loops",
                    "lattice_gauge.loop_family.step_shape",
                    "every loop step must be a typed lattice traversal",
                )
            edge_id = getattr(step, "edge_id", _MISSING)
            forward = getattr(step, "forward", _MISSING)
            if not _gauge_label_is_valid(edge_id) or type(forward) is not bool:
                _reject(
                    "loops",
                    "lattice_gauge.loop_family.step_shape",
                    "loop steps require a strict edge label and boolean orientation",
                )
            text_units += len(cast(str, edge_id))
        basepoint = getattr(path, "basepoint", None)
        if basepoint is not None:
            if not _gauge_label_is_valid(basepoint):
                _reject(
                    "loops",
                    "lattice_gauge.loop_family.basepoint_shape",
                    "a supplied loop basepoint must be a valid lattice label",
                )
            text_units += len(cast(str, basepoint))
    return text_units


def _loop_family_output_units(
    field: GaugeField,
    loops: tuple[tuple[OrientedGaugePath, str], ...],
) -> int:
    degree = field.degree
    units = 32
    units += sum(len(vertex) + 4 for vertex in field.lattice.vertices)
    units += sum(
        len(edge.edge_id) + len(edge.tail) + len(edge.head) + 8
        for edge in field.lattice.edges
    )
    units += sum(len(entry.edge_id) + degree + 4 for entry in field.edge_labels)
    units += sum(
        len(basepoint) + degree + 12 + sum(len(step.edge_id) + 4 for step in path.steps)
        for path, basepoint in loops
    )
    return units


def loop_family_holonomies(
    field: GaugeField,
    loops: tuple[OrientedGaugePath, ...] | list[OrientedGaugePath],
) -> GaugeLoopFamilyHolonomies:
    """Compute the exact holonomy of each supplied closed path.

    The field is admitted once and retained once in the result.  Each path is
    an explicit based loop; this operation does not enumerate loops or search
    for a generating family.
    """

    total_steps = _admit_loop_family_size(loops)
    lattice, labels = _admit_transform_field(field)
    degree = field.degree
    path_text_units = _admit_loop_family_path_shapes(loops)
    work_units = len(lattice.vertices) + len(lattice.edges) * degree * degree
    work_units += len(loops)
    work_units += total_steps * (2 * degree + MAX_GAUGE_LABEL_LENGTH + 1)
    work_units += len(lattice.edges) * MAX_GAUGE_LABEL_LENGTH * 8
    work_units += sum(len(vertex) for vertex in lattice.vertices)
    work_units += sum(
        len(edge.edge_id) + len(edge.tail) + len(edge.head) for edge in lattice.edges
    )
    work_units += sum(len(entry.edge_id) for entry in field.edge_labels)
    work_units += path_text_units
    if work_units > MAX_GAUGE_LOOP_FAMILY_WORK:
        raise OperationResourceAdmissionError(
            location=("loops",),
            code="lattice_gauge.loop_family.work_over_envelope",
            message="loop-family admission work exceeds its exact envelope",
        )

    by_edge = {edge.edge_id: edge for edge in lattice.edges}
    admitted = tuple(
        (path, _admit_closed_loop(path, lattice, by_edge)) for path in loops
    )
    if _loop_family_output_units(field, admitted) > MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS:
        raise OperationResourceAdmissionError(
            location=("loops",),
            code="lattice_gauge.loop_family.output_over_envelope",
            message="source-bound loop results exceed the exact output envelope",
        )

    results: list[GaugeLoopHolonomy] = []
    for path, basepoint in admitted:
        product = tuple(range(degree))
        for step in path.steps:
            edge_value = labels[step.edge_id].image
            oriented_value = edge_value if step.forward else _inverse(edge_value)
            product = _compose(product, oriented_value)
        results.append(
            GaugeLoopHolonomy(
                path=path,
                basepoint=basepoint,
                holonomy=PermutationLabel(degree=degree, image=product),
            )
        )
    return GaugeLoopFamilyHolonomies._from_kernel(field=field, loops=tuple(results))


def _run_path_holonomy(request: object) -> HolonomyResult:
    from jacobian.math.gauge._models import HolonomyRequest as _Request

    if not isinstance(request, _Request):
        raise TypeError("path holonomy expects HolonomyRequest")
    return path_holonomy(request.field, request.path)


__all__ = [
    "gauge_transform",
    "loop_family_holonomies",
    "path_holonomy",
    "plaquette_curvature",
]
