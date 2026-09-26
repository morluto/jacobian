"""Finite multiplication-table gauge fields and ordered path holonomy."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    FiniteGroupGaugeBasepointTransportRequest,
    FiniteGroupGaugeBasepointTransportResult,
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeContribution,
    FiniteGroupGaugeCurvatureRequest,
    FiniteGroupGaugeCurvatureResult,
    FiniteGroupGaugeFaceCurvature,
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyRequest,
    FiniteGroupGaugeHolonomyResult,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
)
from jacobian.math.groups._table_models import FiniteGroupTable, FiniteGroupTableElement

MAX_FINITE_GROUP_GAUGE_WORK = 100_000
MAX_FINITE_GROUP_GAUGE_OUTPUT_UNITS = 100_000


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _is_gauge_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 64
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _admit_group(
    group: FiniteGroupTable,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int, int]:
    if not isinstance(group, FiniteGroupTable) or not all(
        name in group.__dict__ for name in ("multiplication", "inverse", "identity")
    ):
        _reject(
            "field",
            "lattice_gauge.finite_group.table_shape",
            "finite group table is malformed",
        )
    table, inverse, identity = group.multiplication, group.inverse, group.identity
    order = len(table) if isinstance(table, tuple) else 0
    if not 1 <= order <= 24 or not isinstance(inverse, tuple) or len(inverse) != order:
        _reject(
            "field",
            "lattice_gauge.finite_group.table_shape",
            "finite group table is malformed",
        )
    if any(not isinstance(row, tuple) or len(row) != order for row in table):
        _reject(
            "field",
            "lattice_gauge.finite_group.table_shape",
            "finite group table must be square",
        )
    if type(identity) is not int or not 0 <= identity < order:
        _reject(
            "field",
            "lattice_gauge.finite_group.identity",
            "group identity index is malformed",
        )
    if any(
        type(x) is not int or not 0 <= x < order for row in table for x in row
    ) or any(type(x) is not int or not 0 <= x < order for x in inverse):
        _reject(
            "field",
            "lattice_gauge.finite_group.table_index",
            "table entries and inverses must index the group",
        )
    if any(
        table[i][inverse[i]] != identity or table[inverse[i]][i] != identity
        for i in range(order)
    ):
        _reject(
            "field",
            "lattice_gauge.finite_group.inverse_law",
            "inverse map must give two-sided inverses",
        )
    if any(table[identity][i] != i or table[i][identity] != i for i in range(order)):
        _reject(
            "field",
            "lattice_gauge.finite_group.identity_law",
            "identity index must be two-sided",
        )
    if order**3 > MAX_FINITE_GROUP_GAUGE_WORK:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="lattice_gauge.finite_group.work_bound",
            message="group-law admission exceeds the work bound",
        )
    for a in range(order):
        for b in range(order):
            for c in range(order):
                if table[table[a][b]][c] != table[a][table[b][c]]:
                    _reject(
                        "field",
                        "lattice_gauge.finite_group.associativity",
                        "multiplication table must be associative",
                    )
    return table, inverse, identity, order


def _admit_field(
    field: FiniteGroupGaugeField, group: FiniteGroupTable, order: int
) -> tuple[tuple[str, ...], tuple[GaugeEdge, ...], dict[str, int]]:
    lattice = field.lattice
    vertices, edges, values = lattice.vertices, lattice.edges, field.edge_values
    if (
        not isinstance(vertices, tuple)
        or not vertices
        or len(vertices) > 64
        or any(not _is_gauge_label(vertex) for vertex in vertices)
        or len(set(vertices)) != len(vertices)
    ):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "lattice vertices are malformed",
        )
    if (
        not isinstance(edges, tuple)
        or not 1 <= len(edges) <= 128
        or any(not isinstance(e, GaugeEdge) for e in edges)
    ):
        _reject(
            "field", "lattice_gauge.finite_group.lattice", "lattice edges are malformed"
        )
    edge_ids = tuple(e.edge_id for e in edges)
    if any(
        not _is_gauge_label(edge.edge_id)
        or not _is_gauge_label(edge.tail)
        or not _is_gauge_label(edge.head)
        for edge in edges
    ):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "edge identifiers and endpoints must be bounded labels",
        )
    if tuple(sorted(edge_ids)) != edge_ids or len(set(edge_ids)) != len(edge_ids):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "lattice edge identifiers must be unique and ordered",
        )
    vertex_set = set(vertices)
    if any(e.tail not in vertex_set or e.head not in vertex_set for e in edges):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "edge endpoints must belong to the lattice",
        )
    if not isinstance(values, tuple) or len(values) != len(edges):
        _reject(
            "field",
            "lattice_gauge.finite_group.edge_values",
            "field must label every lattice edge exactly once",
        )
    value_by_id = {}
    for item in values:
        if (
            not hasattr(item, "edge_id")
            or not hasattr(item, "value")
            or not isinstance(item.value, FiniteGroupTableElement)
        ):
            _reject(
                "field",
                "lattice_gauge.finite_group.edge_value",
                "edge labels must be typed table indices",
            )
        if not _is_gauge_label(item.edge_id):
            _reject(
                "field",
                "lattice_gauge.finite_group.edge_value",
                "edge identifier must be a bounded label",
            )
        if item.value.group != group:
            _reject(
                "field",
                "lattice_gauge.finite_group.edge_value_parent",
                "edge element must use the field's exact group table",
            )
        if type(item.value.index) is not int or not 0 <= item.value.index < order:
            _reject(
                "field",
                "lattice_gauge.finite_group.edge_value",
                "edge value must index the bound group",
            )
        value_by_id[item.edge_id] = item.value.index
    if set(value_by_id) != set(edge_ids) or len(value_by_id) != len(values):
        _reject(
            "field",
            "lattice_gauge.finite_group.edge_values",
            "field must label every lattice edge exactly once",
        )
    return vertices, edges, value_by_id


def _resolve_path(
    path: OrientedGaugePath,
    vertices: tuple[str, ...],
    edges: tuple[GaugeEdge, ...],
    values: dict[str, int],
    inverse: tuple[int, ...],
) -> tuple[str, str, tuple[tuple[str, bool, int], ...]]:
    steps = path.steps
    if not isinstance(steps, tuple) or len(steps) > 256:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="lattice_gauge.finite_group.path_bound",
            message="path exceeds the 256-step bound",
        )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    if not steps:
        basepoint = path.basepoint
        if not _is_gauge_label(basepoint) or basepoint not in set(vertices):
            _reject(
                "path",
                "lattice_gauge.finite_group.empty_path_basepoint",
                "empty path basepoint must name a lattice vertex",
            )
        return basepoint, basepoint, ()
    cursor = start = None
    resolved = []
    for step in steps:
        if (
            not isinstance(step, GaugePathStep)
            or type(step.forward) is not bool
            or not _is_gauge_label(step.edge_id)
            or step.edge_id not in edge_by_id
        ):
            _reject(
                "path",
                "lattice_gauge.finite_group.path_step",
                "path step must name a lattice edge and orientation",
            )
        edge = edge_by_id[step.edge_id]
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and cursor != tail:
            _reject(
                "path",
                "lattice_gauge.finite_group.path_chain",
                "oriented path steps must chain head-to-tail",
            )
        if start is None:
            start = tail
        cursor = head
        index = values[step.edge_id]
        resolved.append(
            (step.edge_id, step.forward, index if step.forward else inverse[index])
        )
    if start is None or cursor is None:
        raise RuntimeError("admitted nonempty path produced no endpoints")
    return start, cursor, tuple(resolved)


def finite_group_gauge_holonomy(
    request: FiniteGroupGaugeHolonomyRequest,
) -> FiniteGroupGaugeHolonomyResult:
    """Compute left-to-right path product in the field's exact table parent."""
    if not isinstance(request, FiniteGroupGaugeHolonomyRequest):
        _reject(
            "request",
            "lattice_gauge.finite_group.request_type",
            "expected a finite-group gauge holonomy request",
        )
    field, path = request.field, request.path
    if not isinstance(field, FiniteGroupGaugeField) or not isinstance(
        path, OrientedGaugePath
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.request_shape",
            "field and path must be typed values",
        )
    group = field.group
    if not isinstance(group, FiniteGroupTable):
        _reject(
            "field",
            "lattice_gauge.finite_group.parent",
            "field must be bound to a finite group table",
        )
    table, inverse, identity, order = _admit_group(group)
    vertices, edges, values = _admit_field(field, group, order)
    steps = path.steps
    if not isinstance(steps, tuple) or len(steps) > 256:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="lattice_gauge.finite_group.path_bound",
            message="path exceeds the 256-step bound",
        )
    work = order**3 + len(edges) + len(steps)
    # The result retains the field (which nests one copy of the parent table
    # per edge value), one parent-bound element per path contribution, and a
    # parent-bound holonomy. Count every serialized multiplication table before
    # constructing the contribution ledger.
    parent_copies = len(edges) + len(steps) + 2
    output_units = parent_copies * order**2 + len(edges) + len(vertices) + len(steps)
    if work > MAX_FINITE_GROUP_GAUGE_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="lattice_gauge.finite_group.curvature_work_bound",
            message="finite-group curvature work exceeds the admitted envelope",
        )
    if output_units > MAX_FINITE_GROUP_GAUGE_OUTPUT_UNITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="lattice_gauge.finite_group.output_bound",
            message="finite-group holonomy result exceeds the admitted output bound",
        )
    start, end, resolved = _resolve_path(path, vertices, edges, values, inverse)
    product = identity
    contributions = []
    for edge_id, forward, index in resolved:
        product = table[product][index]
        contributions.append(
            FiniteGroupGaugeContribution(
                edge_id=edge_id,
                forward=forward,
                value=FiniteGroupTableElement(group=group, index=index),
            )
        )
    return FiniteGroupGaugeHolonomyResult.model_construct(
        field=field,
        path=path,
        holonomy=FiniteGroupTableElement(group=group, index=product),
        contributions=tuple(contributions),
        start=start,
        end=end,
    )


def finite_group_gauge_basepoint_transport(  # noqa: C901
    request: FiniteGroupGaugeBasepointTransportRequest,
) -> FiniteGroupGaugeBasepointTransportResult:
    r"""Transport a based loop by conjugating with a connector holonomy.

    For a connector ``gamma`` from ``s`` to ``t`` and loop ``ell`` based at
    ``s``, the returned loop is ``reverse(gamma) * ell * gamma``. Its
    holonomy is therefore ``Hol(gamma)^-1 Hol(ell) Hol(gamma)``.
    """
    if not isinstance(request, FiniteGroupGaugeBasepointTransportRequest):
        _reject(
            "request",
            "lattice_gauge.finite_group.basepoint_request_type",
            "expected a finite-group basepoint transport request",
        )
    if not all(name in request.__dict__ for name in ("field", "loop", "connector")):
        _reject(
            "request",
            "lattice_gauge.finite_group.basepoint_request_shape",
            "request is missing required fields",
        )
    field, loop, connector = request.field, request.loop, request.connector
    if (
        not isinstance(field, FiniteGroupGaugeField)
        or not isinstance(loop, OrientedGaugePath)
        or not isinstance(connector, OrientedGaugePath)
        or not all(
            name in field.__dict__ for name in ("lattice", "group", "edge_values")
        )
        or not isinstance(field.group, FiniteGroupTable)
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.basepoint_request_shape",
            "field, based loop, and connector must be typed finite-group values",
        )
    if not isinstance(field.lattice, GaugeLattice):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "field lattice is malformed",
        )
    if (
        not all(name in field.lattice.__dict__ for name in ("vertices", "edges"))
        or not isinstance(field.lattice.vertices, tuple)
        or not isinstance(field.lattice.edges, tuple)
        or not isinstance(field.edge_values, tuple)
    ):
        _reject(
            "field",
            "lattice_gauge.finite_group.lattice",
            "field lattice and edge labels are malformed",
        )
    group = field.group
    table, inverse, identity, order = _admit_group(group)
    vertices, edges, values = _admit_field(field, group, order)
    if (
        "steps" not in loop.__dict__
        or "basepoint" not in loop.__dict__
        or "steps" not in connector.__dict__
        or "basepoint" not in connector.__dict__
        or not isinstance(loop.steps, tuple)
        or not isinstance(connector.steps, tuple)
        or not isinstance(loop.basepoint, (str, type(None)))
        or not isinstance(connector.basepoint, (str, type(None)))
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.basepoint_request_shape",
            "loop and connector paths are malformed",
        )
    if len(loop.steps) > 256 or len(connector.steps) > 256:
        raise OperationResourceAdmissionError(
            location=("loop", "connector"),
            code="lattice_gauge.finite_group.basepoint_input_path_bound",
            message="source paths exceed the 256-step path bound",
        )
    if any(
        not isinstance(step, GaugePathStep)
        or not all(name in step.__dict__ for name in ("edge_id", "forward"))
        or not isinstance(step.edge_id, str)
        or not isinstance(step.forward, bool)
        for steps in (loop.steps, connector.steps)
        for step in steps
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.basepoint_request_shape",
            "loop and connector paths are malformed",
        )
    transported_length = 2 * len(connector.steps) + len(loop.steps)
    if transported_length > 256:
        raise OperationResourceAdmissionError(
            location=("connector", "loop"),
            code="lattice_gauge.finite_group.basepoint_path_bound",
            message="transported loop exceeds the 256-step path bound",
        )
    work = order**3 + len(edges) + len(loop.steps) + len(connector.steps) + 2
    output_units = (
        (len(edges) + 4) * order**2
        + 3 * len(connector.steps)
        + 2 * len(loop.steps)
        + len(edges)
        + len(vertices)
    )
    if work > MAX_FINITE_GROUP_GAUGE_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="lattice_gauge.finite_group.basepoint_work_bound",
            message="basepoint transport work exceeds its admitted envelope",
        )
    if output_units > MAX_FINITE_GROUP_GAUGE_OUTPUT_UNITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="lattice_gauge.finite_group.basepoint_output_bound",
            message="basepoint transport result exceeds its admitted output envelope",
        )
    loop_start, loop_end, loop_resolved = _resolve_path(
        loop, vertices, edges, values, inverse
    )
    connector_start, connector_end, connector_resolved = _resolve_path(
        connector, vertices, edges, values, inverse
    )
    for name, path, start in (
        ("loop", loop, loop_start),
        ("connector", connector, connector_start),
    ):
        if path.basepoint is not None and (
            not _is_gauge_label(path.basepoint) or path.basepoint != start
        ):
            _reject(
                name,
                "lattice_gauge.finite_group.basepoint_mismatch",
                "an authored path basepoint must equal its first vertex",
            )
    if loop_start != loop_end:
        _reject(
            "loop",
            "lattice_gauge.finite_group.loop_not_closed",
            "source path must be a loop before changing its basepoint",
        )
    if connector_start != loop_start:
        _reject(
            "connector",
            "lattice_gauge.finite_group.connector_start_mismatch",
            "connector must start at the source loop basepoint",
        )

    def product_of(resolved: tuple[tuple[str, bool, int], ...]) -> int:
        product = identity
        for _, _, element in resolved:
            product = table[product][element]
        return product

    source_holonomy = product_of(loop_resolved)
    connector_holonomy = product_of(connector_resolved)
    target_holonomy = table[table[inverse[connector_holonomy]][source_holonomy]][
        connector_holonomy
    ]
    reverse_steps = tuple(
        GaugePathStep(edge_id=step.edge_id, forward=not step.forward)
        for step in reversed(connector.steps)
    )
    transported_loop = OrientedGaugePath(
        steps=reverse_steps + loop.steps + connector.steps,
        basepoint=connector_end,
    )
    return FiniteGroupGaugeBasepointTransportResult.model_construct(
        field=field,
        loop=loop,
        connector=connector,
        transported_loop=transported_loop,
        source_basepoint=loop_start,
        target_basepoint=connector_end,
        source_holonomy=FiniteGroupTableElement(group=group, index=source_holonomy),
        connector_holonomy=FiniteGroupTableElement(
            group=group, index=connector_holonomy
        ),
        transported_holonomy=FiniteGroupTableElement(
            group=group, index=target_holonomy
        ),
    )


def finite_group_gauge_curvature(
    request: FiniteGroupGaugeCurvatureRequest,
) -> FiniteGroupGaugeCurvatureResult:
    """Return ordered face holonomies and whether every face is flat.

    Face curvature is the path-ordered product around each oriented attaching
    walk. Reversing the face therefore takes the group inverse, including for
    noncommutative groups. Flatness means every represented 2-cell has identity
    boundary product; it makes no claim about cells absent from the complex.
    """
    if not isinstance(request, FiniteGroupGaugeCurvatureRequest):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_type",
            "expected a finite-group gauge curvature request",
        )
    complex_value, field = request.complex, request.field
    if not isinstance(complex_value, FiniteGroupGaugeComplex) or not isinstance(
        field, FiniteGroupGaugeField
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_shape",
            "complex and field must be typed values",
        )
    if not all(
        name in complex_value.__dict__ for name in ("lattice", "group", "faces")
    ) or not all(
        name in field.__dict__ for name in ("lattice", "group", "edge_values")
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_malformed",
            "complex and field are missing required source fields",
        )
    source_group = getattr(complex_value, "group", None)
    field_group = getattr(field, "group", None)
    if any(
        not isinstance(group_value, FiniteGroupTable)
        or not all(
            name in group_value.__dict__
            for name in ("multiplication", "inverse", "identity")
        )
        for group_value in (source_group, field_group)
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_malformed",
            "complex and field must retain complete finite-group tables",
        )
    for lattice_value in (
        getattr(complex_value, "lattice", None),
        getattr(field, "lattice", None),
    ):
        if not isinstance(lattice_value, GaugeLattice) or not all(
            name in lattice_value.__dict__ for name in ("vertices", "edges")
        ):
            _reject(
                "request",
                "lattice_gauge.finite_group.curvature_request_malformed",
                "complex and field must retain complete lattices",
            )
    try:
        complex_value = FiniteGroupGaugeComplex.model_validate(
            complex_value.model_dump()
        )
        field = FiniteGroupGaugeField.model_validate(field.model_dump())
    except (TypeError, ValueError):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_malformed",
            "complex and field must satisfy their complete typed contracts",
        )
    # model_construct/copy can bypass required-field validation. Retrieve
    # potentially absent attributes without allowing AttributeError to escape.
    lattice = getattr(complex_value, "lattice", None)
    group = getattr(complex_value, "group", None)
    field_lattice = getattr(field, "lattice", None)
    field_group = getattr(field, "group", None)
    if (
        not isinstance(lattice, GaugeLattice)
        or not isinstance(field_lattice, GaugeLattice)
        or not isinstance(group, FiniteGroupTable)
        or not isinstance(field_group, FiniteGroupTable)
        or lattice != field_lattice
        or group != field_group
    ):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_parent_mismatch",
            "complex and field must share the exact lattice and group parent",
        )
    table, inverse, identity, order = _admit_group(group)
    vertices, edges, values = _admit_field(field, group, order)
    # The complex is caller-supplied; re-admit its faces at the consuming
    # boundary rather than trusting constructor provenance.
    from jacobian.math.gauge.finite_group_complex import _admit_faces, _admit_lattice

    _, _, vertex_set, edge_by_id, _ = _admit_lattice(lattice)
    faces = getattr(complex_value, "faces", None)
    if not isinstance(faces, tuple):
        _reject(
            "request",
            "lattice_gauge.finite_group.curvature_request_shape",
            "complex faces must be a validated tuple",
        )
    _admit_faces(faces, vertex_set, edge_by_id)
    total_steps = sum(len(face.boundary.steps) for face in faces)
    work = (
        order**3
        + (len(edges) + 2) * order**2
        + len(edges)
        + total_steps
        + len(complex_value.faces)
    )
    # Returned values retain both source parents, the field's edge-bound table
    # elements, and one table-bound curvature element per face.
    parent_copies = len(edges) + len(faces) + 3
    output_units = (
        parent_copies * order**2
        + len(vertices)
        + len(edges) * 4
        + total_steps * 2
        + len(faces) * 4
    )
    if (
        work > MAX_FINITE_GROUP_GAUGE_WORK
        or output_units > MAX_FINITE_GROUP_GAUGE_OUTPUT_UNITS
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="lattice_gauge.finite_group.curvature_output_bound",
            message="finite-group curvature result exceeds admitted work or output",
        )
    face_values = []
    edge_by_id = {edge.edge_id: edge for edge in edges}
    for face in faces:
        product = identity
        path = face.boundary
        if path.steps:
            for step in path.steps:
                index = values[step.edge_id]
                if not step.forward:
                    index = inverse[index]
                product = table[product][index]
        face_values.append(
            FiniteGroupGaugeFaceCurvature(
                face_id=face.face_id,
                value=FiniteGroupTableElement(group=group, index=product),
            )
        )
    return FiniteGroupGaugeCurvatureResult.model_construct(
        complex=complex_value,
        field=field,
        face_values=tuple(face_values),
        flat=all(value.value.index == identity for value in face_values),
    )
