"""Native exact permutation-valued lattice-gauge path holonomy."""

from __future__ import annotations

from typing import NoReturn

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_VERTICES,
    EdgeContribution,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeTransformRequest,
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

    if not isinstance(field, GaugeField):
        _reject(
            "field",
            "lattice_gauge.holonomy.field_not_a_gauge_field",
            "holonomy source must be an exact permutation-valued gauge field",
        )
    if not isinstance(path, OrientedGaugePath):
        _reject(
            "path",
            "lattice_gauge.holonomy.path_not_a_gauge_path",
            "holonomy path must be an oriented lattice edge path",
        )
    if not path.steps:
        _reject(
            "path",
            "lattice_gauge.holonomy.empty_path",
            "holonomy path must contain at least one oriented edge step",
        )
    if len(path.steps) > 256:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="lattice_gauge.holonomy.path_over_envelope",
            message="oriented path exceeds the 256-step envelope",
        )
    by_id = {edge.edge_id: edge for edge in field.lattice.edges}
    labels = {label.edge_id: label.label for label in field.edge_labels}
    cursor: str | None = None
    for step in path.steps:
        edge = by_id.get(step.edge_id)
        if edge is None or step.edge_id not in labels:
            _reject(
                "path",
                "lattice_gauge.holonomy.unknown_edge_step",
                f"path step references unknown lattice edge {step.edge_id!r}",
            )
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and tail != cursor:
            _reject(
                "path",
                "lattice_gauge.holonomy.disconnected_path",
                "oriented path steps must chain head-to-tail",
            )
        cursor = head


def _reparse_field(field: GaugeField) -> GaugeField:
    """Re-establish the complete field boundary for direct native callers."""

    try:
        return GaugeField.model_validate(field.model_dump(mode="python"))
    except Exception:
        _reject(
            "field",
            "lattice_gauge.transform.field_malformed",
            "transform source must be a complete bounded gauge field",
        )


def _admit_transform(
    field: object, vertex_values: object
) -> tuple[GaugeField, tuple[GaugeVertexValue, ...]]:
    """Shared strict admission for native and catalog gauge transforms."""

    if not isinstance(field, GaugeField):
        _reject(
            "field",
            "lattice_gauge.transform.field_not_a_gauge_field",
            "transform source must be a gauge field",
        )
    canonical_field = _reparse_field(field)
    if not isinstance(vertex_values, (tuple, list)):
        _reject(
            "vertex_values",
            "lattice_gauge.transform.vertex_values_type",
            "vertex values must be a finite labelled family",
        )
    if not vertex_values or len(vertex_values) > MAX_GAUGE_VERTICES:
        _reject(
            "vertex_values",
            "lattice_gauge.transform.vertex_values_shape",
            "vertex values must be a bounded nonempty labelled family",
        )
    canonical_values: list[GaugeVertexValue] = []
    for entry in vertex_values:
        if not isinstance(entry, GaugeVertexValue):
            _reject(
                "vertex_values",
                "lattice_gauge.transform.vertex_value_type",
                "every frame entry must be a gauge vertex value",
            )
        # Check scalar/hashable identity before serializing a forged
        # model_construct value; otherwise Pydantic itself may emit a warning
        # or a raw unhashable-value failure before owner admission runs.
        if type(getattr(entry, "vertex", None)) is not str:
            _reject(
                "vertex_values",
                "lattice_gauge.transform.vertex_value_malformed",
                "vertex identifiers must be bounded strings",
            )
        try:
            canonical_values.append(
                GaugeVertexValue.model_validate(entry.model_dump(mode="python"))
            )
        except Exception:
            _reject(
                "vertex_values",
                "lattice_gauge.transform.vertex_value_malformed",
                "every frame entry must carry a bounded permutation label",
            )
    values = tuple(canonical_values)
    # Reuse the request model's complete coverage and degree rules after the
    # native values have been reparsed.  This is structural validation only;
    # the operation's actual admission remains here.
    try:
        request = GaugeTransformRequest.model_validate(
            {"field": canonical_field, "vertex_values": values}
        )
    except (ValidationError, TypeError, ValueError):
        _reject(
            "vertex_values",
            "lattice_gauge.transform.vertex_coverage",
            "transform must label every lattice vertex exactly once with the field degree",
        )
    return request.field, request.vertex_values


def gauge_transform(
    field: GaugeField,
    vertex_values: tuple[GaugeVertexValue, ...] | list[GaugeVertexValue],
) -> GaugeTransformResult:
    """Apply ``U'_e = h_tail^-1 U_e h_head`` on one finite lattice."""
    field, vertex_values = _admit_transform(field, vertex_values)
    by_vertex = {entry.vertex: entry.value for entry in vertex_values}
    labels = {entry.edge_id: entry.label for entry in field.edge_labels}
    transformed_labels: list[GaugeFieldEdgeLabel] = []
    for edge in field.lattice.edges:
        value = _compose(
            _compose(_inverse(by_vertex[edge.tail].image), labels[edge.edge_id].image),
            by_vertex[edge.head].image,
        )
        transformed_labels.append(
            GaugeFieldEdgeLabel(
                edge_id=edge.edge_id,
                label=PermutationLabel(degree=field.degree, image=value),
            )
        )
    transformed = GaugeField(
        lattice=field.lattice,
        degree=field.degree,
        edge_labels=tuple(transformed_labels),
    )
    canonical_values = tuple(
        GaugeVertexValue(vertex=vertex, value=by_vertex[vertex])
        for vertex in field.lattice.vertices
    )
    return GaugeTransformResult(
        source=field, transformed=transformed, vertex_values=canonical_values
    )


def plaquette_curvature(field: GaugeField, path: OrientedGaugePath) -> PlaquetteResult:
    """Return exact curvature for a closed oriented plaquette path."""
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
    the exact inverse of its forward label. Before return the kernel replays
    endpoint chaining and the full ordered product against the contribution
    ledger.
    """

    _admit_holonomy(field, path)
    by_edge = {edge.edge_id: edge for edge in field.lattice.edges}
    labels = {label.edge_id: label.label for label in field.edge_labels}
    degree = field.degree
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
    # Replay the reverse-traversal identity: the reversed path (reversed
    # step order, flipped orientations) must compose to the exact inverse,
    # so the round trip is the identity permutation.
    reversed_product = tuple(range(degree))
    for contribution in reversed(contributions):
        reversed_product = _compose(
            reversed_product, _inverse(tuple(contribution.value.image))
        )
    if _compose(product, reversed_product) != tuple(range(degree)):
        _reject(
            "field",
            "lattice_gauge.holonomy.reverse_replay_failed",
            "reverse traversal must compose to the exact holonomy inverse",
        )
    return HolonomyResult._from_kernel(
        holonomy=PermutationLabel(degree=degree, image=product),
        contributions=tuple(contributions),
        start=start,
        end=cursor,
    )


def _run_path_holonomy(request: object) -> HolonomyResult:
    from jacobian.math.gauge._models import HolonomyRequest as _Request

    if not isinstance(request, _Request):
        raise TypeError("path holonomy expects HolonomyRequest")
    return path_holonomy(request.field, request.path)


__all__ = ["gauge_transform", "path_holonomy", "plaquette_curvature"]
