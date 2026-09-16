"""Native exact permutation-valued lattice-gauge path holonomy."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    EdgeContribution,
    GaugeField,
    HolonomyResult,
    OrientedGaugePath,
    PermutationLabel,
)


def _reject(location: str, code: str, message: str) -> None:
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
    assert isinstance(field, GaugeField)
    assert isinstance(path, OrientedGaugePath)
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
        assert edge is not None
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and tail != cursor:
            _reject(
                "path",
                "lattice_gauge.holonomy.disconnected_path",
                "oriented path steps must chain head-to-tail",
            )
        cursor = head


def path_holonomy(field: GaugeField, path: OrientedGaugePath) -> HolonomyResult:
    """Compute the ordered exact group product along an oriented edge path.

    Under the published left-to-right convention
    ``Hol(gamma) = U_{e_1} ... U_{e_m}`` where a backward step resolves to
    the exact inverse of its forward label. Before return the kernel replays
    endpoint chaining and the full ordered product against the contribution
    ledger.
    """

    _admit_holonomy(field, path)
    assert isinstance(field, GaugeField)
    assert isinstance(path, OrientedGaugePath)
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
    assert start is not None and cursor is not None
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

    assert isinstance(request, _Request)
    return path_holonomy(request.field, request.path)


__all__ = ["path_holonomy"]
