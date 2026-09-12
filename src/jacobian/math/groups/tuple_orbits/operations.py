"""Exact diagonal permutation action on explicit tuple families."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.actions._models import (
    MAX_FAMILY_MEMBERS,
    MAX_GROUP_ORDER,
    FinitePermutationAction,
)
from jacobian.math.groups.operations import _backend_group, _full_permutation_form
from jacobian.math.groups.tuple_orbits._models import (
    MAX_TUPLE_ARITY,
    TupleFamilyOrbitResult,
    TupleFamilyOrbitSource,
    TupleOrbitRow,
)

# These are mathematical execution-envelope bounds, not transport-byte caps.
MAX_TUPLE_ORBIT_GROUP_ORDER = MAX_GROUP_ORDER
MAX_TUPLE_ORBIT_GROUP_WORK = 2_000_000
MAX_TUPLE_ORBIT_ACTIONS = 1_000_000
MAX_TUPLE_ORBIT_IMAGES = 200_000
MAX_TUPLE_ORBIT_TRANSPORTERS = 100_000
MAX_TUPLE_ORBIT_RESULT_CELLS = 2_000_000


def _backend_action(action: FinitePermutationAction) -> Any:
    """Adapt Jacobian's labelled action to the maintained SymPy backend."""

    return _backend_group(
        PermutationGroup(degree=len(action.domain), generators=action.generators)
    )


def _admit_source(
    request: TupleFamilyOrbitSource,
) -> tuple[Any | None, int, int, TupleFamilyOrbitSource]:
    request_checkpoint("before tuple-family orbit admission")
    if not isinstance(request, TupleFamilyOrbitSource):
        raise OperationDomainValidationError(
            location=("request",),
            code="finite_group_action.tuple_family_request_type",
            message="tuple-family operation requires TupleFamilyOrbitSource",
        )
    action = request.action
    if not isinstance(action, FinitePermutationAction):
        raise OperationDomainValidationError(
            location=("action",),
            code="finite_group_action.tuple_family_action_type",
            message="tuple-family source must retain a finite permutation action",
        )
    try:
        action = FinitePermutationAction.model_validate(
            {"domain": action.domain, "generators": action.generators}
        )
    except ValidationError as error:
        detail = error.errors()[0]
        raise OperationDomainValidationError(
            location=("action", *tuple(detail.get("loc", ()))),
            code=str(detail["type"]),
            message=str(detail["msg"]),
        ) from error
    if (
        not isinstance(request.arity, int)
        or isinstance(request.arity, bool)
        or not 0 <= request.arity <= MAX_TUPLE_ARITY
    ):
        raise OperationDomainValidationError(
            location=("arity",),
            code="finite_group_action.tuple_family_arity_out_of_range",
            message="tuple arity must be a non-negative action-domain-sized integer",
        )
    if not isinstance(request.family, tuple) or any(
        not isinstance(member, tuple) for member in request.family
    ):
        raise OperationDomainValidationError(
            location=("family",),
            code="finite_group_action.tuple_family_shape",
            message="tuple families must use immutable tuple rows",
        )
    family_size = len(request.family)
    if family_size > MAX_FAMILY_MEMBERS:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_input_bound",
            message=f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
        )
    # Request-model validators are not invoked for native callers that use a
    # forged model or a plain object. Recheck only cheap structural facts here;
    # no group enumeration is repeated by result construction.
    degree = len(action.domain)
    if any(len(member) != request.arity for member in request.family):
        raise OperationDomainValidationError(
            location=("family",),
            code="finite_group_action.tuple_family_arity_mismatch",
            message="every family member must have the declared arity",
        )
    if any(
        not isinstance(coordinate, int)
        or isinstance(coordinate, bool)
        or not 0 <= coordinate < degree
        for member in request.family
        for coordinate in member
    ):
        raise OperationDomainValidationError(
            location=("family",),
            code="finite_group_action.tuple_family_coordinate_out_of_range",
            message="tuple coordinates must lie on the action domain axis",
        )
    admitted = TupleFamilyOrbitSource(
        action=action,
        arity=request.arity,
        family=request.family,
    )
    if family_size == 0:
        return None, 0, degree, admitted
    backend = _backend_action(action)
    group_order = int(backend.order())
    request_checkpoint("after tuple-family group-order admission")
    if group_order > MAX_TUPLE_ORBIT_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("action",),
            code="finite_group_action.tuple_family_group_order_bound",
            message=(
                "generated group order exceeds the admitted tuple-orbit "
                f"maximum {MAX_TUPLE_ORBIT_GROUP_ORDER}"
            ),
        )
    group_work = group_order * degree
    if group_work > MAX_TUPLE_ORBIT_GROUP_WORK:
        raise OperationResourceAdmissionError(
            location=("action",),
            code="finite_group_action.tuple_family_group_work_bound",
            message="group element materialization exceeds the admitted work bound",
        )
    unique_family_size = len(set(request.family))
    image_work = group_order * unique_family_size * (max(1, request.arity) + 1)
    if image_work > MAX_TUPLE_ORBIT_ACTIONS:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_action_work_bound",
            message="diagonal tuple-action work exceeds the admitted bound",
        )
    image_cells = group_order * unique_family_size
    if image_cells > MAX_TUPLE_ORBIT_IMAGES:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_image_bound",
            message="ambient tuple-image intermediates exceed the admitted bound",
        )
    transporter_work = group_order * unique_family_size
    if transporter_work > MAX_TUPLE_ORBIT_TRANSPORTERS:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_transporter_bound",
            message="transporter enumeration exceeds the admitted bound",
        )
    # Every output row retains a representative, source-index references, and
    # one full-axis transporter; this upper bound is independent of |X|^arity.
    output_upper = unique_family_size * (request.arity + degree + 4) + family_size
    if output_upper > MAX_TUPLE_ORBIT_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_result_bound",
            message="tuple orbit-profile result cells exceed the admitted bound",
        )
    return backend, group_order, degree, admitted


def tuple_family_orbit_profile(
    request: TupleFamilyOrbitSource,
) -> TupleFamilyOrbitResult:
    """Partition source-indexed tuples by the ambient diagonal action.

    For each represented ambient orbit, the operation computes the least tuple
    representative, all source indices in that orbit, the full orbit and
    stabilizer sizes, and the least transporter from the first source row to
    that representative. Tuple coordinate order and repeated coordinates are
    preserved exactly.
    """

    backend, group_order, degree, request = _admit_source(request)
    if backend is None:
        return TupleFamilyOrbitResult(
            source=request,
            rows=(),
            is_union_of_complete_ambient_orbits=True,
        )
    request_checkpoint("before tuple-family group materialization")
    elements = tuple(
        sorted(_full_permutation_form(element, degree) for element in backend.elements)
    )
    request_checkpoint("after tuple-family group materialization")
    source_positions = tuple(tuple(member) for member in request.family)
    source_indices_by_tuple: dict[tuple[int, ...], list[int]] = {}
    for index, member in enumerate(source_positions):
        source_indices_by_tuple.setdefault(member, []).append(index)
    unclassified = set(source_positions)
    rows: list[TupleOrbitRow] = []
    ambient_orbits: list[set[tuple[int, ...]]] = []
    generated_images = 0
    generated_transporters = 0

    while unclassified:
        request_checkpoint("during tuple-family orbit partition")
        seed = min(unclassified)
        images: dict[tuple[int, ...], tuple[int, ...]] = {}
        for element in elements:
            request_checkpoint("during tuple-family image enumeration")
            image = tuple(element[value] for value in seed)
            generated_images += 1
            if generated_images > MAX_TUPLE_ORBIT_IMAGES:
                raise OperationResourceAdmissionError(
                    location=("family",),
                    code="finite_group_action.tuple_family_image_bound",
                    message="ambient tuple-image intermediates exceed the admitted bound",
                )
            previous = images.get(image)
            if previous is None or element < previous:
                images[image] = element
        ambient_images = set(images)
        ambient_orbits.append(ambient_images)
        representative = min(ambient_images)
        source_indices = tuple(
            sorted(
                index
                for image in ambient_images
                for index in source_indices_by_tuple.get(image, ())
            )
        )
        if not source_indices:
            raise RuntimeError("orbit generation lost every source tuple")
        first_source = source_positions[source_indices[0]]
        transporter: tuple[int, ...] | None = None
        for element in elements:
            request_checkpoint("during tuple-family transporter enumeration")
            generated_transporters += 1
            if generated_transporters > MAX_TUPLE_ORBIT_TRANSPORTERS:
                raise OperationResourceAdmissionError(
                    location=("family",),
                    code="finite_group_action.tuple_family_transporter_bound",
                    message="transporter enumeration exceeds the admitted bound",
                )
            if tuple(element[value] for value in first_source) == representative and (
                transporter is None or element < transporter
            ):
                transporter = element
        if transporter is None:
            raise RuntimeError(
                "orbit generation lost a source-to-representative transporter"
            )
        rows.append(
            TupleOrbitRow(
                representative=representative,
                source_indices=source_indices,
                orbit_size=len(ambient_images),
                stabilizer_size=group_order // len(ambient_images),
                least_transporter=transporter,
            )
        )
        unclassified.difference_update(ambient_images)

    rows = sorted(rows, key=lambda row: row.representative)
    source_counts = Counter(source_positions)
    complete = all(
        source_counts[image] == 1
        for ambient_images in ambient_orbits
        for image in ambient_images
    )
    result_cells = sum(
        len(row.representative)
        + len(row.source_indices)
        + len(row.least_transporter)
        + 4
        for row in rows
    )
    if result_cells > MAX_TUPLE_ORBIT_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="finite_group_action.tuple_family_result_bound",
            message="tuple orbit-profile result cells exceed the admitted bound",
        )
    request_checkpoint("before tuple-family result construction")
    return TupleFamilyOrbitResult(
        source=request,
        rows=tuple(rows),
        is_union_of_complete_ambient_orbits=complete,
    )


__all__ = [
    "MAX_TUPLE_ORBIT_ACTIONS",
    "MAX_TUPLE_ORBIT_GROUP_ORDER",
    "MAX_TUPLE_ORBIT_GROUP_WORK",
    "MAX_TUPLE_ORBIT_IMAGES",
    "MAX_TUPLE_ORBIT_RESULT_CELLS",
    "MAX_TUPLE_ORBIT_TRANSPORTERS",
    "tuple_family_orbit_profile",
]
