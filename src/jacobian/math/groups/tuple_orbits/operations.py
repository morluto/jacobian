"""Diagonal permutation action on explicit tuple families."""

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.operations import _backend_group, _full_permutation_form
from jacobian.math.groups.tuple_orbits._models import (
    TupleFamilyOrbitResult,
    TupleFamilyOrbitSource,
    TupleOrbitRow,
)

MAX_TUPLE_ORBIT_GROUP_ORDER = 5_000
MAX_TUPLE_ORBIT_ACTIONS = 1_000_000


def tuple_family_orbit_profile(
    request: TupleFamilyOrbitSource,
) -> TupleFamilyOrbitResult:
    backend = _backend_group(request.group)
    order = int(backend.order())
    work = order * max(1, len(request.family)) * max(1, request.arity)
    if order > MAX_TUPLE_ORBIT_GROUP_ORDER or work > MAX_TUPLE_ORBIT_ACTIONS:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="group.tuple_orbit.enumeration_bound",
            message="tuple-family group actions exceed the admitted exact bound",
        )
    elements = tuple(
        sorted(
            _full_permutation_form(element, request.group.degree)
            for element in backend.elements
        )
    )
    source_by_representative: dict[tuple[int, ...], list[int]] = {}
    orbit_data: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    source_members = {tuple(member) for member in request.family}
    complete_orbits = True
    for source_index, member in enumerate(request.family):
        images = {tuple(element[value] for value in member) for element in elements}
        complete_orbits = complete_orbits and images.issubset(source_members)
        representative = min(images)
        source_by_representative.setdefault(representative, []).append(source_index)
        if representative not in orbit_data:
            transporter = min(
                element
                for element in elements
                if tuple(element[value] for value in member) == representative
            )
            orbit_data[representative] = (len(images), transporter)
    rows = tuple(
        TupleOrbitRow(
            representative=representative,
            source_indices=tuple(source_by_representative[representative]),
            orbit_size=orbit_data[representative][0],
            stabilizer_size=order // orbit_data[representative][0],
            least_transporter=orbit_data[representative][1],
        )
        for representative in sorted(source_by_representative)
    )
    return TupleFamilyOrbitResult(
        source=request,
        rows=rows,
        is_union_of_complete_ambient_orbits=complete_orbits,
    )


__all__ = ["tuple_family_orbit_profile"]
