"""Relational homomorphism admission shared by the native and catalog paths.

The defining invariant — for every relation symbol ``R`` and every tuple
``t`` in ``R^A``, ``h(t)`` lies in ``R^B`` — is replayed exhaustively by the
kernel, so the complete transport work ``sum |R^A|`` is preflighted here
before any tuple is transported. Structural carrier-map defects and
signature mismatches are typed domain rejections; a structurally valid
request whose transport work exceeds the published envelope is a resource
admission failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
)


def admit_homomorphism_check(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> int:
    """Preflight one exhaustive homomorphism replay; return its tuple count.

    Raises ``OperationDomainValidationError`` when the two structures do not
    share one signature or the candidate map is not a total function from the
    source carrier into the target carrier, and
    ``OperationResourceAdmissionError`` when the complete transport work
    exceeds the published envelope.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    if len(carrier_map) != source.carrier_size:
        raise OperationDomainValidationError(
            location=("carrier_map",),
            code="relational.homomorphism.carrier_map_axis",
            message=(
                "a candidate homomorphism must be a total function on the "
                f"complete source carrier; expected {source.carrier_size} "
                f"images, got {len(carrier_map)}"
            ),
        )
    for position, image in enumerate(carrier_map):
        if not isinstance(image, int) or isinstance(image, bool):
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message="carrier map images must be exact integers",
            )
        if not 0 <= image < target.carrier_size:
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message=(
                    "every carrier map image must belong to the target "
                    f"carrier 0..{target.carrier_size - 1}"
                ),
            )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    if transport_tuples > MAX_RELATIONAL_TRANSPORT_TUPLES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.transport_bound",
            message=(
                f"the exhaustive replay transports {transport_tuples} source "
                f"relation tuples, exceeding the "
                f"{MAX_RELATIONAL_TRANSPORT_TUPLES}-tuple envelope"
            ),
        )
    return transport_tuples


__all__ = ["MAX_RELATIONAL_TRANSPORT_TUPLES", "admit_homomorphism_check"]
