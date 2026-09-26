"""Bounded highest-positive-coroot operations."""

from __future__ import annotations

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.root_systems._cartan import connected_components
from jacobian.math.groups.root_systems._highest_coroot_models import (
    HighestCorootComponent,
    HighestCorootsResult,
)
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    CartanMatrix,
    FiniteCartanDatum,
)
from jacobian.math.groups.root_systems.operations import (
    _admit_cartan_finite_type,
    _as_cartan,
    _cartan_datum_from_admitted,
    _positive_coroots_from_admitted,
)

MAX_HIGHEST_COROOT_WORK = 160_000
MAX_HIGHEST_COROOT_OUTPUT_CELLS = 128_000


def highest_coroots(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> HighestCorootsResult:
    """Return each factor's unique maximum positive coroot and comarks.

    Coroot dominance is the coordinatewise order in the matching
    simple-coroot basis: ``alpha^vee <= beta^vee`` iff every coefficient of
    ``beta^vee - alpha^vee`` is nonnegative. The positive-coroot table is
    computed once, and component indices point directly into its canonical
    positive-root order.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    work_bound = (
        MAX_POSITIVE_ROOTS**2 * rank + 2 * MAX_POSITIVE_ROOTS * rank**2 + rank**3
    )
    output_bound = (
        4_096
        + MAX_POSITIVE_ROOTS * (200 + 56 * MAX_RANK)
        + MAX_RANK * (512 + 20 * MAX_RANK)
    )
    if work_bound > MAX_HIGHEST_COROOT_WORK or output_bound > (
        MAX_HIGHEST_COROOT_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.highest_coroot_bounds",
            message="highest-coroot profile exceeds its admitted work or output envelope",
        )

    datum: FiniteCartanDatum = _cartan_datum_from_admitted(cartan)
    positive_coroot_table = _positive_coroots_from_admitted(datum)
    pairs = positive_coroot_table.positive_root_coroot_pairs
    profiles: list[HighestCorootComponent] = []
    for component_index, simple_indices in enumerate(connected_components(rows)):
        factor = set(simple_indices)
        factor_pairs = tuple(
            (pair_index, pair)
            for pair_index, pair in enumerate(pairs)
            if (
                support := {
                    i for i, value in enumerate(pair.root_coefficients) if value
                }
            )
            and support <= factor
        )
        maxima = tuple(
            (pair_index, candidate)
            for pair_index, candidate in factor_pairs
            if all(
                all(
                    other.coroot_coefficients[index]
                    <= candidate.coroot_coefficients[index]
                    for index in simple_indices
                )
                for _, other in factor_pairs
            )
        )
        if len(maxima) != 1:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        pair_index, highest = maxima[0]
        profiles.append(
            HighestCorootComponent(
                component_index=component_index,
                simple_root_indices=simple_indices,
                highest_positive_root_coroot_pair_index=pair_index,
                coroot_preimage_root_coefficients=highest.root_coefficients,
                highest_coroot_coefficients=highest.coroot_coefficients,
                comarks=tuple(
                    highest.coroot_coefficients[index] for index in simple_indices
                ),
            )
        )
    return HighestCorootsResult._from_kernel(
        positive_coroot_table,
        tuple(profiles),
    )
