"""Exact containment-profile kernels shared by native and catalog operations."""

from collections import Counter
from itertools import combinations

from jacobian._execution import request_checkpoint
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    IncidenceMomentComparison,
    IncidenceMultiplicityDifference,
    IncidenceStructure,
    _containment_axes,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

type _SubsetProfile = tuple[tuple[tuple[str, ...], int], ...]
type _Histogram = tuple[tuple[int, int], ...]
type ContainmentProfileData = tuple[
    _SubsetProfile, _Histogram, int, int, int, bool, int | None
]


def containment_profile_data(
    incidence: IncidenceStructure | FiniteHypergraph, order: int
) -> ContainmentProfileData:
    """Return one complete fixed-order multiplicity profile."""

    points, blocks = _containment_axes(incidence)
    request_checkpoint("before containment profile indexing")
    if order == 0:
        count = len(blocks)
        return (((), count),), ((count, 1),), count, count, count, True, count
    if order > len(points):
        return (), (), 0, 0, 0, True, 0

    # One bit per indexed block, not per distinct member set: duplicate
    # blocks contribute independently. Short blocks cannot contain an output
    # subset, so they need neither bits nor membership indexing at this order.
    eligible_blocks = tuple(block for block in blocks if len(block) >= order)
    byte_count = (len(eligible_blocks) + 7) // 8
    # Owner admission precedes these allocations. Even at the hypergraph
    # carrier ceilings, packed storage is at most 256 * 1,500 bytes; conversion
    # temporarily adds at most 256 * 12,000 bits of integer-mask content.
    packed = {point: bytearray(byte_count) for point in points}
    for index, block in enumerate(eligible_blocks):
        request_checkpoint("during containment profile indexing")
        byte_index, bit_index = divmod(index, 8)
        bit = 1 << bit_index
        for point in block:
            packed[point][byte_index] |= bit
    # Byte writes avoid repeatedly copying a growing Python integer for each
    # incidence. Conversion and intersections are bounded by the block axis.
    masks = {point: int.from_bytes(bits, "little") for point, bits in packed.items()}
    entries: list[tuple[tuple[str, ...], int]] = []
    remaining_positions = range(1, order)
    for index, subset in enumerate(combinations(points, order)):
        if index % 64 == 0:
            request_checkpoint("during containment profile enumeration")
        containing = masks[subset[0]]
        for position in remaining_positions:
            containing &= masks[subset[position]]
            if not containing:
                break
        entries.append((subset, containing.bit_count()))
    subset_profile = tuple(entries)
    histogram = tuple(sorted(Counter(count for _, count in subset_profile).items()))
    multiplicities = tuple(count for _, count in subset_profile)
    minimum = min(multiplicities, default=0)
    maximum = max(multiplicities, default=0)
    return (
        subset_profile,
        histogram,
        sum(multiplicities),
        minimum,
        maximum,
        minimum == maximum,
        minimum if minimum == maximum else None,
    )


def incidence_trade_data(
    left: IncidenceStructure, right: IncidenceStructure, max_order: int
) -> tuple[int, tuple[IncidenceMomentComparison, ...], bool]:
    comparisons: list[IncidenceMomentComparison] = []
    for order in range(1, max_order + 1):
        left_profile = containment_profile_data(left, order)
        right_profile = containment_profile_data(right, order)
        differences = tuple(
            IncidenceMultiplicityDifference(
                subset=left_entry[0],
                left_multiplicity=left_entry[1],
                right_multiplicity=right_entry[1],
            )
            for left_entry, right_entry in zip(
                left_profile[0], right_profile[0], strict=True
            )
            if left_entry[1] != right_entry[1]
        )
        comparisons.append(
            IncidenceMomentComparison._from_kernel(
                left, right, order, left_profile[2], right_profile[2], differences
            )
        )
    comparison_tuple = tuple(comparisons)
    return (
        len(left.blocks) - len(right.blocks),
        comparison_tuple,
        all(comparison.equal for comparison in comparison_tuple),
    )


__all__ = ["ContainmentProfileData", "containment_profile_data", "incidence_trade_data"]
