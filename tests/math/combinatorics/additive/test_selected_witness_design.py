"""Design evidence for retaining a selected collision alongside its source.

The full profile already contains the evidence. This fixture motivates retaining
it when selecting a compact witness, not publishing another profile operation.
It does not depend on the pending first_collision representation repair.
"""

from jacobian.math.combinatorics.additive._models import (
    IntegerVector,
    IntegerVectorSet,
    OrderedDifferenceEntry,
)
from jacobian.math.combinatorics.additive.operations import ordered_difference_profile


def _source(*values: int) -> IntegerVectorSet:
    return IntegerVectorSet(
        vectors=tuple(IntegerVector(coordinates=(n,)) for n in values)
    )


def _certifies_selected_collision(
    source: IntegerVectorSet, witness: OrderedDifferenceEntry
) -> bool:
    """Consume the retained relation directly, without repeating the profile."""
    pairs = witness.pairs
    if len(pairs) != 2 or pairs[0] == pairs[1]:
        return False
    count = len(source.vectors)
    for pair in pairs:
        if not 0 <= pair.left_index < count or not 0 <= pair.right_index < count:
            return False
        if pair.left_index == pair.right_index:
            return False
        left = source.vectors[pair.left_index].coordinates
        right = source.vectors[pair.right_index].coordinates
        if (
            tuple(a - b for a, b in zip(left, right, strict=True))
            != witness.difference.coordinates
        ):
            return False
    return True


def test_selected_witness_requires_two_distinct_source_pairs() -> None:
    source = _source(0, 1, 2)
    profile = ordered_difference_profile(source)
    collision = next(entry for entry in profile.entries if entry.multiplicity == 2)
    decoded = OrderedDifferenceEntry.model_validate_json(collision.model_dump_json())
    assert decoded.difference.coordinates == (-1,)
    assert _certifies_selected_collision(source, decoded)
    assert not _certifies_selected_collision(
        source, decoded.model_copy(update={"pairs": (decoded.pairs[0],)})
    )
    assert not _certifies_selected_collision(
        source,
        decoded.model_copy(update={"pairs": (decoded.pairs[0], decoded.pairs[0])}),
    )
    assert not _certifies_selected_collision(_source(0, 1, 3), decoded)
    assert not ordered_difference_profile(_source(0, 1, 3)).has_repeated_difference


def test_selected_witness_does_not_claim_all_collisions() -> None:
    source = _source(0, 1, 2, 3)
    collisions = tuple(
        entry
        for entry in ordered_difference_profile(source).entries
        if entry.multiplicity > 1
    )
    # A complete row for one difference does not enumerate all collisions.
    selected = next(entry for entry in collisions if entry.multiplicity == 2)
    assert _certifies_selected_collision(source, selected)
    assert len(collisions) > 1
    assert sum(len(entry.pairs) for entry in collisions) > len(selected.pairs)
