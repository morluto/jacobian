"""A collision is a source-checkable equality of two ordered differences."""

import json
from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.additive._models import (
    IntegerVector,
    OrderedDifferenceCollision,
    OrderedDifferenceEntry,
    OrderedDifferencePair,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
)
from jacobian.math.combinatorics.additive.operations import (
    ordered_difference_profile,
    verify_ordered_difference_profile,
)

_SQUARE = ((0, 0), (1, 0), (0, 1), (1, 1))


def _profile(*vectors: tuple[int, ...]) -> OrderedDifferenceProfileResult:
    request = OrderedDifferenceProfileRequest.model_validate(
        {"vectors": {"vectors": [{"coordinates": v} for v in vectors]}}
    )
    return ordered_difference_profile(request.vectors)


def _source_oracle(
    source: tuple[tuple[int, ...], ...],
) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for i, left in enumerate(source):
        for j, right in enumerate(source):
            if i != j:
                difference = tuple(a - b for a, b in zip(left, right, strict=True))
                groups[difference].append((i, j))
    return dict(groups)


def test_square_witness_contains_both_source_pairs_and_difference() -> None:
    result = _profile(*_SQUARE)
    witness = result.first_collision
    assert witness is not None
    assert witness.difference.as_int_tuple() == (-1, 0)
    assert tuple((p.left_index, p.right_index) for p in witness.pairs) == (
        (0, 1),
        (2, 3),
    )
    for pair in witness.pairs:
        left = result.vectors.vectors[pair.left_index].as_int_tuple()
        right = result.vectors.vectors[pair.right_index].as_int_tuple()
        assert tuple(a - b for a, b in zip(left, right, strict=True)) == (-1, 0)


@pytest.mark.parametrize(
    "source",
    [
        _SQUARE,
        ((0,), (1,), (2,)),
        ((0,), (1,), (2,), (3,), (4,)),
        ((4,), (1,), (3,), (0,), (2,)),
        ((0,), (1,), (3,), (10,), (11,), (13,)),
        ((0, 0), (1, 0), (0, 1)),
        ((0,), (1,), (3,)),
        ((9, -2),),
        ((-999999,), (0,), (999999,)),
    ],
)
def test_profile_and_first_collision_match_independent_source_oracle(
    source: tuple[tuple[int, ...], ...],
) -> None:
    result = _profile(*source)
    groups = _source_oracle(source)
    expected_counts = Counter(
        {difference: len(pairs) for difference, pairs in groups.items()}
    )
    actual_counts = Counter(
        {
            entry.difference.as_int_tuple(): entry.multiplicity
            for entry in result.entries
        }
    )
    assert actual_counts == expected_counts
    assert sum(expected_counts.values()) == len(source) * (len(source) - 1)
    assert result.vectors.vectors == tuple(IntegerVector(coordinates=v) for v in source)
    repeated = sorted(
        difference for difference, pairs in groups.items() if len(pairs) > 1
    )
    assert result.has_repeated_difference is bool(repeated)
    witness = result.first_collision
    if not repeated:
        assert witness is None
    else:
        assert witness is not None
        assert witness.difference.as_int_tuple() == repeated[0]
        assert tuple((p.left_index, p.right_index) for p in witness.pairs) == tuple(
            sorted(groups[repeated[0]])[:2]
        )
    assert verify_ordered_difference_profile(result)


def test_three_term_progression_witness_may_share_an_endpoint() -> None:
    witness = _profile((0,), (1,), (2,)).first_collision
    assert witness is not None
    assert witness.difference.as_int_tuple() == (-1,)
    assert tuple((p.left_index, p.right_index) for p in witness.pairs) == (
        (0, 1),
        (1, 2),
    )


def test_complete_collision_survives_json_round_trip() -> None:
    result = _profile(*_SQUARE)
    wire = result.model_dump_json()
    assert json.loads(wire)["first_collision"] == {
        "difference": {"coordinates": ["-1", "0"]},
        "pairs": [
            {"left_index": 0, "right_index": 1},
            {"left_index": 2, "right_index": 3},
        ],
    }
    decoded = OrderedDifferenceProfileResult.model_validate_json(wire)
    assert decoded == result
    assert verify_ordered_difference_profile(decoded)


@pytest.mark.parametrize("forgery", ["difference", "second_pair", "later_collision"])
def test_false_but_canonical_witness_decodes_until_explicit_verification(
    forgery: str,
) -> None:
    payload = _profile(*_SQUARE).model_dump(mode="json")
    witness = payload["first_collision"]
    if forgery == "difference":
        witness["difference"]["coordinates"] = ["-2", "0"]
    elif forgery == "second_pair":
        witness["pairs"][1] = {"left_index": 1, "right_index": 3}
    else:
        witness["difference"]["coordinates"] = ["0", "-1"]
        witness["pairs"] = [
            {"left_index": 0, "right_index": 2},
            {"left_index": 1, "right_index": 3},
        ]
    decoded = OrderedDifferenceProfileResult.model_validate_json(json.dumps(payload))
    assert decoded.model_dump(mode="json")["first_collision"] == witness
    assert not verify_ordered_difference_profile(decoded)


@pytest.mark.parametrize("field", ["left_index", "right_index"])
@pytest.mark.parametrize("pair_index", [0, 1])
def test_verifier_rejects_boolean_in_either_native_witness_pair(
    field: str,
    pair_index: int,
) -> None:
    result = _profile(*_SQUARE)
    witness = result.first_collision
    assert witness is not None
    pairs = list(witness.pairs)
    pairs[pair_index] = pairs[pair_index].model_copy(update={field: True})
    forged = result.model_copy(
        update={"first_collision": witness.model_copy(update={"pairs": tuple(pairs)})}
    )
    assert not verify_ordered_difference_profile(forged)


@pytest.mark.parametrize("forgery", ["difference", "second_pair", "boolean_coordinate"])
def test_verifier_rejects_native_collision_forgery(forgery: str) -> None:
    result = _profile(*_SQUARE)
    witness = result.first_collision
    assert witness is not None
    if forgery == "second_pair":
        forged_witness = witness.model_copy(
            update={
                "pairs": (
                    witness.pairs[0],
                    OrderedDifferencePair(left_index=1, right_index=3),
                )
            }
        )
    else:
        difference = (
            witness.difference.model_copy(update={"coordinates": (-1, False)})
            if forgery == "boolean_coordinate"
            else IntegerVector(coordinates=(-2, 0))
        )
        forged_witness = witness.model_copy(update={"difference": difference})
    forged = result.model_copy(update={"first_collision": forged_witness})
    assert not verify_ordered_difference_profile(forged)


@pytest.mark.parametrize(
    "forgery",
    [
        "one_pair",
        "three_pairs",
        "duplicate",
        "reversed",
        "diagonal",
        "out_of_bounds",
        "dimension",
    ],
)
def test_collision_structural_shape_is_validated(forgery: str) -> None:
    payload = _profile(*_SQUARE).model_dump(mode="json")
    witness = payload["first_collision"]
    if forgery == "one_pair":
        witness["pairs"] = witness["pairs"][:1]
    elif forgery == "three_pairs":
        witness["pairs"].append({"left_index": 3, "right_index": 0})
    elif forgery == "duplicate":
        witness["pairs"][1] = witness["pairs"][0]
    elif forgery == "reversed":
        witness["pairs"].reverse()
    elif forgery == "diagonal":
        witness["pairs"][1] = {"left_index": 2, "right_index": 2}
    elif forgery == "out_of_bounds":
        witness["pairs"][1]["right_index"] = 4
    else:
        witness["difference"]["coordinates"] = ["-1"]
    with pytest.raises(ValidationError):
        OrderedDifferenceProfileResult.model_validate_json(json.dumps(payload))


def test_first_collision_uses_first_two_of_three_pairs() -> None:
    result = _profile((0,), (1,), (3,), (10,), (11,), (13,))
    witness = result.first_collision
    assert witness is not None
    assert witness.difference.as_int_tuple() == (-10,)
    assert tuple((p.left_index, p.right_index) for p in witness.pairs) == (
        (0, 3),
        (1, 4),
    )
    payload = result.model_dump(mode="json")
    payload["first_collision"]["pairs"][1] = {"left_index": 2, "right_index": 5}
    decoded = OrderedDifferenceProfileResult.model_validate_json(json.dumps(payload))
    assert not verify_ordered_difference_profile(decoded)


@pytest.mark.parametrize(
    "missing", ["coordinates", "left_index", "right_index", "difference", "pairs"]
)
def test_incomplete_native_collision_values_are_false_claims(missing: str) -> None:
    result = _profile(*_SQUARE)
    witness = result.first_collision
    assert witness is not None
    if missing == "coordinates":
        malformed = witness.model_copy(
            update={"difference": IntegerVector.model_construct()}
        )
    elif missing in {"left_index", "right_index"}:
        pair = (
            OrderedDifferencePair.model_construct(right_index=1)
            if missing == "left_index"
            else OrderedDifferencePair.model_construct(left_index=0)
        )
        malformed = witness.model_copy(update={"pairs": (pair, witness.pairs[1])})
    elif missing == "difference":
        malformed = OrderedDifferenceCollision.model_construct(pairs=witness.pairs)
    else:
        malformed = OrderedDifferenceCollision.model_construct(
            difference=witness.difference
        )
    assert not verify_ordered_difference_profile(
        result.model_copy(update={"first_collision": malformed})
    )


def test_collision_preserves_seven_digit_derived_difference() -> None:
    result = _profile((-999999,), (-999998,), (999998,), (999999,))
    witness = result.first_collision
    assert witness is not None
    assert witness.difference.as_int_tuple() == (-1999997,)
    assert tuple((pair.left_index, pair.right_index) for pair in witness.pairs) == (
        (0, 2),
        (1, 3),
    )
    decoded = OrderedDifferenceProfileResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_ordered_difference_profile(decoded)


def test_collision_preserves_maximum_canonical_dimension() -> None:
    from jacobian.math.combinatorics.additive._models import _MAX_DIMENSION

    tail = (0,) * (_MAX_DIMENSION - 1)
    result = _profile((0, *tail), (1, *tail), (2, *tail))
    assert result.first_collision is not None
    assert result.first_collision.difference.coordinates == (-1, *tail)
    decoded = OrderedDifferenceProfileResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_ordered_difference_profile(decoded)


def test_shared_row_payload_is_bounded_before_pair_traversal() -> None:
    result = _profile(*((index,) for index in range(128)))
    pairs = tuple(
        OrderedDifferencePair(left_index=left, right_index=right)
        for left in range(128)
        for right in range(128)
        if left != right
    )
    row = OrderedDifferenceEntry.model_construct(
        difference=IntegerVector(coordinates=(-1,)),
        multiplicity=len(pairs),
        pairs=pairs,
    )
    claim = result.model_copy(update={"entries": (row,) * len(pairs)})
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_ordered_difference_profile(claim)
    assert (
        error.value.errors()[0]["type"]
        == "additive_combinatorics.ordered_difference_output_exceeded"
    )
