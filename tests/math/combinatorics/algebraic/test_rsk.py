"""Exact permutation RSK pairs and their inverse correspondence."""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics import algebraic as algebraic_combinatorics
from jacobian.math.combinatorics.algebraic._models import (
    RSKInversePermutationRequest,
    RSKPermutationRequest,
)
from jacobian.math.combinatorics.algebraic._tools import (
    TOOLS,
    inverse_rsk_permutation,
    rsk_permutation,
)
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_LENGTH,
    FinitePermutation,
    PermutationRSKPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import StandardYoungTableau


def _reference_rsk(
    permutation: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Independent linear-scan row insertion oracle for short permutations."""

    insertion: list[list[int]] = []
    recording: list[list[int]] = []
    for position, value in enumerate(permutation, start=1):
        current = value
        for row_index, row in enumerate(insertion):
            column = next(
                (index for index, entry in enumerate(row) if entry > current), len(row)
            )
            if column == len(row):
                row.append(current)
                recording[row_index].append(position)
                break
            row[column], current = current, row[column]
        else:
            insertion.append([current])
            recording.append([position])
    return tuple(map(tuple, insertion)), tuple(map(tuple, recording))


def _forward(images: tuple[int, ...]) -> PermutationRSKPair:
    source = FinitePermutation(images=images)
    return rsk_permutation(RSKPermutationRequest(permutation=source))


def _inverse(pair: PermutationRSKPair) -> FinitePermutation:
    return inverse_rsk_permutation(RSKInversePermutationRequest(pair=pair))


def test_permutation_pair_schema_names_convention_and_source_free_value() -> None:
    request_schema = RSKPermutationRequest.model_json_schema()
    inverse_schema = RSKInversePermutationRequest.model_json_schema()
    pair_schema = PermutationRSKPair.model_json_schema()
    assert request_schema["properties"]["convention"]["const"] == (
        "ROW_INSERTION_RSK_V1"
    )
    assert inverse_schema["properties"]["convention"]["const"] == (
        "ROW_INSERTION_RSK_V1"
    )
    assert pair_schema["properties"].keys() == {
        "p_tableau",
        "q_tableau",
        "convention",
    }
    permutation_schema = request_schema["$defs"]["FinitePermutation"]
    assert permutation_schema["properties"]["images"]["maxItems"] == (
        MAX_RSK_WORD_LENGTH
    )
    assert "N(N-1)/2" in (inverse_schema.get("description") or "")


def test_empty_and_identity_pairs_roundtrip() -> None:
    empty = _forward(())
    assert empty.shape.parts == ()
    assert empty.p_tableau.rows == empty.q_tableau.rows == ()
    assert _inverse(empty).images == ()

    identity = _forward((1, 2, 3, 4, 5))
    assert identity.shape.parts == (5,)
    assert identity.p_tableau.rows == ((1, 2, 3, 4, 5),)
    assert identity.q_tableau.rows == ((1, 2, 3, 4, 5),)
    assert _inverse(identity).images == (1, 2, 3, 4, 5)


def test_reverse_permutation_pair_roundtrips() -> None:
    pair = _forward((5, 4, 3, 2, 1))
    assert pair.shape.parts == (1, 1, 1, 1, 1)
    assert _inverse(pair).images == (5, 4, 3, 2, 1)


def test_all_permutations_through_size_five_match_independent_rsk_and_inverse() -> None:
    for size in range(6):
        for images in itertools.permutations(range(1, size + 1)):
            pair = _forward(images)
            expected_p, expected_q = _reference_rsk(images)
            assert pair.p_tableau.rows == expected_p
            assert pair.q_tableau.rows == expected_q

            decoded_pair = PermutationRSKPair.model_validate_json(
                pair.model_dump_json()
            )
            inverse_request = RSKInversePermutationRequest.model_validate_json(
                json.dumps({"pair": json.loads(decoded_pair.model_dump_json())})
            )
            recovered = inverse_rsk_permutation(inverse_request)
            assert recovered.images == images
            forward_request = RSKPermutationRequest.model_validate_json(
                json.dumps({"permutation": json.loads(recovered.model_dump_json())})
            )
            assert rsk_permutation(forward_request) == decoded_pair


def test_permutation_inversion_swaps_the_standard_tableaux() -> None:
    for images in itertools.permutations(range(1, 6)):
        pair = _forward(images)
        inverse_images = tuple(images.index(value) + 1 for value in range(1, 6))
        inverse_pair = _forward(inverse_images)
        assert pair.p_tableau == inverse_pair.q_tableau
        assert pair.q_tableau == inverse_pair.p_tableau


def test_pair_validation_rejects_invalid_permutations_and_shapes() -> None:
    with pytest.raises(ValidationError):
        FinitePermutation(images=(1, 2, 2))
    with pytest.raises(ValidationError):
        FinitePermutation.model_validate({"images": [True]})
    with pytest.raises(ValidationError):
        PermutationRSKPair(
            p_tableau=StandardYoungTableau(rows=((1, 2),)),
            q_tableau=StandardYoungTableau(rows=((1,), (2,))),
        )


def test_native_permutation_entry_admits_canonical_values_before_insertion() -> None:
    with pytest.raises(OperationDomainValidationError):
        algebraic_combinatorics.permutation_rsk((1, 2, 3))

    oversized = FinitePermutation.model_construct(
        images=tuple(range(1, MAX_RSK_WORD_LENGTH + 2))
    )
    with pytest.raises(OperationResourceAdmissionError):
        algebraic_combinatorics.permutation_rsk(oversized)


def test_inverse_rejects_nonstandard_tableaux_before_reverse_insertion() -> None:
    malformed = PermutationRSKPair.model_construct(
        p_tableau=StandardYoungTableau.model_construct(rows=((2,),)),
        q_tableau=StandardYoungTableau.model_construct(rows=((1,),)),
    )
    with pytest.raises(OperationDomainValidationError):
        _inverse(malformed)


def test_native_inverse_admits_tableau_cells_before_reverse_insertion() -> None:
    oversized_row = tuple(range(1, MAX_RSK_WORD_LENGTH + 2))
    oversized = PermutationRSKPair.model_construct(
        p_tableau=StandardYoungTableau.model_construct(rows=(oversized_row,)),
        q_tableau=StandardYoungTableau.model_construct(rows=(oversized_row,)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        algebraic_combinatorics.inverse_permutation_rsk(oversized)


def test_inverse_accepts_the_full_cell_boundary_and_rejects_above_it() -> None:
    identity = tuple(range(1, MAX_RSK_WORD_LENGTH + 1))
    pair = _forward(identity)
    assert pair.shape.parts == (MAX_RSK_WORD_LENGTH,)
    assert _inverse(pair).images == identity
    assert MAX_RSK_WORD_LENGTH.bit_length() == MAX_RSK_ROW_SEARCH_COMPARISONS

    decreasing = tuple(range(MAX_RSK_WORD_LENGTH, 0, -1))
    column_pair = _forward(decreasing)
    assert column_pair.shape.parts == (1,) * MAX_RSK_WORD_LENGTH
    assert _inverse(column_pair).images == decreasing

    with pytest.raises(ValidationError):
        FinitePermutation(images=tuple(range(1, MAX_RSK_WORD_LENGTH + 2)))


def test_operation_examples_publish_both_directions() -> None:
    catalog = Catalog(TOOLS)
    for operation_id, expected in (
        ("combinatorics.rsk.permutation.compute", _forward((1, 3, 2))),
        (
            "tableau.rsk.inverse_permutation.compute",
            FinitePermutation(images=(3, 1, 2)),
        ),
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert operation.examples
        result = invoke_operation(operation_id, operation.examples[0].input, catalog)
        assert result.output == expected.model_dump(mode="json")

    assert {tool.operation_id for tool in TOOLS} >= {
        "combinatorics.rsk.permutation.compute",
        "tableau.rsk.inverse_permutation.compute",
    }
