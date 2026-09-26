"""Exact reverse row-insertion traces checked by a linear-scan oracle."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic import (
    RSKWordInverseTraceResult,
    inverse_row_insertion_rsk_trace,
    row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic._models import RSKInverseWordRequest
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
from jacobian.math.logic.languages.words.values import FiniteWord

_ReverseEvent = tuple[
    int,
    str,
    int,
    int,
    int,
    int,
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
]


def _linear_reverse_oracle(
    pair: RSKTableauPair,
) -> tuple[tuple[str, ...], tuple[_ReverseEvent, ...]]:
    insertion = [list(row) for row in pair.insertion_tableau.rows]
    labels: list[tuple[int, int] | None] = [None] * sum(pair.shape.parts)
    for row, values in enumerate(pair.recording_tableau.rows):
        for column, label in enumerate(values):
            labels[label - 1] = (row, column)
    letters = [""] * len(labels)
    events = []
    for position in range(len(labels), 0, -1):
        cell = labels[position - 1]
        assert cell is not None
        row, column = cell
        carried = insertion[row].pop()
        removed = carried
        if not insertion[row]:
            insertion.pop()
        path = []
        for upper in range(row - 1, -1, -1):
            target = next(
                (
                    index
                    for index in range(len(insertion[upper]) - 1, -1, -1)
                    if insertion[upper][index] < carried
                ),
                None,
            )
            assert target is not None
            displaced = insertion[upper][target]
            insertion[upper][target] = carried
            carried = displaced
            path.append((upper, target, displaced))
        letters[position - 1] = pair.alphabet[carried - 1]
        events.append(
            (
                position,
                letters[position - 1],
                row,
                column,
                removed,
                carried,
                tuple(path),
                tuple(map(len, insertion)),
            )
        )
    return tuple(letters), tuple(events)


@pytest.mark.parametrize("length", range(5))
def test_inverse_trace_matches_independent_exhaustive_oracle(length: int) -> None:
    alphabet = ("z", "a", "m")
    for letters in product(alphabet, repeat=length):
        source = FiniteWord(alphabet=alphabet, letters=letters)
        pair = row_insertion_rsk(source)
        result = inverse_row_insertion_rsk_trace(pair)
        expected_word, expected_events = _linear_reverse_oracle(pair)
        assert result.word.letters == expected_word
        assert (
            tuple(
                (
                    event.position,
                    event.letter,
                    event.removed_row,
                    event.removed_column,
                    event.removed_entry,
                    event.output_entry,
                    tuple(
                        (step.row, step.column, step.displaced_entry)
                        for step in event.reverse_bump_path
                    ),
                    event.row_lengths,
                )
                for event in result.reverse_insertion_events
            )
            == expected_events
        )
        assert row_insertion_rsk(result.word) == pair


def test_reverse_events_replay_to_empty_tableaux_and_serialize() -> None:
    source = FiniteWord(alphabet=("a", "b", "c"), letters=("c", "b", "a"))
    pair = row_insertion_rsk(source)
    result = inverse_row_insertion_rsk_trace(pair)
    insertion = [list(row) for row in pair.insertion_tableau.rows]
    recording = [list(row) for row in pair.recording_tableau.rows]
    for event in result.reverse_insertion_events:
        assert recording[event.removed_row].pop() == event.position
        carried = event.removed_entry
        assert insertion[event.removed_row].pop() == carried
        for step in event.reverse_bump_path:
            assert insertion[step.row][step.column] == step.displaced_entry
            insertion[step.row][step.column] = carried
            carried = step.displaced_entry
        assert carried == event.output_entry
        while insertion and not insertion[-1]:
            insertion.pop()
            recording.pop()
        assert tuple(map(len, insertion)) == event.row_lengths
    assert insertion == recording == []
    assert (
        RSKWordInverseTraceResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_empty_and_single_cell_inverse_traces() -> None:
    for source in (
        FiniteWord(alphabet=(), letters=()),
        FiniteWord(alphabet=("x",), letters=("x",)),
    ):
        result = inverse_row_insertion_rsk_trace(row_insertion_rsk(source))
        assert result.word == source
        assert len(result.reverse_insertion_events) == len(source.letters)


def test_inverse_trace_tool_uses_canonical_inverse_request() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "tableau.rsk.inverse_word.trace.compute"
    )
    request = RSKInverseWordRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.word.letters == _linear_reverse_oracle(request.pair)[0]


def test_inverse_trace_revalidates_forged_pair_shape_before_indexing() -> None:
    pair = row_insertion_rsk(FiniteWord(alphabet=("a",), letters=("a",))).model_copy(
        update={"shape": row_insertion_rsk(FiniteWord(alphabet=(), letters=())).shape}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        inverse_row_insertion_rsk_trace(pair)
    assert (
        error.value.errors()[0]["type"]
        == "algebraic_combinatorics.rsk_inverse_trace_pair"
    )


def test_inverse_trace_executes_against_canonicalized_pair() -> None:
    original = row_insertion_rsk(FiniteWord(alphabet=("a", "b"), letters=("b", "a")))
    forged = original.model_copy(update={"insertion_tableau": {"rows": [[1], [2]]}})
    result = inverse_row_insertion_rsk_trace(forged)
    assert result.word == FiniteWord(alphabet=("a", "b"), letters=("b", "a"))
    assert row_insertion_rsk(result.word) == original


def test_inverse_trace_deserialization_rejects_missing_event() -> None:
    result = inverse_row_insertion_rsk_trace(
        row_insertion_rsk(FiniteWord(alphabet=("a", "b"), letters=("b", "a")))
    )
    payload = result.model_dump(mode="python")
    payload["reverse_insertion_events"] = payload["reverse_insertion_events"][:-1]
    with pytest.raises(ValidationError, match="rsk_reverse_trace_positions"):
        RSKWordInverseTraceResult.model_validate(payload)


def test_inverse_trace_deserialization_rejects_impossible_corner() -> None:
    result = inverse_row_insertion_rsk_trace(
        row_insertion_rsk(FiniteWord(alphabet=("a",), letters=("a",)))
    )
    payload = result.model_dump(mode="python")
    payload["reverse_insertion_events"][0]["removed_column"] = 499
    with pytest.raises(ValidationError, match="rsk_reverse_trace_cell"):
        RSKWordInverseTraceResult.model_validate(payload)


def test_inverse_trace_deserialization_rejects_unbound_emitted_rank() -> None:
    result = inverse_row_insertion_rsk_trace(
        row_insertion_rsk(FiniteWord(alphabet=("a",), letters=("a",)))
    )
    payload = result.model_dump(mode="python")
    payload["reverse_insertion_events"][0]["removed_entry"] = 50
    payload["reverse_insertion_events"][0]["output_entry"] = 50
    with pytest.raises(ValidationError, match="rsk_reverse_trace_ranks"):
        RSKWordInverseTraceResult.model_validate(payload)
