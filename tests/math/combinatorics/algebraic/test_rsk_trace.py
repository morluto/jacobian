"""Exact row-insertion ledgers checked against an independent insertion loop."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.algebraic import (
    RSKWordTraceResult,
    row_insertion_rsk,
    row_insertion_rsk_trace,
)
from jacobian.math.combinatorics.algebraic._models import (
    MAX_RSK_TRACE_RESULT_BYTES,
    RSKWordTraceRequest,
)
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.logic.languages.words.values import FiniteWord


def _reference_trace(
    word: FiniteWord,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[
        tuple[str, tuple[tuple[int, int, int], ...], int, int, int, tuple[int, ...]],
        ...,
    ],
]:
    """Use linear scans to independently compute pair and bump events."""
    ranks = {symbol: index for index, symbol in enumerate(word.alphabet, 1)}
    insertion: list[list[int]] = []
    recording: list[list[int]] = []
    events = []
    for position, letter in enumerate(word.letters, 1):
        carried = ranks[letter]
        row_index = 0
        path = []
        while row_index < len(insertion):
            row = insertion[row_index]
            column = next(
                (index for index, value in enumerate(row) if value > carried),
                len(row),
            )
            if column == len(row):
                row.append(carried)
                recording[row_index].append(position)
                added_row, added_column, added_entry = row_index, column, carried
                break
            bumped = row[column]
            row[column] = carried
            path.append((row_index, column, bumped))
            carried = bumped
            row_index += 1
        else:
            insertion.append([carried])
            recording.append([position])
            added_row, added_column, added_entry = row_index, 0, carried
        events.append(
            (
                letter,
                tuple(path),
                added_row,
                added_column,
                added_entry,
                tuple(len(row) for row in insertion),
            )
        )
    return (
        tuple(tuple(row) for row in insertion),
        tuple(tuple(row) for row in recording),
        tuple(events),
    )


@pytest.mark.parametrize("length", range(5))
def test_trace_matches_independent_exhaustive_small_word_oracle(length: int) -> None:
    alphabet = ("z", "a", "m")
    for letters in product(alphabet, repeat=length):
        word = FiniteWord(alphabet=alphabet, letters=letters)
        result = row_insertion_rsk_trace(word)
        expected_p, expected_q, expected_events = _reference_trace(word)
        assert result.tableau_pair.insertion_tableau.rows == expected_p
        assert result.tableau_pair.recording_tableau.rows == expected_q
        assert (
            tuple(
                (
                    event.letter,
                    tuple(
                        (step.row, step.column, step.bumped_entry)
                        for step in event.bump_path
                    ),
                    event.added_row,
                    event.added_column,
                    event.added_entry,
                    event.row_lengths,
                )
                for event in result.insertion_events
            )
            == expected_events
        )
        assert result.tableau_pair == row_insertion_rsk(word)


def test_trace_events_replay_to_both_tableaux_and_roundtrip() -> None:
    word = FiniteWord(alphabet=("a", "b", "c"), letters=("c", "b", "a"))
    result = row_insertion_rsk_trace(word)
    insertion: list[list[int]] = []
    recording: list[list[int]] = []
    rank = {symbol: index for index, symbol in enumerate(word.alphabet, 1)}
    for event in result.insertion_events:
        assert event.letter == word.letters[event.position - 1]
        carried = rank[word.letters[event.position - 1]]
        for step in event.bump_path:
            assert insertion[step.row][step.column] == step.bumped_entry
            insertion[step.row][step.column] = carried
            carried = step.bumped_entry
        assert event.added_entry == carried
        while len(insertion) <= event.added_row:
            insertion.append([])
            recording.append([])
        assert event.added_column == len(insertion[event.added_row])
        insertion[event.added_row].append(event.added_entry)
        recording[event.added_row].append(event.position)
        assert tuple(len(row) for row in insertion) == event.row_lengths
    assert tuple(map(tuple, insertion)) == result.tableau_pair.insertion_tableau.rows
    assert tuple(map(tuple, recording)) == result.tableau_pair.recording_tableau.rows
    assert RSKWordTraceResult.model_validate_json(result.model_dump_json()) == result


def test_empty_alphabet_trace_keeps_the_unique_empty_word() -> None:
    word = FiniteWord(alphabet=(), letters=())
    result = row_insertion_rsk_trace(word)
    assert result.word == word
    assert result.tableau_pair.alphabet == ()
    assert result.tableau_pair.shape.parts == ()
    assert result.insertion_events == ()


def test_trace_accepts_long_word_with_full_admitted_alphabet_height() -> None:
    alphabet = tuple(f"rank-{rank:02}" for rank in range(50))
    letters = tuple(reversed(alphabet)) * 10
    result = row_insertion_rsk_trace(FiniteWord(alphabet=alphabet, letters=letters))
    assert len(result.insertion_events) == 500
    assert len(result.tableau_pair.shape.parts) == 50
    assert result.tableau_pair == row_insertion_rsk(result.word)
    assert len(result.model_dump_json().encode()) <= MAX_RSK_TRACE_RESULT_BYTES


def test_trace_request_and_operation_have_canonical_convention_and_example() -> None:
    tool = next(
        item for item in TOOLS if item.operation_id == "tableau.rsk.word.trace.compute"
    )
    example = tool.examples[0]
    request = RSKWordTraceRequest.model_validate(example.input)
    assert request.convention == "ROW_INSERTION_RSK_V1"
    result = tool.run(request)
    assert len(result.insertion_events) == len(request.word.letters)


def test_trace_result_deserialization_rejects_impossible_terminal_row() -> None:
    result = row_insertion_rsk_trace(
        FiniteWord(alphabet=("a", "c"), letters=("c", "a", "a"))
    )
    payload = result.model_dump(mode="python")
    event = payload["insertion_events"][-1]
    event["added_row"] = 1
    event["added_column"] = 0
    event["bump_path"] = ({"row": 0, "column": 0, "bumped_entry": 2},)
    with pytest.raises(ValidationError, match="rsk_trace_prefix_shape"):
        RSKWordTraceResult.model_validate(payload)


def test_trace_result_deserialization_rejects_inconsistent_entry_chain() -> None:
    result = row_insertion_rsk_trace(
        FiniteWord(alphabet=("a", "b"), letters=("b", "a"))
    )
    payload = result.model_dump(mode="python")
    payload["insertion_events"][0]["added_entry"] = 1
    with pytest.raises(ValidationError, match="rsk_trace_entry_chain"):
        RSKWordTraceResult.model_validate(payload)


def test_trace_result_deserialization_rejects_missing_source_positions() -> None:
    result = row_insertion_rsk_trace(
        FiniteWord(alphabet=("a", "b"), letters=("b", "a"))
    )
    payload = result.model_dump(mode="python")
    payload["insertion_events"] = payload["insertion_events"][:1]
    with pytest.raises(ValidationError, match="rsk_trace_event_positions"):
        RSKWordTraceResult.model_validate(payload)
