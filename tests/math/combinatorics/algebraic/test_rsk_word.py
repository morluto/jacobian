"""Exact forward and inverse RSK contracts for finite ordered words."""

from __future__ import annotations

import itertools
import json
from collections import Counter

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics import algebraic as algebraic_combinatorics
from jacobian.math.combinatorics.algebraic import (
    inverse_row_insertion_rsk,
    row_insertion_rsk,
)
from jacobian.math.combinatorics.algebraic._models import (
    MAX_RSK_PERMUTATION_LENGTH,
    HookLengthRequest,
    RSKInverseWordRequest,
    RSKPermutationRequest,
    RSKResult,
    RSKWordRequest,
)
from jacobian.math.combinatorics.algebraic._tools import (
    TOOLS,
    hook_lengths,
    inverse_rsk_word,
    rsk_permutation,
    rsk_word,
)
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_BYTES,
    MAX_RSK_WORD_LENGTH,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    require_standard,
)
from jacobian.math.combinatorics.symmetric_functions._models import PartitionRequest
from jacobian.math.logic.languages.words import FiniteWord
from jacobian.math.logic.languages.words._models import PeriodsRequest
from jacobian.math.logic.languages.words.values import (
    Symbol,
    _require_unicode_scalar_string,
)


def _word(
    letters: tuple[str, ...], alphabet: tuple[str, ...] = ("a", "b", "c", "d")
) -> FiniteWord:
    return FiniteWord(alphabet=alphabet, letters=letters)


def _pair(word: FiniteWord) -> RSKTableauPair:
    return rsk_word(RSKWordRequest(word=word))


@pytest.mark.parametrize(
    ("word", "p_rows", "q_rows", "shape"),
    [
        (_word((), ("a",)), (), (), ()),
        (_word(("a",), ("a",)), ((1,),), ((1,),), (1,)),
        (
            _word(("a", "b", "c"), ("a", "b", "c")),
            ((1, 2, 3),),
            ((1, 2, 3),),
            (3,),
        ),
        (
            _word(("c", "b", "a"), ("a", "b", "c")),
            ((1,), (2,), (3,)),
            ((1,), (2,), (3,)),
            (1, 1, 1),
        ),
        (
            _word(("a", "a", "a"), ("a",)),
            ((1, 1, 1),),
            ((1, 2, 3),),
            (3,),
        ),
        (
            _word(("c", "c", "b", "d", "a")),
            ((1, 3, 4), (2,), (3,)),
            ((1, 2, 4), (3,), (5,)),
            (3, 1, 1),
        ),
    ],
)
def test_row_insertion_known_values(
    word: FiniteWord,
    p_rows: tuple[tuple[int, ...], ...],
    q_rows: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
) -> None:
    pair = _pair(word)
    assert pair.insertion_tableau.rows == p_rows
    assert pair.recording_tableau.rows == q_rows
    assert pair.shape.parts == shape
    assert pair.convention == "ROW_INSERTION_RSK_V1"
    assert pair.source_kind == "WORD"
    assert inverse_row_insertion_rsk(pair) == word


def test_plactic_normal_form_is_canonical_row_reading_and_knuth_invariant() -> None:
    alphabet = ("1", "2", "3")
    source = FiniteWord(alphabet=alphabet, letters=("1", "3", "2"))
    equivalent = FiniteWord(alphabet=alphabet, letters=("3", "1", "2"))

    result = algebraic_combinatorics.plactic_normal_form(source)
    moved = algebraic_combinatorics.plactic_normal_form(equivalent)

    assert result.insertion_tableau.rows == ((1, 2), (3,))
    assert result.normal_form.letters == ("3", "1", "2")
    assert result.insertion_tableau == moved.insertion_tableau
    assert result.normal_form == moved.normal_form
    assert (
        algebraic_combinatorics.row_insertion_rsk(result.normal_form).insertion_tableau
        == result.insertion_tableau
    )


def test_plactic_normal_form_empty_word_keeps_its_empty_alphabet() -> None:
    result = algebraic_combinatorics.plactic_normal_form(
        FiniteWord(alphabet=(), letters=())
    )

    assert result.source_word == FiniteWord(alphabet=(), letters=())
    assert result.normal_form == FiniteWord(alphabet=(), letters=())
    assert result.insertion_tableau.rows == ()


def test_plactic_normal_form_matches_independent_row_insertion_on_small_words() -> None:
    alphabet = ("z", "a", "m")

    def reference_insertion(letters: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
        ranks = {letter: index + 1 for index, letter in enumerate(alphabet)}
        rows: list[list[int]] = []
        for letter in letters:
            bumped = ranks[letter]
            row_index = 0
            while True:
                if row_index == len(rows):
                    rows.append([bumped])
                    break
                row = rows[row_index]
                column = next(
                    (index for index, value in enumerate(row) if value > bumped),
                    len(row),
                )
                if column == len(row):
                    row.append(bumped)
                    break
                row[column], bumped = bumped, row[column]
                row_index += 1
        return tuple(tuple(row) for row in rows)

    for length in range(5):
        for letters in itertools.product(alphabet, repeat=length):
            source = FiniteWord(alphabet=alphabet, letters=letters)
            result = algebraic_combinatorics.plactic_normal_form(source)
            expected_rows = reference_insertion(letters)
            assert result.insertion_tableau.rows == expected_rows
            expected_reading = tuple(
                alphabet[entry - 1] for row in reversed(expected_rows) for entry in row
            )
            assert result.normal_form.letters == expected_reading
            assert reference_insertion(result.normal_form.letters) == expected_rows


def test_first_strictly_greater_rule_preserves_repeated_letters_in_a_row() -> None:
    pair = _pair(_word(("b", "b", "a"), ("a", "b")))
    assert pair.insertion_tableau.rows == ((1, 2), (2,))
    assert pair.recording_tableau.rows == ((1, 2), (3,))


def test_pair_content_and_recording_labels_reconstruct_the_source() -> None:
    word = _word(("c", "a", "c", "b", "a"))
    pair = _pair(word)
    rank = {letter: index for index, letter in enumerate(word.alphabet, start=1)}
    assert Counter(entry for row in pair.insertion_tableau.rows for entry in row) == (
        Counter(rank[letter] for letter in word.letters)
    )
    assert sorted(
        entry for row in pair.recording_tableau.rows for entry in row
    ) == list(range(1, len(word.letters) + 1))
    assert inverse_row_insertion_rsk(pair) == word


def test_explicit_alphabet_order_is_interpretation_critical() -> None:
    increasing = _pair(_word(("a", "b"), ("a", "b")))
    decreasing = _pair(_word(("a", "b"), ("b", "a")))
    assert increasing.shape.parts == (2,)
    assert decreasing.shape.parts == (1, 1)


def test_order_preserving_relabelling_transports_only_the_alphabet() -> None:
    first = _pair(_word(("b", "a", "c", "b"), ("a", "b", "c")))
    relabelled = _pair(_word(("y", "x", "z", "y"), ("x", "y", "z")))
    assert first.insertion_tableau == relabelled.insertion_tableau
    assert first.recording_tableau == relabelled.recording_tableau
    assert first.shape == relabelled.shape
    assert first.alphabet != relabelled.alphabet


def test_forward_output_feeds_inverse_without_representation_repair() -> None:
    source = _word(("c", "c", "b", "d", "a"))
    produced = rsk_word(RSKWordRequest(word=source))
    consumed = RSKInverseWordRequest.model_validate({"pair": produced.model_dump()})
    result = inverse_rsk_word(consumed)
    assert result == source
    assert row_insertion_rsk(result) == produced

    partition_request = PartitionRequest.model_validate(
        {"partition": produced.shape.model_dump()}
    )
    assert partition_request.partition == produced.shape
    assert HookLengthRequest(partition=produced.shape).partition is produced.shape
    expected_hooks = (
        (5, 2, 1),
        (2,),
        (1,),
    )
    assert (
        hook_lengths(HookLengthRequest(partition=produced.shape)).hooks
        == expected_hooks
    )
    assert algebraic_combinatorics.hook_lengths(produced.shape) == expected_hooks
    assert algebraic_combinatorics.conjugate_partition(produced.shape).parts == (
        3,
        1,
        1,
    )
    assert algebraic_combinatorics.standard_young_tableaux_count(produced.shape) == 6


def test_inverse_result_chains_into_word_requests_without_rebuilding() -> None:
    source = _word(("c", "a", "c", "b", "a"))
    pair = rsk_word(RSKWordRequest(word=source))
    result = inverse_rsk_word(
        RSKInverseWordRequest.model_validate({"pair": pair.model_dump()})
    )
    assert result == source
    assert PeriodsRequest(word=result).word == source
    assert FiniteWord.model_validate(result.model_dump()) == source


def test_all_short_ternary_words_round_trip_both_directions() -> None:
    alphabet = ("a", "b", "c")
    for length in range(7):
        for letters in itertools.product(alphabet, repeat=length):
            word = FiniteWord(alphabet=alphabet, letters=letters)
            pair = row_insertion_rsk(word)
            reconstructed = inverse_row_insertion_rsk(pair)
            assert reconstructed == word
            assert row_insertion_rsk(reconstructed) == pair


def test_forward_pair_matches_independent_scan_oracle_exhaustively() -> None:
    """Compare both tableaux to direct first-greater row scans.

    Round trips alone can let mutually consistent forward/reverse errors pass,
    so this checks P and Q against a small independent insertion oracle.
    """
    alphabet = ("a", "b", "c")

    def reference_pair(letters: tuple[str, ...]):
        insertion: list[list[int]] = []
        recording: list[list[int]] = []
        ranks = {letter: index + 1 for index, letter in enumerate(alphabet)}
        for position, letter in enumerate(letters, start=1):
            carried = ranks[letter]
            row_index = 0
            while row_index < len(insertion):
                row = insertion[row_index]
                column = next(
                    (index for index, value in enumerate(row) if value > carried),
                    len(row),
                )
                if column == len(row):
                    row.append(carried)
                    recording[row_index].append(position)
                    break
                row[column], carried = carried, row[column]
                row_index += 1
            else:
                insertion.append([carried])
                recording.append([position])
        return tuple(map(tuple, insertion)), tuple(map(tuple, recording))

    for length in range(7):
        for letters in itertools.product(alphabet, repeat=length):
            source = FiniteWord(alphabet=alphabet, letters=letters)
            pair = row_insertion_rsk(source)
            expected_p, expected_q = reference_pair(letters)
            assert pair.insertion_tableau.rows == expected_p
            assert pair.recording_tableau.rows == expected_q
            assert pair.shape.parts == tuple(map(len, expected_p))
            assert inverse_row_insertion_rsk(pair) == source


def test_permutation_operation_agrees_with_word_specialization() -> None:
    permutation = (3, 1, 4, 2)
    old_result = rsk_permutation(RSKPermutationRequest(permutation=permutation))
    word_pair = _pair(
        FiniteWord(
            alphabet=("1", "2", "3", "4"),
            letters=tuple(str(entry) for entry in permutation),
        )
    )
    assert old_result.p_tableau.rows == word_pair.insertion_tableau.rows
    assert old_result.q_tableau.rows == word_pair.recording_tableau.rows
    assert old_result.shape == word_pair.shape


def test_permutation_inversion_swaps_the_tableaux() -> None:
    for permutation in itertools.permutations(range(1, 5)):
        inverse = [0] * len(permutation)
        for position, value in enumerate(permutation, start=1):
            inverse[value - 1] = position
        pair = rsk_permutation(RSKPermutationRequest(permutation=permutation))
        inverse_pair = rsk_permutation(
            RSKPermutationRequest(permutation=tuple(inverse))
        )
        assert pair.p_tableau == inverse_pair.q_tableau
        assert pair.q_tableau == inverse_pair.p_tableau


def test_permutation_envelope_is_derived_from_the_canonical_cell_budget() -> None:
    assert MAX_RSK_PERMUTATION_LENGTH == MAX_RSK_WORD_LENGTH

    identity_51 = tuple(range(1, 52))
    result = rsk_permutation(RSKPermutationRequest(permutation=identity_51))
    assert result.shape.parts == (51,)
    assert result.lis_length == 51
    assert result.lds_length == 1

    identity_at_cap = tuple(range(1, MAX_RSK_PERMUTATION_LENGTH + 1))
    wide = rsk_permutation(RSKPermutationRequest(permutation=identity_at_cap))
    assert wide.p_tableau.rows == (identity_at_cap,)
    assert wide.q_tableau.rows == (identity_at_cap,)
    assert wide.shape.parts == (MAX_RSK_PERMUTATION_LENGTH,)
    assert RSKResult.model_validate(wide.model_dump()) == wide

    descending_at_cap = tuple(range(MAX_RSK_PERMUTATION_LENGTH, 0, -1))
    deep = rsk_permutation(RSKPermutationRequest(permutation=descending_at_cap))
    assert deep.shape.parts == (1,) * MAX_RSK_PERMUTATION_LENGTH
    assert deep.lis_length == 1
    assert deep.lds_length == MAX_RSK_PERMUTATION_LENGTH
    assert RSKResult.model_validate(deep.model_dump()) == deep

    with pytest.raises(ValidationError):
        RSKPermutationRequest(
            permutation=tuple(range(1, MAX_RSK_PERMUTATION_LENGTH + 2))
        )


def test_structurally_incompatible_pairs_fail_before_reverse_insertion() -> None:
    with pytest.raises(ValidationError) as error:
        RSKTableauPair(
            alphabet=("a", "b"),
            insertion_tableau=SemistandardYoungTableau(rows=((1, 2),)),
            recording_tableau=StandardYoungTableau(rows=((1, 2),)),
            shape=IntegerPartition(parts=(1, 1)),
        )
    assert (
        error.value.errors()[0]["type"]
        == "algebraic_combinatorics.insertion_shape_mismatch"
    )

    pair = RSKTableauPair(
        alphabet=("a",),
        insertion_tableau=SemistandardYoungTableau(rows=((2,),)),
        recording_tableau=StandardYoungTableau(rows=((1,),)),
        shape=IntegerPartition(parts=(1,)),
    )
    with pytest.raises(ValueError, match="outside the ordered alphabet"):
        inverse_row_insertion_rsk(pair)

    transported = RSKInverseWordRequest(pair=pair.model_copy(update={"alphabet": ()}))
    with pytest.raises(OperationDomainValidationError) as domain_error:
        inverse_rsk_word(transported)
    assert domain_error.value.errors()[0]["type"] == (
        "algebraic_combinatorics.rsk_pair_incompatible"
    )

    pair = RSKTableauPair.model_validate(
        {
            "alphabet": ["a"],
            "insertion_tableau": {"rows": [[1, 1]]},
            "recording_tableau": {"rows": [[1, 3]]},
            "shape": {"parts": [2]},
        }
    )
    with pytest.raises(Exception, match="exactly 1 through n"):
        require_standard(pair.recording_tableau)


def _wide_unicode_symbols(count: int) -> tuple[str, ...]:
    return tuple("\U0001f600" * 63 + chr(0x1F600 + index) for index in range(count))


def test_word_length_and_utf8_payload_bounds_are_closed() -> None:
    length_boundary = FiniteWord(alphabet=("a",), letters=("a",) * MAX_RSK_WORD_LENGTH)
    assert sum(_pair(length_boundary).shape.parts) == MAX_RSK_WORD_LENGTH

    too_long = FiniteWord.model_construct(
        alphabet=("a",), letters=("a",) * (MAX_RSK_WORD_LENGTH + 1)
    )
    with pytest.raises(
        ValueError, match=rf"length must not exceed {MAX_RSK_WORD_LENGTH}"
    ):
        rsk_word(RSKWordRequest(word=too_long))
    with pytest.raises(
        ValueError, match=rf"length must not exceed {MAX_RSK_WORD_LENGTH}"
    ):
        row_insertion_rsk(too_long)
    with pytest.raises(ValidationError):
        FiniteWord(alphabet=("a",), letters=("a",) * (MAX_RSK_WORD_LENGTH + 1))

    ordered_symbols = tuple(f"s{index:02d}" for index in range(50))
    deepest_shape = _pair(
        FiniteWord(
            alphabet=ordered_symbols,
            letters=tuple(
                symbol for symbol in reversed(ordered_symbols) for _ in (0, 1)
            ),
        )
    )
    assert deepest_shape.shape.parts == (2,) * 50

    with pytest.raises(ValidationError):
        FiniteWord(
            alphabet=tuple(f"s{index:02d}" for index in range(51)),
            letters=(),
        )

    boundary_symbols = _wide_unicode_symbols(50)
    per_symbol_bytes = len(boundary_symbols[0].encode("utf-8"))
    boundary_letter_count = MAX_RSK_WORD_BYTES // per_symbol_bytes - len(
        boundary_symbols
    )
    byte_boundary = FiniteWord(
        alphabet=boundary_symbols,
        letters=(boundary_symbols[0],) * boundary_letter_count,
    )
    assert RSKWordRequest(word=byte_boundary).word == byte_boundary

    assert (
        sum(len(symbol.encode("utf-8")) for symbol in boundary_symbols)
        + sum(len(letter.encode("utf-8")) for letter in byte_boundary.letters)
        == MAX_RSK_WORD_BYTES
    )


def test_canonical_tableau_cell_bound_admits_boundary_pairs_end_to_end() -> None:
    single_row = _pair(
        FiniteWord(alphabet=("a",), letters=("a",) * MAX_RSK_WORD_LENGTH)
    )
    assert single_row.insertion_tableau.rows == ((1,) * MAX_RSK_WORD_LENGTH,)
    assert single_row.recording_tableau.rows == (
        tuple(range(1, MAX_RSK_WORD_LENGTH + 1)),
    )
    assert single_row.shape.parts == (MAX_RSK_WORD_LENGTH,)
    assert inverse_row_insertion_rsk(single_row) == FiniteWord(
        alphabet=("a",), letters=("a",) * MAX_RSK_WORD_LENGTH
    )

    reconstructed = inverse_rsk_word(
        RSKInverseWordRequest.model_validate({"pair": single_row.model_dump()})
    )
    assert reconstructed == FiniteWord(
        alphabet=("a",), letters=("a",) * MAX_RSK_WORD_LENGTH
    )

    wide = tuple(f"s{index:02d}" for index in range(50))
    descending_pairs = tuple(symbol for symbol in reversed(wide) for _ in range(10))
    pair = _pair(FiniteWord(alphabet=wide, letters=descending_pairs))
    assert pair.shape.parts == (10,) * 50
    assert inverse_row_insertion_rsk(pair) == FiniteWord(
        alphabet=wide, letters=descending_pairs
    )


@pytest.mark.parametrize(
    "symbol",
    [
        "\ud800",
        "\udfff",
        "a\ud800b",
        "\U0001d400\ud800",
    ],
)
def test_lone_surrogate_symbols_are_rejected_at_request_admission(
    symbol: str,
) -> None:
    with pytest.raises(ValidationError):
        RSKWordRequest.model_validate(
            {"word": {"alphabet": [symbol], "letters": [symbol]}}
        )
    with pytest.raises(ValidationError):
        RSKInverseWordRequest.model_validate(
            {
                "pair": {
                    "alphabet": [symbol],
                    "insertion_tableau": {"rows": [[1]]},
                    "recording_tableau": {"rows": [[1]]},
                    "shape": {"parts": [1]},
                }
            }
        )


def test_surrogate_json_decode_reaches_typed_request_rejection() -> None:
    raw_word = json.loads('{"alphabet": ["\\ud800"], "letters": ["\\ud800"]}')
    with pytest.raises(ValidationError):
        RSKWordRequest(word=FiniteWord.model_validate(raw_word))

    raw_pair = json.loads(
        '{"alphabet": ["\\ud800"], "insertion_tableau": {"rows": [[1]]}, '
        '"recording_tableau": {"rows": [[1]]}, "shape": {"parts": [1]}}'
    )
    with pytest.raises(ValidationError):
        RSKInverseWordRequest(pair=RSKTableauPair.model_validate(raw_pair))


def test_unicode_scalar_validator_rejects_surrogates_and_admits_astral_symbols() -> (
    None
):
    for symbol in ("\ud800", "\udfff", "a\ud800b"):
        with pytest.raises(ValueError, match="Unicode scalar values"):
            _require_unicode_scalar_string(symbol)

    assert _require_unicode_scalar_string("\U0001d400") == "\U0001d400"
    assert TypeAdapter(Symbol).validate_python("\U0001d401\U0001d402")


def test_astral_scalar_symbols_round_trip_through_both_directions() -> None:
    alphabet = ("\U0001d400", "\U0001d401", "\U0001d402")
    word = FiniteWord(alphabet=alphabet, letters=(alphabet[2], alphabet[0]))
    pair = row_insertion_rsk(word)
    assert pair.alphabet == alphabet
    assert inverse_row_insertion_rsk(pair) == word


def test_kernel_converts_unencodable_payloads_into_request_validation() -> None:
    fabricated = FiniteWord.model_construct(alphabet=("\ud800",), letters=("\ud800",))
    with pytest.raises(ValueError, match="Unicode scalar values"):
        row_insertion_rsk(fabricated)


def test_comparison_bound_boundary_word_round_trips() -> None:
    letters = ("b",) * 64 + ("a",)
    word = FiniteWord(alphabet=("a", "b"), letters=letters)
    pair = row_insertion_rsk(word)
    assert pair.insertion_tableau.rows == ((1,) + (2,) * 63, (2,))
    assert pair.recording_tableau.rows == (tuple(range(1, 65)), (65,))
    assert pair.shape.parts == (64, 1)
    assert inverse_row_insertion_rsk(pair) == word


def test_rsk_request_schema_publishes_convention_and_work_envelope() -> None:
    assert MAX_RSK_ROW_SEARCH_COMPARISONS == 9
    assert MAX_RSK_WORD_BYTES == 140_800

    schema = RSKWordRequest.model_json_schema()
    assert schema["properties"]["convention"]["const"] == "ROW_INSERTION_RSK_V1"
    description = schema["properties"]["word"]["description"]
    assert "unique strings" in description
    assert "every positioned letter" in description
    assert f"at most {MAX_RSK_WORD_LENGTH} letters" in description
    assert f"{MAX_RSK_WORD_BYTES} UTF-8 bytes" in description
    class_description = schema["description"]
    assert "2N" in class_description
    assert (
        str(MAX_RSK_WORD_LENGTH * (MAX_RSK_WORD_LENGTH - 1) // 2) in class_description
    )
    assert f"{MAX_RSK_ROW_SEARCH_COMPARISONS} integer comparisons" in class_description
    assert "comparisons per search" in class_description
    assert MAX_RSK_WORD_LENGTH.bit_length() == MAX_RSK_ROW_SEARCH_COMPARISONS

    inverse_schema = RSKInverseWordRequest.model_json_schema()
    pair_schema = inverse_schema["$defs"]["RSKTableauPair"]
    assert f"at most {MAX_RSK_WORD_LENGTH} cells" in pair_schema["description"]
    assert f"{MAX_RSK_WORD_LENGTH} cells" in inverse_schema["description"]
    assert (
        f"at most {MAX_RSK_WORD_LENGTH} cells"
        in pair_schema["properties"]["shape"]["description"]
    )


def test_public_operations_are_admitted_and_examples_execute() -> None:
    public_ids = {
        "tableau.rsk.word.compute",
        "tableau.rsk.inverse_word.compute",
    }
    tools = {tool.operation_id: tool for tool in TOOLS}
    assert public_ids <= tools.keys()

    for operation_id in public_ids:
        tool = tools[operation_id]
        for operation_example in tool.examples:
            request = tool.request_type.model_validate(operation_example.input)
            tool.result_type.model_validate(tool.run(request))


def test_strict_lds_matches_independent_subsequence_enumeration() -> None:
    from itertools import combinations

    from jacobian.math.combinatorics.algebraic import longest_decreasing_subsequence
    from jacobian.math.combinatorics.algebraic._models import (
        LongestDecreasingSubsequenceRequest,
    )

    alphabet = ("a", "b", "c")
    for length in range(7):
        for letters in itertools.product(alphabet, repeat=length):
            source = FiniteWord(alphabet=alphabet, letters=letters)
            result = longest_decreasing_subsequence(
                LongestDecreasingSubsequenceRequest(word=source)
            )
            feasible_lengths = [
                len(indices)
                for size in range(length + 1)
                for indices in combinations(range(length), size)
                if all(
                    letters[left] > letters[right]
                    for left, right in itertools.pairwise(indices)
                )
            ]
            assert result.length == max(feasible_lengths, default=0)
            assert result.values == tuple(letters[index] for index in result.indices)
