from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH,
    FreeAlgebraWord,
    FreeAlgebraWordPairRequest,
    FreeAlgebraWordPowerRequest,
    FreeAlgebraWordRequest,
    FreeAlgebraWordSubstitution,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import (
    compare_words,
    concatenate_words,
    power_word,
    reverse_word,
    substitute_word,
    word_factors,
    word_overlaps,
    word_prefixes,
    word_suffixes,
)

ALPHABET = ("a", "b", "c")


def word(*letters: str, alphabet: tuple[str, ...] = ALPHABET) -> FreeAlgebraWord:
    return FreeAlgebraWord(alphabet=alphabet, letters=letters)


def test_concatenation_is_associative_with_unit_and_returns_offsets() -> None:
    samples = [word(), word("a"), word("b", "a"), word("c", "a", "b")]
    for a, b, c in itertools.product(samples, repeat=3):
        left = concatenate_words(concatenate_words(a, b).product, c).product
        right = concatenate_words(a, concatenate_words(b, c).product).product
        assert left == right
    result = concatenate_words(word("a", "b"), word("c"))
    assert result.product.letters == ("a", "b", "c")
    assert result.left_range == (0, 2)
    assert result.right_range == (2, 3)
    assert result.degree_addition == (2, 1, 3)


def test_concatenation_producer_word_composes_with_non_growing_consumers() -> None:
    source = word(*("a",) * 32)
    product = concatenate_words(source, source).product
    assert product.length == 64
    assert product.letters == source.letters + source.letters
    assert reverse_word(product).reverse.letters == ("a",) * 64
    assert len(word_prefixes(product).splits) == MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1
    assert len(word_suffixes(product).splits) == MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1
    assert word_factors(product).factors[-1].letters == ("a",) * 64


def test_power_producer_word_composes_with_serialized_reverse_consumer() -> None:
    # Review thread: reversing the canonical result of a maximal power failed
    # only because the shared word request applied the 32-letter source bound.
    power = power_word(word("c"), 64).power
    assert compare_words(power, power).comparison == 0
    round_trip = FreeAlgebraWord.model_validate_json(power.model_dump_json())
    assert round_trip.length == 64
    request = FreeAlgebraWordRequest(word=round_trip)
    assert request.word.length == MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    result = reverse_word(request.word)
    assert result.reverse.letters == ("c",) * 64
    assert result.reverse.alphabet == ALPHABET


def test_power_zero_and_bounded_output() -> None:
    value = word("a", "b")
    zero = power_word(value, 0)
    assert zero.power.is_unit
    assert zero.degree_multiplication == (2, 0, 0)
    assert power_word(word("c"), 64).power.letters == ("c",) * 64
    with pytest.raises(OperationResourceAdmissionError) as error:
        power_word(value, 33)
    assert error.value.errors()[0]["type"] == "free_algebra.word_power_output_length"


def test_growing_operations_keep_32_letter_source_bound() -> None:
    producer = word(*("a",) * (MAX_FREE_ALGEBRA_WORD_LENGTH + 1))
    assert producer.length == MAX_FREE_ALGEBRA_WORD_LENGTH + 1
    # Non-growing consumers admit the 33-letter canonical value.
    assert FreeAlgebraWordRequest(word=producer).word.length == 33
    # Growing power and concatenation sources keep the 32-letter expansion bound.
    with pytest.raises(ValidationError):
        FreeAlgebraWordPowerRequest(word=producer, exponent=1)
    with pytest.raises(ValidationError):
        FreeAlgebraWordPairRequest(left=producer, right=word())
    with pytest.raises(OperationDomainValidationError) as error:
        concatenate_words(producer, word())
    assert error.value.errors()[0]["type"] == "free_algebra.word_source_length"
    with pytest.raises(OperationDomainValidationError) as error:
        power_word(producer, 2)
    assert error.value.errors()[0]["type"] == "free_algebra.word_source_length"


def test_reverse_is_involution_and_anti_homomorphism() -> None:
    samples = [word(), word("a"), word("a", "b"), word("c", "a", "b")]
    for a, b in itertools.product(samples, repeat=2):
        assert reverse_word(reverse_word(a).reverse).reverse == a
        reverse_product = reverse_word(concatenate_words(a, b).product).reverse
        reversed_factors = concatenate_words(
            reverse_word(b).reverse, reverse_word(a).reverse
        ).product
        assert reverse_product == reversed_factors


def test_degree_lex_comparison_uses_axis_rank_and_reports_witness() -> None:
    alphabet = ("z", "a", "q")
    values = [
        word(*letters, alphabet=alphabet)
        for degree in range(4)
        for letters in itertools.product(alphabet, repeat=degree)
    ]

    def key(value: FreeAlgebraWord) -> tuple[int, tuple[int, ...]]:
        return (
            len(value.letters),
            tuple(alphabet.index(letter) for letter in value.letters),
        )

    for left, right in itertools.product(values, repeat=2):
        result = compare_words(left, right)
        expected = (key(left) > key(right)) - (key(left) < key(right))
        assert result.comparison == expected
        assert result.degree_comparison == (
            (len(left.letters) > len(right.letters))
            - (len(left.letters) < len(right.letters))
        )
        differing = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(left.letters, right.letters, strict=False)
                )
                if a != b
            ),
            None,
        )
        assert result.first_differing_position == differing
        if differing is None:
            assert result.left_generator_rank is None
            assert result.right_generator_rank is None
        else:
            assert result.left_generator_rank == alphabet.index(left.letters[differing])
            assert result.right_generator_rank == alphabet.index(
                right.letters[differing]
            )


def test_prefix_suffix_families_include_endpoints_and_reconstruct() -> None:
    source = word("a", "b", "a")
    prefixes = word_prefixes(source)
    suffixes = word_suffixes(source)
    assert len(prefixes.splits) == len(source.letters) + 1
    assert len(suffixes.splits) == len(source.letters) + 1
    assert prefixes.splits[0].prefix_letters == ()
    assert prefixes.splits[-1].prefix_letters == source.letters
    assert suffixes.splits[0].suffix_letters == source.letters
    assert suffixes.splits[-1].suffix_letters == ()
    for split in prefixes.splits:
        assert split.prefix_letters + split.completing_suffix_letters == source.letters
    for split in suffixes.splits:
        assert split.completing_prefix_letters + split.suffix_letters == source.letters


def test_factors_are_distinct_and_include_all_occurrence_positions() -> None:
    source = word("a", "b", "a", "a")
    actual = {item.letters: item.positions for item in word_factors(source).factors}
    expected: dict[tuple[str, ...], list[int]] = {}
    for start in range(len(source.letters) + 1):
        for end in range(start, len(source.letters) + 1):
            expected.setdefault(source.letters[start:end], []).append(start)
    assert actual == {
        factor: tuple(positions) for factor, positions in expected.items()
    }
    assert len(actual) == len(set(actual))
    assert actual[()] == tuple(range(len(source.letters) + 1))


def _brute_force_ambiguities(left: tuple[str, ...], right: tuple[str, ...]):
    """Enumerate candidate common words and occurrence offsets independently."""

    if not left or not right:
        return set()
    output = set()
    for common_length in range(max(len(left), len(right)), len(left) + len(right)):
        for common in itertools.product(("a", "b"), repeat=common_length):
            for left_start in range(common_length - len(left) + 1):
                if common[left_start : left_start + len(left)] != left:
                    continue
                for right_start in range(common_length - len(right) + 1):
                    if common[right_start : right_start + len(right)] != right:
                        continue
                    if max(left_start, right_start) >= min(
                        left_start + len(left), right_start + len(right)
                    ):
                        continue
                    if left_start == right_start and len(left) == len(right):
                        continue
                    if (
                        min(left_start, right_start) != 0
                        or max(left_start + len(left), right_start + len(right))
                        != common_length
                    ):
                        continue
                    contained = (
                        left_start <= right_start
                        and right_start + len(right) <= left_start + len(left)
                    ) or (
                        right_start <= left_start
                        and left_start + len(left) <= right_start + len(right)
                    )
                    output.add(
                        (
                            common,
                            left_start,
                            right_start,
                            "INCLUSION" if contained else "OVERLAP",
                        )
                    )
    return output


def test_overlaps_match_independent_common_word_oracle_and_contexts() -> None:
    alphabet = ("a", "b")
    words = [
        letters
        for degree in range(4)
        for letters in itertools.product(alphabet, repeat=degree)
    ]
    for left_letters, right_letters in itertools.product(words, repeat=2):
        left = word(*left_letters, alphabet=alphabet)
        right = word(*right_letters, alphabet=alphabet)
        result = word_overlaps(left, right)
        actual = set()
        for witness in result.witnesses:
            common = witness.common_word.letters
            actual.add(
                (common, witness.left_offset, witness.right_offset, witness.kind)
            )
            assert witness.left_prefix + left.letters + witness.left_suffix == common
            assert witness.right_prefix + right.letters + witness.right_suffix == common
            assert (
                common[witness.left_offset : witness.left_offset + len(left.letters)]
                == left.letters
            )
            assert (
                common[witness.right_offset : witness.right_offset + len(right.letters)]
                == right.letters
            )
        assert actual == _brute_force_ambiguities(left_letters, right_letters)


def test_overlap_admits_maximum_candidate_and_inclusion_cases() -> None:
    alphabet = ("a", "b")
    left = word("a", "b", alphabet=alphabet)
    right = word("b", "a", alphabet=alphabet)
    overlap = word_overlaps(left, right).witnesses
    assert any(w.common_word.letters == ("a", "b", "a") for w in overlap)
    outer = word("a", "b", "a", alphabet=alphabet)
    inner = word("b", alphabet=alphabet)
    assert any(w.kind == "INCLUSION" for w in word_overlaps(outer, inner).witnesses)
    long = word(*(tuple("a" for _ in range(32))), alphabet=alphabet)
    assert len(word_overlaps(long, long).witnesses) == 62
    assert (
        max(len(w.common_word.letters) for w in word_overlaps(long, long).witnesses)
        == 63
    )


def test_substitution_transports_every_source_occurrence_including_empty_images() -> (
    None
):
    source_alphabet = ("a", "b")
    target_alphabet = ("x", "y")
    substitution = FreeAlgebraWordSubstitution(
        source_alphabet=source_alphabet,
        target_alphabet=target_alphabet,
        images=(
            word(alphabet=target_alphabet),
            word("x", "y", alphabet=target_alphabet),
        ),
    )
    source = word("a", "b", "a", alphabet=source_alphabet)
    result = substitute_word(substitution, source)
    assert result.image.letters == ("x", "y")
    assert tuple(
        (interval.source_index, interval.target_start, interval.target_end)
        for interval in result.occurrence_images
    ) == ((0, 0, 0), (1, 0, 2), (2, 2, 2))
    for interval, source_letter in zip(
        result.occurrence_images, source.letters, strict=True
    ):
        expected = substitution.images[source_alphabet.index(source_letter)].letters
        assert (
            result.image.letters[interval.target_start : interval.target_end]
            == expected
        )


def test_substitution_composes_and_preflights_64_letter_output() -> None:
    alphabet = ("a", "b", "c")
    endomorphism = FreeAlgebraWordSubstitution(
        source_alphabet=alphabet,
        target_alphabet=alphabet,
        images=(word("a", "b"), word("c"), word("a")),
    )
    start = word("b", "a", "c")
    mapped = substitute_word(endomorphism, start).image
    other = FreeAlgebraWordSubstitution(
        source_alphabet=alphabet,
        target_alphabet=alphabet,
        images=(word("c"), word("b", "a"), word("a")),
    )
    composed = substitute_word(other, mapped).image
    direct_image = tuple(
        letter
        for source_letter in start.letters
        for letter in substitute_word(
            other, endomorphism.images[alphabet.index(source_letter)]
        ).image.letters
    )
    assert composed.letters == direct_image

    binary = ("a", "b")
    doubling = FreeAlgebraWordSubstitution(
        source_alphabet=binary,
        target_alphabet=binary,
        images=(word(*("a",) * 32, alphabet=binary), word("b", alphabet=binary)),
    )
    output = substitute_word(doubling, word("a", "a", alphabet=binary))
    assert output.image.length == 64
    with pytest.raises(OperationResourceAdmissionError) as error:
        substitute_word(doubling, word("a", "a", "a", alphabet=binary))
    assert error.value.errors()[0]["type"] == "free_algebra.substitution_output_length"


def test_word_tool_examples_execute_through_catalog() -> None:
    operation_ids = (
        "free_word.concatenate.compute",
        "free_word.power.compute",
        "free_word.reverse.compute",
        "free_word.prefixes.compute",
        "free_word.suffixes.compute",
        "free_word.factors.compute",
        "free_word.overlaps.compute",
        "free_word.order.compare",
        "free_word.substitute.compute",
    )
    for operation_id in operation_ids:
        operation = next(tool for tool in TOOLS if tool.operation_id == operation_id)
        assert operation is not None and operation.examples
        for example in operation.examples:
            request = operation.request_type.model_validate(example.input)
            result = operation.run(request)
            assert result.model_dump(
                mode="json"
            ) == operation.result_type.model_validate(result).model_dump(mode="json")
