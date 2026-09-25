from __future__ import annotations

from functools import cache
from itertools import pairwise, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics import algebraic as algebraic_combinatorics
from jacobian.math.combinatorics.algebraic import compute_greene_witnesses
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.combinatorics.algebraic.biword import (
    GreeneWitnessRequest,
    GreeneWitnessResult,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def _maximum_disjoint_monotone_total(
    ranks: tuple[int, ...], k: int, *, weakly_increasing: bool
) -> int:
    """Independent exhaustive oracle over all subsequence position sets."""

    n = len(ranks)
    valid_masks = []
    for mask in range(1, 1 << n):
        positions = tuple(index for index in range(n) if mask & (1 << index))
        values = tuple(ranks[index] for index in positions)
        if all(
            left <= right if weakly_increasing else left > right
            for left, right in pairwise(values)
        ):
            valid_masks.append(mask)

    @cache
    def best(available: int, slots: int) -> int:
        if slots == 0:
            return 0
        optimum = best(available, slots - 1)
        for mask in valid_masks:
            if mask & available == mask:
                optimum = max(
                    optimum,
                    mask.bit_count() + best(available ^ mask, slots - 1),
                )
        return optimum

    return best((1 << n) - 1, k)


def test_greene_witnesses_match_exhaustive_disjoint_subsequence_oracle() -> None:
    alphabet = ("z", "a", "m")
    rank = {letter: index for index, letter in enumerate(alphabet)}
    for n in range(6):
        for letters in product(alphabet, repeat=n):
            word = FiniteWord(alphabet=alphabet, letters=letters)
            ranks = tuple(rank[letter] for letter in letters)
            result = compute_greene_witnesses(GreeneWitnessRequest(word=word, k=3))
            for family in result.families:
                assert family.increasing_total == _maximum_disjoint_monotone_total(
                    ranks, family.k, weakly_increasing=True
                )
                assert family.decreasing_total == _maximum_disjoint_monotone_total(
                    ranks, family.k, weakly_increasing=False
                )
                for paths, total, weak in (
                    (
                        family.increasing_subsequences,
                        family.increasing_total,
                        True,
                    ),
                    (
                        family.decreasing_subsequences,
                        family.decreasing_total,
                        False,
                    ),
                ):
                    assert len(paths) <= family.k
                    assert sum(map(len, paths)) == total
                    memberships = [index for path in paths for index in path]
                    assert len(memberships) == len(set(memberships))
                    for path in paths:
                        assert tuple(sorted(path)) == path
                        values = tuple(ranks[index] for index in path)
                        assert all(
                            left <= right if weak else left > right
                            for left, right in pairwise(values)
                        )


def test_greene_witness_envelope_rejects_before_path_expansion() -> None:
    word = FiniteWord(alphabet=("a",), letters=("a",) * 33)
    with pytest.raises(OperationResourceAdmissionError):
        compute_greene_witnesses(GreeneWitnessRequest(word=word, k=2))


def test_greene_witness_envelope_accepts_exact_word_and_k_bounds() -> None:
    word = FiniteWord(alphabet=("a",), letters=("a",) * 32)

    result = compute_greene_witnesses(GreeneWitnessRequest(word=word, k=8))

    assert result.shape.parts == (32,)
    assert result.families[-1].increasing_total == 32
    assert result.families[-1].decreasing_total == 8


def test_greene_witness_operation_is_exported_and_catalogued() -> None:
    assert "compute_greene_witnesses" in algebraic_combinatorics.__all__
    assert any(tool.operation_id == "word.greene_witnesses.compute" for tool in TOOLS)


def test_kernel_builds_its_result_through_the_trusted_producer_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = GreeneWitnessResult.model_construct

    def record(**values: object) -> GreeneWitnessResult:
        calls.append(values)
        return original(**values)

    monkeypatch.setattr(GreeneWitnessResult, "model_construct", record)
    word = FiniteWord(alphabet=("z", "a", "m"), letters=("m", "z", "a", "m"))

    result = compute_greene_witnesses(GreeneWitnessRequest(word=word, k=2))

    assert len(calls) == 1
    trusted = calls[0]
    assert trusted["word"] == result.word
    assert trusted["shape"] == result.shape
    assert trusted["families"] == result.families
    assert result.families[-1].increasing_total == 4
    assert result.families[-1].decreasing_total == 3
    assert result == GreeneWitnessResult.model_validate_json(result.model_dump_json())


def test_deserialization_still_replays_the_witness_relations() -> None:
    word = FiniteWord(alphabet=("z", "a", "m"), letters=("z", "a", "m"))
    result = compute_greene_witnesses(GreeneWitnessRequest(word=word, k=2))
    payload = result.model_dump(mode="json")
    payload["families"][0]["increasing_total"] += 1

    with pytest.raises(ValidationError) as error:
        GreeneWitnessResult.model_validate(payload)
    assert error.value.errors()[0]["type"] == (
        "algebraic_combinatorics.greene_witness_total_mismatch"
    )
