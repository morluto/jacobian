"""Morphism-algebra slice (#3712): apply, compose, powers, iterates, lengths."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.words import (
    FiniteWord,
    WordMorphism,
    apply_morphism,
    compose_morphisms,
    iterate_morphism,
    morphism_image_lengths,
    morphism_power,
)
from jacobian.math.logic.languages.words._models import (
    MorphismApplyRequest,
    MorphismApplyResult,
    MorphismComposeRequest,
    MorphismImageLengthsRequest,
    MorphismImageLengthsResult,
    MorphismIterateRequest,
    MorphismPowerRequest,
)
from jacobian.math.logic.languages.words._tools import (
    compute_morphism_apply,
    compute_morphism_compose,
    compute_morphism_image_lengths,
    compute_morphism_iterate,
    compute_morphism_power,
)


def _fib() -> WordMorphism:
    return WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("a", "b"),
        images=(("a", "b"), ("a",)),
    )


def test_apply_known_answer() -> None:
    result = compute_morphism_apply(
        MorphismApplyRequest(
            morphism=_fib(),
            word=FiniteWord(alphabet=("a", "b"), letters=("a", "b")),
        )
    )
    assert result.image.letters == ("a", "b", "a")
    assert result.image.alphabet == ("a", "b")
    # Defining invariant: replay through the image map.
    assert result.image == apply_morphism(result.morphism, result.word)


def test_compose_known_answer() -> None:
    swap = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("x", "y"),
        images=(("y",), ("x",)),
    )
    result = compute_morphism_compose(MorphismComposeRequest(first=_fib(), second=swap))
    assert result.composite.images == (("y", "x"), ("y",))
    assert result.composite.source_alphabet == ("a", "b")
    assert result.composite.target_alphabet == ("x", "y")
    # Composition replays through sequential application.
    word = FiniteWord(alphabet=("a", "b"), letters=("a", "b"))
    assert apply_morphism(result.composite, word) == apply_morphism(
        swap, apply_morphism(_fib(), word)
    )


def test_power_zero_is_identity_and_iterate_zero_is_fixed() -> None:
    power = compute_morphism_power(MorphismPowerRequest(morphism=_fib(), exponent=0))
    assert power.power.images == (("a",), ("b",))
    word = FiniteWord(alphabet=("a", "b"), letters=("a", "b"))
    iterated = compute_morphism_iterate(
        MorphismIterateRequest(morphism=_fib(), word=word, steps=0)
    )
    assert iterated.image == word


def test_power_and_iterate_agree() -> None:
    power = compute_morphism_power(MorphismPowerRequest(morphism=_fib(), exponent=2))
    assert power.power.images == (("a", "b", "a"), ("a", "b"))
    word = FiniteWord(alphabet=("a", "b"), letters=("a",))
    iterated = compute_morphism_iterate(
        MorphismIterateRequest(morphism=_fib(), word=word, steps=2)
    )
    assert iterated.image.letters == ("a", "b", "a")
    assert iterated.image == apply_morphism(power.power, word)


def test_image_lengths_bound_to_source() -> None:
    result = compute_morphism_image_lengths(
        MorphismImageLengthsRequest(morphism=_fib())
    )
    assert result.lengths == (2, 1)
    assert result.total_length == 3
    assert result.max_length == 2
    assert morphism_image_lengths(_fib()) == (2, 1)


def test_apply_rejects_axis_mismatch() -> None:
    with pytest.raises(Exception, match="source alphabet"):
        apply_morphism(_fib(), FiniteWord(alphabet=("x", "y"), letters=("x",)))


def test_compose_rejects_middle_mismatch() -> None:
    other = WordMorphism(
        source_alphabet=("x", "y"),
        target_alphabet=("x", "y"),
        images=(("x",), ("y",)),
    )
    with pytest.raises(Exception, match="must equal"):
        compose_morphisms(_fib(), other)


def test_power_rejects_non_endomorphism() -> None:
    non_endo = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("x", "y"),
        images=(("x",), ("y",)),
    )
    with pytest.raises(Exception, match="identical source"):
        morphism_power(non_endo, 2)


def test_output_budget_is_resource_refusal() -> None:
    expanding = WordMorphism(
        source_alphabet=("a",),
        target_alphabet=("a",),
        images=(("a",) * 2,),
    )
    word = FiniteWord(alphabet=("a",), letters=("a",) * 500)
    with pytest.raises(OperationResourceAdmissionError, match="exceeds"):
        apply_morphism(expanding, word)
    with pytest.raises(OperationResourceAdmissionError, match="exceeds"):
        iterate_morphism(expanding, FiniteWord(alphabet=("a",), letters=("a",)), 9)
    big = WordMorphism(
        source_alphabet=("a",),
        target_alphabet=("a",),
        images=(("a",) * 1000,),
    )
    with pytest.raises(OperationResourceAdmissionError, match="exceeds"):
        morphism_power(big, 2)


def test_serialized_apply_round_trip() -> None:
    from jacobian.canonical import encode_strict_json

    result = compute_morphism_apply(
        MorphismApplyRequest(
            morphism=_fib(),
            word=FiniteWord(alphabet=("a", "b"), letters=("a",)),
        )
    )
    restored = MorphismApplyResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_serialized_lengths_reject_forged_axis() -> None:
    from pydantic import ValidationError

    from jacobian.canonical import encode_strict_json

    result = compute_morphism_image_lengths(
        MorphismImageLengthsRequest(morphism=_fib())
    )
    payload = result.model_dump(mode="json")
    payload["lengths"] = [1, 1]
    with pytest.raises(ValidationError):
        MorphismImageLengthsResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
