"""Canonical matrices over one exact real simple-number-field embedding."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.analysis._models import InertiaResult
from jacobian.math.matrices.analysis.operations import compute_inertia
from jacobian.math.matrices.values import EmbeddedRealSimpleNumberFieldMatrix
from jacobian.math.number_theory.number_fields import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    embeddings,
)


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _element(
    presentation: SimpleNumberFieldPresentation,
    *coefficients: int | Fraction,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(_rational(value) for value in coefficients),
    )


def test_embedding_profile_composes_with_a_common_embedding_matrix() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    profile = embeddings(presentation)
    embedding = profile.records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)

    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=(
            (_element(presentation, 1, 0, 0, 0), _element(presentation, 0, 1, 0, 0)),
            (_element(presentation, 0, 0, 0, 1), _element(presentation, 0, 0, 1, 0)),
        ),
    )
    replayed = EmbeddedRealSimpleNumberFieldMatrix.model_validate_json(
        encode_strict_json(matrix.model_dump(mode="json")),
        strict=True,
    )

    assert replayed == matrix
    assert replayed.embedding.root.real_root_index == 1
    assert replayed.domain == "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"


def test_common_embedding_matrix_rejects_a_foreign_entry_presentation() -> None:
    quartic = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    quadratic = SimpleNumberFieldPresentation(coefficients_descending=("1", "0", "-2"))
    embedding = embeddings(quartic).records[1].embedding

    with pytest.raises(ValidationError) as exc_info:
        EmbeddedRealSimpleNumberFieldMatrix(
            embedding=embedding,
            entries=((_element(quadratic, 1, 0),),),
        )

    assert exc_info.value.errors()[0]["type"] == "matrix.embedding_presentation"


def test_exact_inertia_distinguishes_the_two_real_quartic_embeddings() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    profile = embeddings(presentation)
    alpha = _element(presentation, 0, 1, 0, 0)
    negative_matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=profile.records[0].embedding,
        entries=((alpha,),),
    )
    positive_matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=profile.records[1].embedding,
        entries=((alpha,),),
    )

    negative = compute_inertia(negative_matrix)
    positive = compute_inertia(positive_matrix)

    assert (negative.n_positive, negative.n_negative, negative.n_zero) == (0, 1, 0)
    assert (positive.n_positive, positive.n_negative, positive.n_zero) == (1, 0, 0)
    assert (
        InertiaResult.model_validate_json(
            encode_strict_json(positive.model_dump(mode="json")),
            strict=True,
        )
        == positive
    )


def test_exact_algebraic_inertia_eliminates_a_hyperbolic_plane_with_tail() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "-2")
    )
    embedding = embeddings(presentation).records[1].embedding
    zero = _element(presentation, 0, 0)
    alpha = _element(presentation, 0, 1)
    one = _element(presentation, 1, 0)
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=(
            (zero, alpha, one),
            (alpha, zero, one),
            (one, one, zero),
        ),
    )

    result = compute_inertia(matrix)

    assert (result.n_positive, result.n_negative, result.n_zero) == (1, 2, 0)


def test_inertia_rejects_a_structural_embedding_with_no_selected_real_root() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "1")
    )
    embedding = RealNumberFieldEmbedding.model_validate(
        {
            "kind": "REAL",
            "presentation": presentation.model_dump(mode="json"),
            "root": {
                "polynomial": ["1", "0", "1"],
                "real_root_index": 0,
            },
        }
    )
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=((_element(presentation, 1, 0),),),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_inertia(matrix)

    assert exc_info.value.errors()[0]["type"] == "matrix.invalid_embedding"
