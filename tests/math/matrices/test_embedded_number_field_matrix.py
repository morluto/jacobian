"""Canonical matrices over one exact real simple-number-field embedding."""

from fractions import Fraction
from threading import Event

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.analysis._models import (
    InertiaResult,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis._tools import (
    compute_inertia as compute_inertia_wire,
)
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
    round_tripped = EmbeddedRealSimpleNumberFieldMatrix.model_validate_json(
        encode_strict_json(matrix.model_dump(mode="json")),
        strict=True,
    )

    assert round_tripped == matrix
    assert round_tripped.embedding.root.real_root_index == 1
    assert round_tripped.domain == "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"


def test_common_embedding_matrix_rejects_a_foreign_entry_presentation() -> None:
    quartic = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    quadratic = SimpleNumberFieldPresentation(coefficients_descending=("1", "0", "-2"))
    embedding = embeddings(quartic).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)

    with pytest.raises(ValidationError) as exc_info:
        EmbeddedRealSimpleNumberFieldMatrix(
            embedding=embedding,
            entries=((_element(quadratic, 1, 0),),),
        )

    assert exc_info.value.errors()[0]["type"] == "matrix.embedding_presentation"


def test_raw_embedded_matrix_rejects_deep_malformed_entries_without_recursing() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "-2")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    nested: object = None
    for _ in range(1_500):
        nested = {"next": nested}

    with pytest.raises(ValidationError) as exc_info:
        EmbeddedRealSimpleNumberFieldMatrix.model_validate(
            {
                "domain": "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD",
                "embedding": embedding.model_dump(mode="json"),
                "entries": [[nested]],
            }
        )

    assert exc_info.value.errors(include_input=False)[0]["type"] == (
        "matrix.shape_mismatch"
    )


def test_exact_inertia_distinguishes_the_two_real_quartic_embeddings() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    profile = embeddings(presentation)
    alpha = _element(presentation, 0, 1, 0, 0)
    negative_embedding = profile.records[0].embedding
    positive_embedding = profile.records[1].embedding
    assert isinstance(negative_embedding, RealNumberFieldEmbedding)
    assert isinstance(positive_embedding, RealNumberFieldEmbedding)
    negative_matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=negative_embedding,
        entries=((alpha,),),
    )
    positive_matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=positive_embedding,
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

    wire_request = SymmetricMatrixRequest.model_validate_json(
        encode_strict_json({"matrix": positive.matrix.model_dump(mode="json")}),
        strict=True,
    )
    assert compute_inertia_wire(wire_request) == positive


def test_exact_sign_isolation_handles_close_quartic_power_basis_values() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    embedding = RealNumberFieldEmbedding.model_validate(
        {
            "kind": "REAL",
            "presentation": presentation.model_dump(mode="json"),
            "root": {
                "polynomial": presentation.coefficients_descending,
                "real_root_index": 1,
            },
        }
    )
    alpha_cubed_minus_five_thirds = _element(presentation, Fraction(-5, 3), 0, 0, 1)
    seven_fourths_minus_alpha_cubed = _element(presentation, Fraction(7, 4), 0, 0, -1)
    zero = _element(presentation, 0, 0, 0, 0)
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=(
            (alpha_cubed_minus_five_thirds, zero),
            (zero, seven_fourths_minus_alpha_cubed),
        ),
    )

    result = compute_inertia(matrix)

    # alpha^3 > 5/3 because 8 > (5/3)^4, while alpha^3 < 7/4 because
    # 8 < (7/4)^4. Both signs are established by exact common-root isolation.
    assert (result.n_positive, result.n_negative, result.n_zero) == (2, 0, 0)


def test_exact_sign_isolation_handles_the_admitted_degree_boundary() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "0", "0", "0", "0", "-2")
    )
    embedding = RealNumberFieldEmbedding.model_validate(
        {
            "kind": "REAL",
            "presentation": presentation.model_dump(mode="json"),
            "root": {
                "polynomial": presentation.coefficients_descending,
                "real_root_index": 1,
            },
        }
    )
    # For alpha = 2^(1/8), alpha^7 > 11/6 because 128 > (11/6)^8.
    near_positive = _element(
        presentation,
        Fraction(-11, 6),
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    )
    result = compute_inertia(
        EmbeddedRealSimpleNumberFieldMatrix(
            embedding=embedding,
            entries=((near_positive,),),
        )
    )
    assert (result.n_positive, result.n_negative, result.n_zero) == (1, 0, 0)


def test_exact_sign_isolation_handles_a_nonmonic_field_presentation() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("2", "0", "-1")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    positive = _element(presentation, -1, 2)
    negative = _element(presentation, 1, -2)
    zero = _element(presentation, 0, 0)
    result = compute_inertia(
        EmbeddedRealSimpleNumberFieldMatrix(
            embedding=embedding,
            entries=((positive, zero), (zero, negative)),
        )
    )
    assert (result.n_positive, result.n_negative, result.n_zero) == (1, 1, 0)


def test_algebraic_inertia_obeys_caller_cancellation() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "-2")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=((_element(presentation, 0, 1),),),
    )
    cancellation = Event()
    cancellation.set()
    with (
        request_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError),
    ):
        compute_inertia(matrix)


def test_exact_algebraic_inertia_eliminates_a_hyperbolic_plane_with_tail() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "-2")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
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
