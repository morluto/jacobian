"""Tests for quadratic form operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.quadratic_forms._models import (
    MAX_ENTRY_DIGITS,
    MAX_VECTOR_DIGITS,
    DiscriminantRequest,
    EvaluationRequest,
    SignatureRequest,
    SymmetricMatrix,
)
from jacobian.math.quadratic_forms._operations import (
    compute_discriminant,
    compute_signature,
    evaluate_form,
)


def _form(matrix: list[list[int]]) -> dict[str, list[list[str]]]:
    return {"matrix": [[str(entry) for entry in row] for row in matrix]}


def _vector(vector: list[int]) -> list[str]:
    return [str(entry) for entry in vector]


def test_evaluate_identity() -> None:
    result = evaluate_form(
        EvaluationRequest(form=_form([[1, 0], [0, 1]]), vector=_vector([3, 4]))
    )
    assert result.value == "25"
    assert result.dimension == 2


def test_evaluate_diagonal() -> None:
    result = evaluate_form(
        EvaluationRequest(form=_form([[2, 0], [0, 3]]), vector=_vector([1, 1]))
    )
    assert result.value == "5"


def test_evaluate_cross_terms() -> None:
    result = evaluate_form(
        EvaluationRequest(form=_form([[1, 1], [1, 1]]), vector=_vector([1, 2]))
    )
    assert result.value == "9"


def test_discriminant_identity() -> None:
    result = compute_discriminant(DiscriminantRequest(form=_form([[1, 0], [0, 1]])))
    assert result.discriminant == "1"


def test_discriminant_diagonal() -> None:
    result = compute_discriminant(DiscriminantRequest(form=_form([[2, 0], [0, 3]])))
    assert result.discriminant == "6"


def test_signature_positive_definite() -> None:
    result = compute_signature(SignatureRequest(form=_form([[1, 0], [0, 1]])))
    assert result.n_positive == 2
    assert result.is_positive_definite is True


def test_signature_indefinite() -> None:
    result = compute_signature(SignatureRequest(form=_form([[1, 0], [0, -1]])))
    assert result.n_positive == 1
    assert result.n_negative == 1
    assert result.is_indefinite is True


def test_signature_irrational_eigenvalues() -> None:
    """Matrix [[1,1],[1,2]] has eigenvalues (3±√5)/2, both positive.
    The old int() truncation misclassified (3-√5)/2 ≈ 0.382 as zero."""
    request = SignatureRequest(form=SymmetricMatrix(**_form([[1, 1], [1, 2]])))
    result = compute_signature(request)
    assert result.n_positive == 2
    assert result.n_negative == 0
    assert result.n_zero == 0
    assert result.is_positive_definite is True


def test_signature_negative_definite() -> None:
    result = compute_signature(SignatureRequest(form=_form([[-1, 0], [0, -1]])))
    assert result.is_negative_definite is True


def test_signature_exact_quartic_root_signs() -> None:
    matrix = [
        [2, -1, -3, 1, -3],
        [-1, 1, 2, 3, -3],
        [-3, 2, 2, 1, 0],
        [1, 3, 1, 2, -1],
        [-3, -3, 0, -1, 1],
    ]
    result = compute_signature(SignatureRequest(form=_form(matrix)))
    assert (result.n_positive, result.n_negative, result.n_zero) == (3, 2, 0)
    assert result.is_indefinite is True


def test_signature_singular_semidefinite() -> None:
    result = compute_signature(SignatureRequest(form=_form([[1, 1], [1, 1]])))
    assert (result.n_positive, result.n_negative, result.n_zero) == (1, 0, 1)


def test_signature_counts_negative_root_sharing_square_free_factor_with_zero() -> None:
    result = compute_signature(SignatureRequest(form=_form([[-1, 0], [0, 0]])))
    assert (result.n_positive, result.n_negative, result.n_zero) == (0, 1, 1)


def test_non_symmetric_rejected() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        SignatureRequest(form=_form([[1, 2], [0, 1]]))


def test_quadratic_integer_wire_values_are_canonical_strings() -> None:
    with pytest.raises(ValidationError, match="string"):
        EvaluationRequest(form={"matrix": [[1]]}, vector=["1"])
    with pytest.raises(ValidationError, match="string"):
        EvaluationRequest(form={"matrix": [["1"]]}, vector=[1])


def test_quadratic_integer_digit_bounds_reject_immediately_above_boundary() -> None:
    with pytest.raises(ValidationError, match="matrix entries"):
        SymmetricMatrix(matrix=(("1" + "0" * MAX_ENTRY_DIGITS,),))
    with pytest.raises(ValidationError, match="vector entries"):
        EvaluationRequest(
            form={"matrix": [["1"]]},
            vector=["1" + "0" * MAX_VECTOR_DIGITS],
        )
