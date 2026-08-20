"""Tests for quadratic form operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.quadratic_forms._models import (
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


def test_evaluate_identity() -> None:
    result = evaluate_form(
        EvaluationRequest(form={"matrix": [[1, 0], [0, 1]]}, vector=[3, 4])
    )
    assert result.value == 25
    assert result.dimension == 2


def test_evaluate_diagonal() -> None:
    result = evaluate_form(
        EvaluationRequest(form={"matrix": [[2, 0], [0, 3]]}, vector=[1, 1])
    )
    assert result.value == 5


def test_evaluate_cross_terms() -> None:
    result = evaluate_form(
        EvaluationRequest(form={"matrix": [[1, 1], [1, 1]]}, vector=[1, 2])
    )
    assert result.value == 9


def test_discriminant_identity() -> None:
    result = compute_discriminant(
        DiscriminantRequest(form={"matrix": [[1, 0], [0, 1]]})
    )
    assert result.discriminant == 1


def test_discriminant_diagonal() -> None:
    result = compute_discriminant(
        DiscriminantRequest(form={"matrix": [[2, 0], [0, 3]]})
    )
    assert result.discriminant == 6


def test_signature_positive_definite() -> None:
    result = compute_signature(SignatureRequest(form={"matrix": [[1, 0], [0, 1]]}))
    assert result.n_positive == 2
    assert result.is_positive_definite is True


def test_signature_indefinite() -> None:
    result = compute_signature(SignatureRequest(form={"matrix": [[1, 0], [0, -1]]}))
    assert result.n_positive == 1
    assert result.n_negative == 1
    assert result.is_indefinite is True


def test_signature_irrational_eigenvalues() -> None:
    """Matrix [[1,1],[1,2]] has eigenvalues (3±√5)/2, both positive.
    The old int() truncation misclassified (3-√5)/2 ≈ 0.382 as zero."""
    request = SignatureRequest(form=SymmetricMatrix(matrix=((1, 1), (1, 2))))
    result = compute_signature(request)
    assert result.n_positive == 2
    assert result.n_negative == 0
    assert result.n_zero == 0
    assert result.is_positive_definite is True


def test_signature_negative_definite() -> None:
    result = compute_signature(SignatureRequest(form={"matrix": [[-1, 0], [0, -1]]}))
    assert result.is_negative_definite is True


def test_non_symmetric_rejected() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        SignatureRequest(form={"matrix": [[1, 2], [0, 1]]})


class TestRepresentationNumbers:
    """Tests for representation numbers."""

    def test_identity_1d(self) -> None:
        """q(x) = x^2: r(0)=2, r(1)=2, r(2)=0, r(4)=2."""
        from jacobian.math.quadratic_forms._models import (
            RepresentationNumbersRequest,
            SymmetricMatrix,
        )
        from jacobian.math.quadratic_forms._operations import compute_representation_numbers

        form = SymmetricMatrix(matrix=((1,),))
        result = compute_representation_numbers(
            RepresentationNumbersRequest(form=form, bound=5)
        )
        # r(0) = 1 (x=0), r(1) = 2 (x=±1), r(4) = 2 (x=±2)
        assert result.counts[0] == 1
        assert result.counts[1] == 2
        assert result.counts[4] == 2
        assert result.counts[5] == 0

    def test_identity_2d(self) -> None:
        """q(x,y) = x^2 + y^2: r(0)=1, r(1)=4, r(2)=4."""
        from jacobian.math.quadratic_forms._models import (
            RepresentationNumbersRequest,
            SymmetricMatrix,
        )
        from jacobian.math.quadratic_forms._operations import compute_representation_numbers

        form = SymmetricMatrix(matrix=((1, 0), (0, 1)))
        result = compute_representation_numbers(
            RepresentationNumbersRequest(form=form, bound=5)
        )
        assert result.counts[0] == 1  # (0,0)
        assert result.counts[1] == 4  # (±1,0), (0,±1)
        assert result.counts[2] == 4  # (±1,±1)


class TestScaling:
    """Tests for form scaling."""

    def test_scale_identity(self) -> None:
        from jacobian.math.quadratic_forms._models import ScalingRequest, SymmetricMatrix
        from jacobian.math.quadratic_forms._operations import compute_scaling

        form = SymmetricMatrix(matrix=((1, 0), (0, 1)))
        result = compute_scaling(ScalingRequest(form=form, factor=3))
        assert result.scaled_form.matrix == ((3, 0), (0, 3))


class TestDirectSum:
    """Tests for direct sum."""

    def test_direct_sum_1d_1d(self) -> None:
        from jacobian.math.quadratic_forms._models import (
            DirectSumRequest,
            SymmetricMatrix,
        )
        from jacobian.math.quadratic_forms._operations import compute_direct_sum

        form1 = SymmetricMatrix(matrix=((1,),))
        form2 = SymmetricMatrix(matrix=((2,),))
        result = compute_direct_sum(DirectSumRequest(form1=form1, form2=form2))
        assert result.direct_sum.matrix == ((1, 0), (0, 2))

    def test_direct_sum_2d_1d(self) -> None:
        from jacobian.math.quadratic_forms._models import (
            DirectSumRequest,
            SymmetricMatrix,
        )
        from jacobian.math.quadratic_forms._operations import compute_direct_sum

        form1 = SymmetricMatrix(matrix=((1, 0), (0, 1)))
        form2 = SymmetricMatrix(matrix=((5,),))
        result = compute_direct_sum(DirectSumRequest(form1=form1, form2=form2))
        assert result.direct_sum.matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 5))
