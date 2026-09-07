"""Sparse checking scales with support, not ambient monomial degree."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.sum_of_squares._models import (
    GramCertificateRequest,
    SOSDecompositionCheckResult,
)
from jacobian.math.polynomials.sum_of_squares._tools import _check_gram
from jacobian.math.polynomials.sum_of_squares.operations import check_sos_decomposition
from jacobian.math.polynomials.values import RationalPolynomial


def _monomial(exponents: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": [f"x{i}" for i in range(len(exponents))],
            "polynomial": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": exponents}]
            },
        }
    )


@pytest.mark.parametrize("degree", [14, 108, 32768])
def test_one_term_high_degree_sos_and_invalid_reconstruction(degree: int) -> None:
    target, summand = _monomial((degree,)), _monomial((degree // 2,))
    result = check_sos_decomposition(target, (summand,))
    assert result.is_valid
    wrong = check_sos_decomposition(_monomial((degree - 1,)), (summand,))
    assert not wrong.is_valid
    assert wrong.computed_sum == target
    assert (
        SOSDecompositionCheckResult.model_validate_json(wrong.model_dump_json())
        == wrong
    )


def test_sparse_multivariate_gram_does_not_expand_ambient_degree_box() -> None:
    target = _monomial((32768,) * 8)
    basis = (_monomial((16384,) * 8),)
    request = GramCertificateRequest(
        polynomial=target,
        monomial_basis=basis,
        gram_matrix=RationalMatrix(entries=((CanonicalRational(num=1, den=1),),)),
    )
    assert _check_gram(request).is_valid


def test_unrepresentable_sos_sum_is_rejected_before_multiplication() -> None:
    with pytest.raises(OperationDomainValidationError, match="reconstruction"):
        check_sos_decomposition(_monomial((32768,)), (_monomial((16385,)),))
