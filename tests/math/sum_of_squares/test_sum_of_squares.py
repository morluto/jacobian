"""Tests for sum-of-squares operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.sum_of_squares._models import (
    GramCertificateRequest,
    GramCertificateResult,
    SOSDecompositionCheckRequest,
    SOSDecompositionCheckResult,
)
from jacobian.math.sum_of_squares._operations import (
    check_gram_certificate,
    check_sos_decomposition,
)


def _poly(variables, *terms):
    return RationalPolynomial.model_validate(
        {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": str(n), "den": str(d)},
                        "exponents": list(e),
                    }
                    for n, d, e in terms
                ]
            },
        }
    )


class TestSOSDecompositionCheck:
    """Test sum-of-squares decomposition checking."""

    def test_valid_decomposition(self):
        """x^2 + 1 = x^2 + 1^2 is a valid SOS decomposition."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        q2 = _poly(("x",), (1, 1, (0,)))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_invalid_decomposition(self):
        """x^2 + 1 ≠ x^2 alone."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1,))
        )
        assert not result.is_valid

    def test_two_variable_decomposition(self):
        """x^2 + y^2 = x^2 + y^2 is valid."""
        p = _poly(("x", "y"), (1, 1, (2, 0)), (1, 1, (0, 2)))
        q1 = _poly(("x", "y"), (1, 1, (1, 0)))
        q2 = _poly(("x", "y"), (1, 1, (0, 1)))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_cross_term_decomposition(self):
        """(x+y)^2 + (x-y)^2 = 2x^2 + 2y^2 is valid."""
        p = _poly(("x", "y"), (2, 1, (2, 0)), (2, 1, (0, 2)))
        q1 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        q2 = _poly(("x", "y"), (1, 1, (1, 0)), (-1, 1, (0, 1)))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_ring_mismatch_rejected(self):
        """Summands must use the same ring as the polynomial."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("y",), (1, 1, (1,)))
        with pytest.raises(ValueError, match="same ring"):
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1,))

    def test_single_summand(self):
        """x^2 = (x)^2 is valid."""
        p = _poly(("x",), (1, 1, (2,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1,))
        )
        assert result.is_valid


class TestGramCertificateAdmission:
    """Gram certificates bound every matrix coefficient before PSD work."""

    def _request(self, gram_entries):
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        return GramCertificateRequest(
            polynomial=p,
            monomial_basis=basis,
            gram_matrix=gram_entries,
        )

    @staticmethod
    def _entry(num: str, den: str = "1") -> dict:
        return {"num": num, "den": den}

    def test_valid_certificate(self):
        result = check_gram_certificate(
            self._request(
                (
                    (self._entry("1"), self._entry("0")),
                    (self._entry("0"), self._entry("1")),
                )
            )
        )
        assert result.is_valid
        assert result.is_symmetric
        assert result.reconstructs_polynomial
        assert result.is_psd

    def test_oversized_matrix_coefficient_rejected_before_eigenvalues(self):
        huge = "9" * 129
        with pytest.raises(
            ValueError, match="gram_matrix coefficient exceeds digit bound"
        ):
            self._request(
                (
                    (self._entry(huge), self._entry("0")),
                    (self._entry("0"), self._entry("1")),
                )
            )

    def test_boundary_coefficient_admitted(self):
        edge = "9" * 128
        request = self._request(
            (
                (self._entry(edge), self._entry("0")),
                (self._entry("0"), self._entry("1")),
            )
        )
        assert request.gram_matrix[0][0].num == edge


class TestGramCertificateResultAdmission:
    """Deserialized Gram results replay through the bounded request contract."""

    def _valid_result(self):
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        request = check_gram_certificate(
            GramCertificateRequest(
                polynomial=p,
                monomial_basis=basis,
                gram_matrix=(
                    ({"num": "1", "den": "1"}, {"num": "0", "den": "1"}),
                    ({"num": "0", "den": "1"}, {"num": "1", "den": "1"}),
                ),
            )
        )
        return request.model_dump(mode="json")

    def test_oversized_result_matrix_is_rejected_before_replay(self):
        payload = self._valid_result()
        huge = "9" * 129
        payload["gram_matrix"][0][0] = {"num": huge, "den": "1"}
        with pytest.raises(
            ValidationError, match="gram_matrix coefficient exceeds digit bound"
        ):
            GramCertificateResult.model_validate(payload)

    def test_non_monomial_basis_entry_is_rejected(self):
        payload = self._valid_result()
        shifted = RationalPolynomial.model_validate(
            {
                "polynomial_schema_version": "1",
                "domain": "QQ",
                "variables": ["x"],
                "polynomial": {
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
                    ]
                },
            }
        )
        payload["monomial_basis"] = [
            shifted.model_dump(mode="json"),
            payload["monomial_basis"][1],
        ]
        with pytest.raises(ValidationError, match="single-term monomial"):
            GramCertificateResult.model_validate(payload)


class TestGramMonomialBasisAdmission:
    def test_request_with_polynomial_basis_entry_is_rejected(self):
        p = _poly(("x",), (1, 1, (2,)), (2, 1, (1,)), (1, 1, (0,)))
        with pytest.raises(ValidationError, match="single-term monomial"):
            check_gram_certificate(
                GramCertificateRequest(
                    polynomial=p,
                    monomial_basis=(_poly(("x",), (1, 1, (1,)), (1, 1, (0,))),),
                    gram_matrix=(({"num": "1", "den": "1"},),),
                )
            )

    def test_duplicate_monomials_are_rejected(self):
        p = _poly(("x",), (2, 1, (2,)), (1, 1, (0,)))
        with pytest.raises(ValidationError, match="distinct"):
            check_gram_certificate(
                GramCertificateRequest(
                    polynomial=p,
                    monomial_basis=(
                        _poly(("x",), (1, 1, (1,))),
                        _poly(("x",), (1, 1, (1,))),
                    ),
                    gram_matrix=(
                        ({"num": "2", "den": "1"}, {"num": "0", "den": "1"}),
                        ({"num": "0", "den": "1"}, {"num": "1", "den": "1"}),
                    ),
                )
            )


class TestSOSResultAdmission:
    def test_oversized_result_summands_are_rejected_before_expansion(self):
        variables = tuple(f"x{i}" for i in range(8))
        exponent_rows = [
            tuple(1 if j == k else 0 for j in range(8)) for k in range(8)
        ] + [tuple([0] * 8)]
        wide = _poly(variables, *[(1, 1, row) for row in exponent_rows])
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        summands = tuple(_poly(("x",), (1, 1, (0,))) for _ in range(64))
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=summands)
        )
        payload = result.model_dump(mode="json")
        payload["summands"] = [wide.model_dump(mode="json")] * 64
        payload["is_valid"] = False
        with pytest.raises(
            ValidationError, match="predicted SOS expansion exceeds term bound"
        ):
            SOSDecompositionCheckResult.model_validate(payload)
