"""Tests for sum-of-squares operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.sum_of_squares._models import (
    MAX_SOS_SUMMAND_TERMS,
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


def _expand_poly(expr, variables) -> RationalPolynomial:
    """Exact sympy expansion into the domain's canonical polynomial."""
    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
    )

    poly = sympy.Poly(expr, *sympy.symbols(list(variables)), domain=sympy.QQ)
    return rational_polynomial_from_sympy(poly, variables)


def _sos_poly(*polys: RationalPolynomial) -> RationalPolynomial:
    """Exact q_1^2 + ... + q_r^2 for canonical polynomials."""
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    expr = sum(
        (rational_polynomial_to_sympy(q).as_expr() ** 2 for q in polys),
        start=0,
    )
    return _expand_poly(expr, polys[0].variables)


class TestSOSTermBudgets:
    """Target polynomials take the wider budget; squared summands stay narrow."""

    def test_wide_target_polynomial_is_admitted(self):
        """q1 = x1^3+...+x4^3 + x1+...+x8 + 1 squares to 91 distinct terms;
        q2 = x1^4+...+x8^4 squares to 36 more. The 127-term target sits
        above the 64-term summand budget but inside the 256-term target
        budget while both summands stay individually bounded."""
        names = tuple(f"x{k}" for k in range(1, 9))
        monomials = [tuple(3 if i == j else 0 for j in range(8)) for i in range(4)]
        monomials += [tuple(1 if i == j else 0 for j in range(8)) for i in range(8)]
        monomials.append(tuple(0 for _ in names))
        q1 = _poly(names, *[(1, 1, e) for e in sorted(monomials, reverse=True)])
        fourths = [tuple(4 if i == j else 0 for j in range(8)) for i in range(8)]
        q2 = _poly(names, *[(1, 1, e) for e in sorted(fourths, reverse=True)])
        assert len(q1.polynomial.terms) == 13
        wide_target = _sos_poly(q1, q2)
        assert len(wide_target.polynomial.terms) > MAX_SOS_SUMMAND_TERMS
        expanded = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=wide_target, summands=(q1, q2))
        )
        assert expanded.is_valid

    def test_wide_summand_still_rejected_by_narrow_budget(self):
        """A single 65-term summand fails the 64-term summand budget even
        though its predicted square stays inside the product cap."""
        wide = _poly(("x",), *[(1, 1, (k,)) for k in range(64, -1, -1)])
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        with pytest.raises(ValidationError, match=r"summand\[0\] exceeds the 64"):
            SOSDecompositionCheckRequest(polynomial=p, summands=(wide,))

    def test_target_above_256_terms_rejected(self):
        """The target budget is 256: a 257-term polynomial is rejected."""
        triples = [
            (a, b, c)
            for a in range(12, -1, -1)
            for b in range(12 - a, -1, -1)
            for c in range(12 - a - b, -1, -1)
        ]
        p = _poly(("x", "y", "z"), *[(1, 1, t) for t in triples[:257]])
        q = _poly(("x", "y", "z"), (1, 1, (0, 0, 0)))
        with pytest.raises(ValidationError, match="polynomial exceeds the 256"):
            SOSDecompositionCheckRequest(polynomial=p, summands=(q,))


class TestGramCertificateAdmission:
    """Gram certificates bound every matrix coefficient before PSD work."""

    def _request(self, gram_entries):
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        return GramCertificateRequest(
            polynomial=p,
            monomial_basis=basis,
            gram_matrix={
                "matrix_schema_version": "1",
                "domain": "QQ",
                "entries": gram_entries,
            },
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

    def test_produced_rational_matrix_is_accepted_unchanged(self):
        """A serialized producer {matrix_schema_version, domain, entries}
        value validates directly and its returned form enters a matrices
        rank consumer unchanged."""
        from jacobian.math.matrices._operation_models import MatrixRankRequest

        payload = {
            "matrix_schema_version": "1",
            "domain": "QQ",
            "entries": (
                (self._entry("1"), self._entry("0")),
                (self._entry("0"), self._entry("1")),
            ),
        }
        request = GramCertificateRequest.model_validate(
            {
                "polynomial": _poly(("x",), (1, 1, (2,)), (1, 1, (0,))).model_dump(
                    mode="json"
                ),
                "monomial_basis": [
                    _poly(("x",), (1, 1, (1,))).model_dump(mode="json"),
                    _poly(("x",), (1, 1, (0,))).model_dump(mode="json"),
                ],
                "gram_matrix": payload,
            }
        )
        assert request.gram_matrix.entries[0][0].num == "1"
        # The returned value is a canonical RationalMatrix that downstream
        # matrix consumers accept unchanged.
        MatrixRankRequest.model_validate(
            {"matrix": request.gram_matrix.model_dump(mode="json")}
        )

    def test_non_square_side_vs_basis_rejected(self):
        with pytest.raises(ValueError, match="square"):
            self._request(((self._entry("1"),),))

    def test_oversized_matrix_coefficient_rejected_before_eigenvalues(self):
        huge = "9" * 129
        with pytest.raises(
            ValidationError, match="gram_matrix scalars are limited to 128"
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
        assert request.gram_matrix.entries[0][0].num == edge


class TestGramCertificateResultAdmission:
    """Deserialized Gram results replay through the bounded request contract."""

    def _valid_result(self):
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        request = check_gram_certificate(
            GramCertificateRequest(
                polynomial=p,
                monomial_basis=basis,
                gram_matrix={
                    "matrix_schema_version": "1",
                    "domain": "QQ",
                    "entries": (
                        ({"num": "1", "den": "1"}, {"num": "0", "den": "1"}),
                        ({"num": "0", "den": "1"}, {"num": "1", "den": "1"}),
                    ),
                },
            )
        )
        return request.model_dump(mode="json")

    def test_oversized_result_matrix_is_rejected_before_replay(self):
        payload = self._valid_result()
        huge = "9" * 129
        payload["gram_matrix"]["entries"][0][0] = {"num": huge, "den": "1"}
        with pytest.raises(
            ValidationError, match="gram_matrix scalars are limited to 128"
        ):
            GramCertificateResult.model_validate(payload)

    def test_oversized_result_dimension_is_rejected_at_field_validation(self):
        """A 40x40 result matrix fails the parse-time dimension bound before
        any invariant replay traverses its entries."""
        payload = self._valid_result()
        payload["monomial_basis"] = [
            _poly(("x",), (1, 1, (k,))).model_dump(mode="json") for k in range(40)
        ]
        payload["polynomial"] = _poly(
            ("x",), *[(1, 1, (2 * k,)) for k in range(39, -1, -1)]
        ).model_dump(mode="json")
        zero_row = tuple({"num": "0", "den": "1"} for _ in range(40))
        payload["gram_matrix"] = {
            "matrix_schema_version": "1",
            "domain": "QQ",
            "entries": [zero_row for _ in range(40)],
        }
        with pytest.raises(ValidationError, match="at most 32"):
            GramCertificateResult.model_validate(payload)

    def test_oversized_result_basis_is_rejected_at_field_validation(self):
        """A basis longer than the Gram dimension bound is rejected at the
        field level, before admission scans any coefficient."""
        payload = self._valid_result()
        payload["monomial_basis"] = [
            _poly(("x",), (1, 1, (k,))).model_dump(mode="json") for k in range(33)
        ]
        with pytest.raises(ValidationError, match="at most 32"):
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
                    gram_matrix={"entries": (({"num": "1", "den": "1"},),)},
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
                    gram_matrix={
                        "entries": (
                            ({"num": "2", "den": "1"}, {"num": "0", "den": "1"}),
                            ({"num": "0", "den": "1"}, {"num": "1", "den": "1"}),
                        )
                    },
                )
            )


class TestSOSResultAdmission:
    def test_oversized_result_summands_are_rejected_before_expansion(self):
        exponent_rows = [(k,) for k in range(8, 0, -1)] + [(0,)]
        wide = _poly(("x",), *[(1, 1, row) for row in exponent_rows])
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


class TestSOSResultRingAdmission:
    def test_result_replay_rejects_mismatched_summand_ring(self):
        """A serialized result whose summand uses another ring is rejected at
        the typed boundary instead of leaking a SymPy coercion exception."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        summands = (_poly(("x",), (1, 1, (0,))),)
        result = check_sos_decomposition(
            SOSDecompositionCheckRequest(polynomial=p, summands=summands)
        )
        payload = result.model_dump(mode="json")
        payload["summands"] = [_poly(("y",), (1, 1, (0,))).model_dump(mode="json")]
        payload["is_valid"] = False
        payload["computed_sum"] = _poly(("x",), (1, 1, (2,)), (1, 1, (0,))).model_dump(
            mode="json"
        )
        with pytest.raises(ValidationError, match="same ring as the polynomial"):
            SOSDecompositionCheckResult.model_validate(payload)


class TestExactPsdCriterion:
    """The Gram PSD test is total: no backend exception on any input."""

    def _request(self, gram_entries):
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        return GramCertificateRequest(
            polynomial=p,
            monomial_basis=basis,
            gram_matrix={"entries": gram_entries},
        )

    def test_irreducible_characteristic_polynomial_is_decided(self) -> None:
        """A 5x5 PD tridiagonal matrix whose characteristic quintic is
        irreducible over QQ: eigenvals() raises MatrixError, the exact
        symmetric-elimination criterion must still decide PSD."""

        def entry(value: int) -> dict:
            return {"num": str(value), "den": "1"}

        rows = [
            [10, 1, 0, 0, 0],
            [1, 12, 1, 0, 0],
            [0, 1, 15, 1, 0],
            [0, 0, 1, 19, 1],
            [0, 0, 0, 1, 24],
        ]
        gram = tuple(tuple(entry(v) for v in row) for row in rows)
        result = check_gram_certificate(
            GramCertificateRequest(
                polynomial=_poly(
                    ("x",),
                    (1, 1, (2,)),
                    (1, 1, (0,)),
                ),
                monomial_basis=tuple(_poly(("x",), (1, 1, (k,))) for k in range(5)),
                gram_matrix={"entries": gram},
            )
        )
        assert result.is_psd is True

    def test_indefinite_matrix_is_decided_not_psd(self) -> None:
        def entry(num: str) -> dict:
            return {"num": num, "den": "1"}

        gram = ((entry("-1"),),)
        result = check_gram_certificate(
            GramCertificateRequest(
                polynomial=_poly(("x",), (-1, 1, (2,))),
                monomial_basis=(_poly(("x",), (1, 1, (1,))),),
                gram_matrix={"entries": gram},
            )
        )
        assert result.is_psd is False


class TestSOSCoefficientGrowthAdmission:
    """Admission replays the exact expansion so over-canonical computed sums
    fail parsing instead of raising inside execution."""

    def test_aligned_prime_denominators_rejected_at_admission(self) -> None:
        """64 eight-term summands meet the 4,096-product cap while their
        121-digit denominators align onto one output coefficient; the
        reduced denominator far exceeds the canonical limit, so admission —
        not execution — must reject."""

        def summand(k: int):
            # Eight terms with pairwise-distinct large prime-like
            # denominators, all squaring down onto low degrees.
            terms = [
                {
                    "coefficient": {
                        "num": "1",
                        "den": str(10**120 + 64 * e + 2 * k + 3),
                    },
                    "exponents": [7 - e],
                }
                for e in range(8)
            ]
            return RationalPolynomial.model_validate(
                {
                    "polynomial_schema_version": "1",
                    "domain": "QQ",
                    "variables": ["x"],
                    "polynomial": {"terms": terms},
                }
            )

        summands = tuple(summand(k) for k in range(64))
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        with pytest.raises(ValidationError):
            SOSDecompositionCheckRequest(polynomial=p, summands=summands)
