"""Tests for sum-of-squares operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.sum_of_squares._models import (
    MAX_SOS_SUMMAND_TERMS,
    GramCertificateRequest,
    GramCertificateResult,
    SOSDecompositionCheckRequest,
)
from jacobian.math.polynomials.sum_of_squares.operations import (
    check_gram_certificate,
    check_sos_decomposition,
)
from jacobian.math.polynomials.values import RationalPolynomial


class RationalWire(TypedDict):
    """JSON representation accepted for one canonical rational."""

    num: str
    den: str


type GramEntries = tuple[tuple[RationalWire, ...], ...]


def _check_sos(request: SOSDecompositionCheckRequest):
    return check_sos_decomposition(request.polynomial, request.summands)


def _check_gram(request: GramCertificateRequest):
    return check_gram_certificate(
        request.polynomial, request.monomial_basis, request.gram_matrix
    )


def _poly(
    variables: Sequence[str], *terms: tuple[int, int, tuple[int, ...]]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
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

    def test_valid_decomposition(self) -> None:
        """x^2 + 1 = x^2 + 1^2 is a valid SOS decomposition."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        q2 = _poly(("x",), (1, 1, (0,)))
        result = _check_sos(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_invalid_decomposition(self) -> None:
        """x^2 + 1 ≠ x^2 alone."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        result = _check_sos(SOSDecompositionCheckRequest(polynomial=p, summands=(q1,)))
        assert not result.is_valid

    def test_two_variable_decomposition(self) -> None:
        """x^2 + y^2 = x^2 + y^2 is valid."""
        p = _poly(("x", "y"), (1, 1, (2, 0)), (1, 1, (0, 2)))
        q1 = _poly(("x", "y"), (1, 1, (1, 0)))
        q2 = _poly(("x", "y"), (1, 1, (0, 1)))
        result = _check_sos(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_cross_term_decomposition(self) -> None:
        """(x+y)^2 + (x-y)^2 = 2x^2 + 2y^2 is valid."""
        p = _poly(("x", "y"), (2, 1, (2, 0)), (2, 1, (0, 2)))
        q1 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        q2 = _poly(("x", "y"), (1, 1, (1, 0)), (-1, 1, (0, 1)))
        result = _check_sos(
            SOSDecompositionCheckRequest(polynomial=p, summands=(q1, q2))
        )
        assert result.is_valid

    def test_ring_mismatch_rejected(self) -> None:
        """Summands must use the same ring as the polynomial."""
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        q1 = _poly(("y",), (1, 1, (1,)))
        request = SOSDecompositionCheckRequest(polynomial=p, summands=(q1,))
        with pytest.raises(OperationDomainValidationError, match="same ring"):
            _check_sos(request)

    def test_single_summand(self) -> None:
        """x^2 = (x)^2 is valid."""
        p = _poly(("x",), (1, 1, (2,)))
        q1 = _poly(("x",), (1, 1, (1,)))
        result = _check_sos(SOSDecompositionCheckRequest(polynomial=p, summands=(q1,)))
        assert result.is_valid

    def test_empty_decomposition_is_the_canonical_zero_sum(self) -> None:
        """The zero polynomial has the empty sum-of-squares decomposition."""
        zero = _poly(("x",))
        result = _check_sos(SOSDecompositionCheckRequest(polynomial=zero, summands=()))
        assert result.is_valid
        assert result.summands == ()
        assert result.computed_sum == zero

    def test_empty_decomposition_does_not_certify_a_nonzero_polynomial(self) -> None:
        """The same degenerate witness is rejected by the exact identity check."""
        nonzero = _poly(("x",), (1, 1, (0,)))
        result = _check_sos(
            SOSDecompositionCheckRequest(polynomial=nonzero, summands=())
        )
        assert not result.is_valid
        assert result.computed_sum.polynomial.terms == ()


def _expand_poly(expr: Any, variables: tuple[str, ...]) -> RationalPolynomial:
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

    def test_wide_target_polynomial_is_admitted(self) -> None:
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
        expanded = _check_sos(
            SOSDecompositionCheckRequest(polynomial=wide_target, summands=(q1, q2))
        )
        assert expanded.is_valid

    def test_wide_summand_still_rejected_by_narrow_budget(self) -> None:
        """A single 65-term summand fails the 64-term summand budget even
        though its predicted square stays inside the product cap."""
        wide = _poly(("x",), *[(1, 1, (k,)) for k in range(64, -1, -1)])
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        request = SOSDecompositionCheckRequest(polynomial=p, summands=(wide,))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _check_sos(request)
        assert exc_info.value.errors()[0]["type"] == "sum_of_squares.term_bound"

    def test_target_above_256_terms_rejected(self) -> None:
        """The target budget is 256: a 257-term polynomial is rejected."""
        triples = [
            (a, b, c)
            for a in range(12, -1, -1)
            for b in range(12 - a, -1, -1)
            for c in range(12 - a - b, -1, -1)
        ]
        p = _poly(("x", "y", "z"), *[(1, 1, t) for t in triples[:257]])
        q = _poly(("x", "y", "z"), (1, 1, (0, 0, 0)))
        request = SOSDecompositionCheckRequest(polynomial=p, summands=(q,))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _check_sos(request)
        assert exc_info.value.errors()[0]["type"] == "sum_of_squares.term_bound"


class TestGramCertificateAdmission:
    """Gram certificates bound every matrix coefficient before PSD work."""

    def _request(self, gram_entries: GramEntries) -> GramCertificateRequest:
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        return GramCertificateRequest.model_validate(
            {
                "polynomial": p.model_dump(mode="json"),
                "monomial_basis": [item.model_dump(mode="json") for item in basis],
                "gram_matrix": {
                    "domain": "QQ",
                    "entries": gram_entries,
                },
            }
        )

    @staticmethod
    def _entry(num: str, den: str = "1") -> RationalWire:
        return {"num": num, "den": den}

    def test_valid_certificate(self) -> None:
        result = _check_gram(
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

    def test_produced_rational_matrix_is_accepted_unchanged(self) -> None:
        """A serialized producer {domain, entries}
        value validates directly and its returned form enters a matrices
        rank consumer unchanged."""
        from jacobian.math.matrices._operation_models import MatrixRankRequest

        payload = {
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

    def test_non_square_side_vs_basis_rejected(self) -> None:
        request = self._request(((self._entry("1"),),))
        with pytest.raises(OperationDomainValidationError, match="square"):
            _check_gram(request)

    def test_oversized_matrix_coefficient_rejected_before_eigenvalues(self) -> None:
        huge = "9" * 129
        request = self._request(
            (
                (self._entry(huge), self._entry("0")),
                (self._entry("0"), self._entry("1")),
            )
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _check_gram(request)
        assert exc_info.value.errors()[0]["type"] == "sum_of_squares.coefficient_bound"

    def test_boundary_coefficient_admitted(self) -> None:
        edge = "9" * 128
        request = self._request(
            (
                (self._entry(edge), self._entry("0")),
                (self._entry("0"), self._entry("1")),
            )
        )
        assert request.gram_matrix.entries[0][0].num == edge


class TestGramCertificateResultStructure:
    """Deserialized Gram results retain their bounded field shapes."""

    def _valid_result(self) -> dict[str, Any]:
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        request = _check_gram(
            GramCertificateRequest.model_validate(
                {
                    "polynomial": p.model_dump(mode="json"),
                    "monomial_basis": [item.model_dump(mode="json") for item in basis],
                    "gram_matrix": {
                        "domain": "QQ",
                        "entries": (
                            (
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ),
                            (
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ),
                        ),
                    },
                }
            )
        )
        return request.model_dump(mode="json")

    def test_oversized_result_dimension_is_rejected_at_field_validation(self) -> None:
        """A 40x40 result matrix fails the parse-time dimension bound before
        any explicit verification traverses its entries."""
        payload = self._valid_result()
        payload["monomial_basis"] = [
            _poly(("x",), (1, 1, (k,))).model_dump(mode="json") for k in range(40)
        ]
        payload["polynomial"] = _poly(
            ("x",), *[(1, 1, (2 * k,)) for k in range(39, -1, -1)]
        ).model_dump(mode="json")
        zero_row = tuple({"num": "0", "den": "1"} for _ in range(40))
        payload["gram_matrix"] = {
            "domain": "QQ",
            "entries": [zero_row for _ in range(40)],
        }
        with pytest.raises(ValidationError) as exc_info:
            GramCertificateResult.model_validate(payload)
        assert exc_info.value.errors()[0]["type"] == "too_long"

    def test_oversized_result_basis_is_rejected_at_field_validation(self) -> None:
        """A basis longer than the Gram dimension bound is rejected at the
        field level, before admission scans any coefficient."""
        payload = self._valid_result()
        payload["monomial_basis"] = [
            _poly(("x",), (1, 1, (k,))).model_dump(mode="json") for k in range(33)
        ]
        with pytest.raises(ValidationError) as exc_info:
            GramCertificateResult.model_validate(payload)
        assert exc_info.value.errors()[0]["type"] == "too_long"


class TestGramMonomialBasisAdmission:
    def test_request_with_polynomial_basis_entry_is_rejected(self) -> None:
        p = _poly(("x",), (1, 1, (2,)), (2, 1, (1,)), (1, 1, (0,)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _check_gram(
                GramCertificateRequest.model_validate(
                    {
                        "polynomial": p.model_dump(mode="json"),
                        "monomial_basis": [
                            _poly(("x",), (1, 1, (1,)), (1, 1, (0,))).model_dump(
                                mode="json"
                            )
                        ],
                        "gram_matrix": {"entries": (({"num": "1", "den": "1"},),)},
                    }
                )
            )
        assert exc_info.value.errors()[0]["type"] == "sum_of_squares.basis_monomial"

    def test_duplicate_monomials_are_rejected(self) -> None:
        p = _poly(("x",), (2, 1, (2,)), (1, 1, (0,)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            _check_gram(
                GramCertificateRequest.model_validate(
                    {
                        "polynomial": p.model_dump(mode="json"),
                        "monomial_basis": [
                            _poly(("x",), (1, 1, (1,))).model_dump(mode="json"),
                            _poly(("x",), (1, 1, (1,))).model_dump(mode="json"),
                        ],
                        "gram_matrix": {
                            "entries": (
                                (
                                    {"num": "2", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ),
                                (
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ),
                            )
                        },
                    }
                )
            )
        assert exc_info.value.errors()[0]["type"] == "sum_of_squares.basis_distinct"


class TestExactPsdCriterion:
    """The Gram PSD test is total: no backend exception on any input."""

    def _request(self, gram_entries: GramEntries) -> GramCertificateRequest:
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        basis = (_poly(("x",), (1, 1, (1,))), _poly(("x",), (1, 1, (0,))))
        return GramCertificateRequest.model_validate(
            {
                "polynomial": p.model_dump(mode="json"),
                "monomial_basis": [item.model_dump(mode="json") for item in basis],
                "gram_matrix": {"entries": gram_entries},
            }
        )

    def test_irreducible_characteristic_polynomial_is_decided(self) -> None:
        """A 5x5 PD tridiagonal matrix whose characteristic quintic is
        irreducible over QQ: eigenvals() raises MatrixError, the exact
        symmetric-elimination criterion must still decide PSD."""

        def entry(value: int) -> RationalWire:
            return {"num": str(value), "den": "1"}

        rows = [
            [10, 1, 0, 0, 0],
            [1, 12, 1, 0, 0],
            [0, 1, 15, 1, 0],
            [0, 0, 1, 19, 1],
            [0, 0, 0, 1, 24],
        ]
        gram = tuple(tuple(entry(v) for v in row) for row in rows)
        result = _check_gram(
            GramCertificateRequest.model_validate(
                {
                    "polynomial": _poly(
                        ("x",),
                        (1, 1, (2,)),
                        (1, 1, (0,)),
                    ).model_dump(mode="json"),
                    "monomial_basis": [
                        _poly(("x",), (1, 1, (k,))).model_dump(mode="json")
                        for k in range(5)
                    ],
                    "gram_matrix": {"entries": gram},
                }
            )
        )
        assert result.is_psd is True

    def test_indefinite_matrix_is_decided_not_psd(self) -> None:
        def entry(num: str) -> RationalWire:
            return {"num": num, "den": "1"}

        gram = ((entry("-1"),),)
        result = _check_gram(
            GramCertificateRequest.model_validate(
                {
                    "polynomial": _poly(("x",), (-1, 1, (2,))).model_dump(mode="json"),
                    "monomial_basis": [
                        _poly(("x",), (1, 1, (1,))).model_dump(mode="json")
                    ],
                    "gram_matrix": {"entries": gram},
                }
            )
        )
        assert result.is_psd is False


class TestSOSCoefficientGrowthAdmission:
    """Admission bounds coefficient growth without expanding the summands."""

    def test_aligned_prime_denominators_rejected_at_admission(self) -> None:
        """64 eight-term summands meet the 4,096-product cap while their
        121-digit denominators align onto one output coefficient; the
        reduced denominator far exceeds the canonical limit, so admission —
        not execution — must reject."""

        def summand(k: int) -> RationalPolynomial:
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
                    "domain": "QQ",
                    "variables": ["x"],
                    "polynomial": {"terms": terms},
                }
            )

        summands = tuple(summand(k) for k in range(64))
        p = _poly(("x",), (1, 1, (2,)), (1, 1, (0,)))
        request = SOSDecompositionCheckRequest(polynomial=p, summands=summands)
        with pytest.raises(OperationDomainValidationError):
            _check_sos(request)
