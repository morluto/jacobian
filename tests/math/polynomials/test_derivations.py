"""Tests for exact polynomial-derivation application."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import (
    DerivationApplyRequest,
    PolynomialDerivation,
)
from jacobian.math.polynomials.derivations.operations import apply_derivation
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...], terms: tuple[tuple[int, tuple[int, ...]], ...]
) -> RationalPolynomial:
    ordered = sorted(terms, key=lambda term: term[1], reverse=True)
    return RationalPolynomial.model_validate(
        {
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": list(exponents),
                    }
                    for coefficient, exponents in ordered
                ]
            },
        }
    )


def _derivation(
    variables: tuple[str, ...], images: tuple[RationalPolynomial, ...]
) -> PolynomialDerivation:
    return PolynomialDerivation(variables=variables, images=images)


def _terms(polynomial: RationalPolynomial) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (term.coefficient.num // term.coefficient.den, tuple(term.exponents))
        for term in polynomial.polynomial.terms
    )


XY = ("x", "y")


def _triangular() -> PolynomialDerivation:
    return _derivation(XY, (_poly(XY, ((1, (0, 1)),)), _poly(XY, ())))


class TestDerivationApplyKnownAnswers:
    def test_triangular_derivation_on_square(self) -> None:
        result = apply_derivation(_triangular(), _poly(XY, ((1, (2, 0)),)))
        assert _terms(result.result) == ((2, (1, 1)),)
        assert _terms(result.contributions[0]) == ((2, (1, 1)),)
        assert _terms(result.contributions[1]) == ()

    def test_frozen_benchmark_generator_images(self) -> None:
        result = apply_derivation(_triangular(), _poly(XY, ((1, (1, 0)),)))
        assert _terms(result.result) == ((1, (0, 1)),)
        result = apply_derivation(_triangular(), _poly(XY, ((1, (0, 1)),)))
        assert _terms(result.result) == ()

    def test_partial_derivative_projection(self) -> None:
        derivation = _derivation(XY, (_poly(XY, ((1, (0, 0)),)), _poly(XY, ())))
        result = apply_derivation(derivation, _poly(XY, ((3, (2, 1)), (5, (0, 2)))))
        assert _terms(result.result) == ((6, (1, 1)),)

    def test_zero_derivation_gives_zero(self) -> None:
        derivation = _derivation(XY, (_poly(XY, ()), _poly(XY, ())))
        result = apply_derivation(derivation, _poly(XY, ((1, (2, 0)),)))
        assert _terms(result.result) == ()

    def test_constant_polynomial_gives_zero(self) -> None:
        result = apply_derivation(_triangular(), _poly(XY, ((7, (0, 0)),)))
        assert _terms(result.result) == ()

    def test_contributions_sum_to_result(self) -> None:
        derivation = _derivation(
            XY, (_poly(XY, ((1, (0, 1)),)), _poly(XY, ((1, (1, 0)),)))
        )
        result = apply_derivation(derivation, _poly(XY, ((1, (2, 1)),)))
        assert _terms(result.result) == ((1, (3, 0)), (2, (1, 2)))
        assert _terms(result.contributions[0]) == ((2, (1, 2)),)
        assert _terms(result.contributions[1]) == ((1, (3, 0)),)


class TestDerivationInvariants:
    def test_leibniz_rule(self) -> None:
        derivation = _triangular()
        left = _poly(XY, ((1, (2, 0)),))
        right = _poly(XY, ((1, (1, 1)),))
        product = _poly(XY, ((1, (3, 1)),))

        assert _terms(apply_derivation(derivation, product).result) == ((3, (2, 2)),)
        assert _terms(apply_derivation(derivation, left).result) == ((2, (1, 1)),)
        assert _terms(apply_derivation(derivation, right).result) == ((1, (0, 2)),)

    def test_qq_linearity(self) -> None:
        derivation = _triangular()
        left = _poly(XY, ((1, (2, 0)),))
        right = _poly(XY, ((1, (0, 2)),))
        combined = _poly(XY, ((1, (2, 0)), (3, (0, 2))))

        assert _terms(apply_derivation(derivation, combined).result) == ((2, (1, 1)),)
        assert _terms(apply_derivation(derivation, left).result) == ((2, (1, 1)),)
        assert _terms(apply_derivation(derivation, right).result) == ()


class TestDerivationAdmission:
    def test_ring_mismatch_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            apply_derivation(_triangular(), _poly(("y", "x"), ((1, (1, 0)),)))

    def test_request_model_rejects_ring_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            DerivationApplyRequest(
                derivation=_triangular(), polynomial=_poly(("y", "x"), ((1, (1, 0)),))
            )

    def test_image_count_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolynomialDerivation.model_validate(
                {
                    "variables": ["x", "y"],
                    "images": [
                        {
                            "domain": "QQ",
                            "variables": ["x", "y"],
                            "polynomial": {"terms": []},
                        }
                    ],
                }
            )

    def test_published_example_validates(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.polynomials.derivations._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "polynomial_derivation.apply.compute"
        )
        example = tool.examples[0]
        request = tool.request_type.model_validate_json(
            encode_strict_json(example.input), strict=True
        )
        assert _terms(tool.run(request).result) == ((2, (1, 1)),)

    def test_contribution_budget_refused_as_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jacobian.math.polynomials.derivations import operations

        image = _poly(XY, tuple((1, (e, 0)) for e in range(64)))
        big = _poly(XY, tuple((1, (e, 64 - e)) for e in range(1, 65)))
        derivation = _derivation(XY, (image, _poly(XY, ())))
        monkeypatch.setattr(operations, "MAX_DERIVATION_CONTRIBUTION_CELLS", 10)
        with pytest.raises(OperationResourceAdmissionError):
            apply_derivation(derivation, big)
