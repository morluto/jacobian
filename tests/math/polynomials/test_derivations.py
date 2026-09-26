"""Tests for exact polynomial-derivation application."""

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any

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


def test_vector_field_conversion_preserves_generator_semantics() -> None:
    from jacobian.math.polynomials.derivations._tools import TOOLS
    from jacobian.math.polynomials.derivations.operations import (
        derivation_from_vector_field,
    )

    components = (_poly(XY, ((1, (0, 1)),)), _poly(XY, ()))
    derivation = derivation_from_vector_field(components)
    assert derivation.variables == XY
    assert derivation.images == components

    # Independent calculus oracle: y*d_x(x^2 + 3*y) + 0*d_y(...) = 2*x*y.
    source = _poly(XY, ((1, (2, 0)), (3, (0, 1))))
    assert _terms(apply_derivation(derivation, source).result) == ((2, (1, 1)),)

    # The binder is a copy-only projection with no postcondition beyond the
    # derivation value itself, so it stays a native helper and is not a
    # published catalog operation.
    assert all(
        tool.operation_id != "polynomial_derivation.from_vector_field.compute"
        for tool in TOOLS
    )


def test_vector_field_conversion_accepts_decoded_component_values() -> None:
    from jacobian.math.polynomials.derivations.operations import (
        derivation_from_vector_field,
    )

    derivation = derivation_from_vector_field(
        {
            "components": [
                _poly(XY, ((1, (0, 1)),)).model_dump(),
                _poly(XY, ()).model_dump(),
            ]
        }
    )
    assert derivation.variables == XY
    assert derivation.images == (_poly(XY, ((1, (0, 1)),)), _poly(XY, ()))


class _OversizedComponentSequence(Sequence[RationalPolynomial]):
    """A bounded sequence that must never be iterated past admission."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.iterated = False

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> RationalPolynomial:
        raise AssertionError("component parsed beyond the vector-field bound")

    def __iter__(self) -> Iterator[RationalPolynomial]:
        self.iterated = True
        return super().__iter__()


def test_oversized_vector_field_rejected_before_component_parsing() -> None:
    from jacobian.math.polynomials.derivations._models import (
        MAX_DERIVATION_VARIABLES,
    )
    from jacobian.math.polynomials.derivations.operations import (
        derivation_from_vector_field,
    )

    oversized = _OversizedComponentSequence(MAX_DERIVATION_VARIABLES + 1)
    with pytest.raises(OperationDomainValidationError) as error:
        derivation_from_vector_field(oversized)
    assert error.value.errors()[0]["type"] == "polynomial_derivation.vector_field_shape"
    assert not oversized.iterated


def test_vector_field_iteration_cannot_exceed_component_bound() -> None:
    from jacobian.math.polynomials.derivations.operations import (
        derivation_from_vector_field,
    )

    class _LyingSequence(Sequence[Any]):
        # A Sequence whose __len__ passes the early bound check but whose
        # iterator yields one component beyond the admitted count.
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Any:
            raise AssertionError

        def __iter__(self) -> Iterator[Any]:
            yield from (_poly(XY, ((1, (0, 1)),)).model_dump() for _ in range(8))
            yield "ninth-component"

    with pytest.raises(OperationDomainValidationError) as error:
        derivation_from_vector_field(_LyingSequence())
    assert error.value.errors()[0]["type"] == "polynomial_derivation.vector_field_shape"
    assert "bounded by 8" in error.value.errors()[0]["msg"]


@pytest.mark.parametrize(
    "invalid",
    [
        (),
        {"components": []},
        [_poly(XY, ((1, (0, 1)),))],
        [_poly(("x",), ((1, (1,)),)), _poly(("y",), ())],
        "not-a-vector-field",
        7,
        [_poly(XY, ((1, (0, 1)),)), "not-a-polynomial"],
    ],
)
def test_vector_field_invalid_shapes_raise_domain_errors(invalid: Any) -> None:
    from jacobian.math.polynomials.derivations.operations import (
        derivation_from_vector_field,
    )

    with pytest.raises(OperationDomainValidationError) as error:
        derivation_from_vector_field(invalid)
    assert error.value.errors()[0]["type"] == "polynomial_derivation.vector_field_shape"


class TestDerivationInvariants:
    def test_leibniz_rule(self) -> None:
        from jacobian.math.polynomials.derivations.operations import (
            _add,
            _encode,
            _multiply,
            _term_map,
        )

        derivation = _triangular()
        left = _poly(XY, ((1, (2, 0)),))
        right = _poly(XY, ((1, (1, 1)),))
        product = _encode(XY, _multiply(_term_map(left), _term_map(right)))
        direct = _term_map(apply_derivation(derivation, product).result)
        replay: dict[tuple[int, ...], Fraction] = {}
        _add(
            replay,
            _multiply(
                _term_map(apply_derivation(derivation, left).result), _term_map(right)
            ),
        )
        _add(
            replay,
            _multiply(
                _term_map(left), _term_map(apply_derivation(derivation, right).result)
            ),
        )
        assert direct == replay

    def test_qq_linearity(self) -> None:
        from jacobian.math.polynomials.derivations.operations import (
            _add,
            _encode,
            _multiply,
            _term_map,
        )

        derivation = _triangular()
        left = _poly(XY, ((1, (2, 0)),))
        right = _poly(XY, ((1, (0, 2)),))
        combined_terms = _term_map(left)
        _add(combined_terms, _multiply({(0, 0): Fraction(3)}, _term_map(right)))
        separate = _term_map(apply_derivation(derivation, left).result)
        _add(
            separate,
            _multiply(
                {(0, 0): Fraction(3)},
                _term_map(apply_derivation(derivation, right).result),
            ),
        )
        assert (
            _term_map(apply_derivation(derivation, _encode(XY, combined_terms)).result)
            == separate
        )


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

    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.polynomials.derivations._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "polynomial_derivation.apply.compute"
        )
        request = DerivationApplyRequest(
            derivation=_triangular(), polynomial=_poly(XY, ((1, (2, 0)),))
        )
        assert tool.run(request) == apply_derivation(
            request.derivation, request.polynomial
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
