from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import parse_operation_input
from jacobian.math.polynomials.tropical._models import (
    PolynomialSubstituteRequest,
)
from jacobian.math.polynomials.tropical._tools import (
    TOOLS,
    compute_polynomial_substitute,
)
from jacobian.math.polynomials.tropical.values import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)


def _scalar(convention: str, value: int | Fraction) -> TropicalScalar:
    return TropicalScalar(
        semiring=TropicalSemiring(convention=convention, base="QQ"),  # type: ignore[arg-type]
        kind="FINITE",
        value=CanonicalRational.from_fraction(Fraction(value)),
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: tuple[tuple[tuple[int, ...], int | Fraction], ...],
    *,
    convention: str = "MIN_PLUS",
) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base="QQ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=variables,
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponents,
                coefficient=_scalar(convention, coefficient),
            )
            for exponents, coefficient in terms
        ),
    )


def _direct_oracle(
    source: TropicalPolynomial,
    target_variables: tuple[str, ...],
    images: tuple[TropicalPolynomial, ...],
) -> dict[tuple[int, ...], Fraction]:
    """Expand each occurrence directly, independently of binary convolution."""
    zero = tuple(0 for _ in target_variables)
    result: dict[tuple[int, ...], Fraction] = {}

    def merge(
        terms: dict[tuple[int, ...], Fraction],
        exponent: tuple[int, ...],
        coefficient: Fraction,
    ) -> None:
        if exponent not in terms:
            terms[exponent] = coefficient
        elif source.semiring.convention == "MIN_PLUS":
            terms[exponent] = min(terms[exponent], coefficient)
        else:
            terms[exponent] = max(terms[exponent], coefficient)

    for source_term in source.terms:
        expanded = {
            zero: source_term.coefficient.value.as_fraction()  # type: ignore[union-attr]
        }
        vanished = False
        for exponent, image in zip(source_term.exponents, images, strict=True):
            for _ in range(exponent):
                if not image.terms:
                    expanded = {}
                    vanished = True
                    break
                product: dict[tuple[int, ...], Fraction] = {}
                for first, first_coefficient in expanded.items():
                    for image_term in image.terms:
                        summed = tuple(
                            a + b
                            for a, b in zip(first, image_term.exponents, strict=True)
                        )
                        coefficient = (
                            first_coefficient
                            + image_term.coefficient.value.as_fraction()  # type: ignore[union-attr]
                        )
                        merge(product, summed, coefficient)
                expanded = product
            if vanished:
                break
        for output_exponents, coefficient in expanded.items():
            merge(result, output_exponents, coefficient)
    return result


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_substitution_matches_direct_distributive_expansion(
    convention: str,
) -> None:
    source = _polynomial(
        ("x", "y"),
        (((0, 0), Fraction(3, 2)), ((1, 1), Fraction(-1, 3))),
        convention=convention,
    )
    images = (
        _polynomial(
            ("u", "v"),
            (((0, 0), 0), ((1, 0), Fraction(2, 3))),
            convention=convention,
        ),
        _polynomial(
            ("u", "v"),
            (((0, 0), -1), ((0, 1), Fraction(1, 4))),
            convention=convention,
        ),
    )
    request = PolynomialSubstituteRequest(
        polynomial=source,
        target_variables=("u", "v"),
        images=images,
    )

    result = compute_polynomial_substitute(request).result
    actual = {
        term.exponents: term.coefficient.value.as_fraction()  # type: ignore[union-attr]
        for term in result.terms
    }
    assert result.variables == ("u", "v")
    assert result.semiring == source.semiring
    assert actual == _direct_oracle(source, ("u", "v"), images)


def test_substitution_is_simultaneous_and_preserves_positional_axis_meaning() -> None:
    source = _polynomial(("x", "y"), (((0, 1), 0), ((1, 0), 0)))
    images = (
        _polynomial(("u", "v"), (((0, 1), 0),)),
        _polynomial(("u", "v"), (((1, 0), 0),)),
    )

    result = compute_polynomial_substitute(
        PolynomialSubstituteRequest(
            polynomial=source,
            target_variables=("u", "v"),
            images=images,
        )
    ).result

    assert tuple(term.exponents for term in result.terms) == ((0, 1), (1, 0))


def test_zero_image_removes_only_monomials_that_use_it() -> None:
    source = _polynomial(("x",), (((0,), 2), ((1,), -1)))
    zero = _polynomial(("t",), ())
    request = PolynomialSubstituteRequest(
        polynomial=source,
        target_variables=("t",),
        images=(zero,),
    )

    result = compute_polynomial_substitute(request).result

    assert len(result.terms) == 1
    assert result.terms[0].exponents == (0,)
    assert result.terms[0].coefficient.value == CanonicalRational.from_fraction(
        Fraction(2)
    )


def test_constant_polynomial_can_change_target_axis_without_images() -> None:
    source = _polynomial((), (((), Fraction(5, 7)),))
    result = compute_polynomial_substitute(
        PolynomialSubstituteRequest(
            polynomial=source,
            target_variables=("u", "v"),
            images=(),
        )
    ).result

    assert result.variables == ("u", "v")
    assert result.terms[0].exponents == (0, 0)
    assert result.terms[0].coefficient.value == CanonicalRational.from_fraction(
        Fraction(5, 7)
    )


def test_annihilated_monomial_does_not_charge_later_image_coefficients() -> None:
    source = _polynomial(("x", "y"), (((1, 1024), 0),))
    zero = _polynomial(("t",), ())
    large = _polynomial(("t",), (((0,), 10**100),))
    result = compute_polynomial_substitute(
        PolynomialSubstituteRequest(
            polynomial=source,
            target_variables=("t",),
            images=(zero, large),
        )
    ).result
    assert result.terms == ()


def test_substitution_rejects_forged_duplicate_source_axis() -> None:
    source = _polynomial(("x", "y"), (((1, 0), 0),)).model_copy(
        update={"variables": ("x", "x")}
    )
    zero = _polynomial(("t",), ())
    with pytest.raises(Exception, match="tropical.polynomial_axis"):
        compute_polynomial_substitute(
            PolynomialSubstituteRequest(
                polynomial=source,
                target_variables=("t",),
                images=(zero, zero),
            )
        )


def test_substitution_accepts_full_exponent_boundary() -> None:
    source = _polynomial(("x",), (((1024,), 0),))
    image = _polynomial(("t",), (((1,), 0),))
    result = compute_polynomial_substitute(
        PolynomialSubstituteRequest(
            polynomial=source,
            target_variables=("t",),
            images=(image,),
        )
    ).result

    assert result.terms[0].exponents == (1024,)


def test_substitution_rejects_exponent_growth_before_coefficients() -> None:
    source = _polynomial(("x",), (((513,), 0),))
    image = _polynomial(("t",), (((2,), 0),))
    request = PolynomialSubstituteRequest(
        polynomial=source,
        target_variables=("t",),
        images=(image,),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_polynomial_substitute(request)

    assert error.value.errors()[0]["type"] == "tropical.substitution_exponent_bound"


def test_substitution_rejects_inflated_support_before_coefficient_expansion() -> None:
    source = _polynomial(("x",), (((512,), 0),))
    image = _polynomial(("t",), (((0,), 0), ((1,), 0)))
    request = PolynomialSubstituteRequest(
        polynomial=source,
        target_variables=("t",),
        images=(image,),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_polynomial_substitute(request)

    assert error.value.errors()[0]["type"] == "tropical.substitution_term_bound"


def test_substitution_is_published_with_an_executable_example() -> None:
    operation = next(
        item
        for item in TOOLS
        if item.operation_id == "tropical.polynomial.substitute.compute"
    )
    assert len(operation.examples) == 1
    request = parse_operation_input(operation.request_type, operation.examples[0].input)
    result = operation.run(request)
    assert result.result.variables == ("t",)
    assert tuple(term.exponents for term in result.result.terms) == ((0,), (1,))
