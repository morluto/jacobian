import json

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.math.ore_algebras.proper_hypergeometric_terms import (
    IntegerAffineFactorial,
    ProperHypergeometricTerm,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def polynomial(variables, terms):
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=numerator, den=denominator),
                    exponents=exponents,
                )
                for exponents, numerator, denominator in terms
            )
        ),
    )


def binomial_term():
    # n! / (k! (n-k)!) with reciprocal factorials zero on negative arguments.
    return ProperHypergeometricTerm(
        polynomial=polynomial(("n", "k"), [((0, 0), 1, 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=0, k_coefficient=1, offset=0, power=-1
            ),
            IntegerAffineFactorial(
                n_coefficient=1, k_coefficient=-1, offset=0, power=-1
            ),
            IntegerAffineFactorial(n_coefficient=1, k_coefficient=0, offset=0, power=1),
        ),
    )


def test_binomial_factorization_retains_support_and_well_defined_regions():
    term = binomial_term()
    assert [factor.affine_key for factor in term.support_factors] == [
        (0, 1, 0),
        (1, -1, 0),
    ]
    assert [factor.affine_key for factor in term.domain_factors] == [(1, 0, 0)]
    assert (
        term.model_dump()
        == ProperHypergeometricTerm.model_validate(term.model_dump()).model_dump()
    )


def test_factor_order_axes_and_factor_multiplicity_are_canonical():
    value = binomial_term().model_dump()
    value["factorial_factors"] = list(reversed(value["factorial_factors"]))
    with pytest.raises(ValidationError, match="proper_hypergeometric_factor_order"):
        ProperHypergeometricTerm.model_validate(value)

    with pytest.raises(ValidationError, match="proper_hypergeometric_polynomial_axes"):
        ProperHypergeometricTerm(polynomial=polynomial(("k", "n"), [((0, 0), 1, 1)]))

    duplicate = binomial_term().model_dump()
    factors = duplicate["factorial_factors"]
    duplicate["factorial_factors"] = (*factors[:1], factors[0], *factors[1:])
    with pytest.raises(ValidationError, match="proper_hypergeometric_duplicate_factor"):
        ProperHypergeometricTerm.model_validate(duplicate)


def test_zero_term_has_one_structural_spelling():
    zero = ProperHypergeometricTerm(polynomial=polynomial(("n", "k"), []))
    assert zero.is_zero
    assert zero.support_factors == zero.domain_factors == ()

    noncanonical = zero.model_dump()
    noncanonical["n_base"] = {"num": 2, "den": 1}
    with pytest.raises(ValidationError, match="proper_hypergeometric_zero_normal_form"):
        ProperHypergeometricTerm.model_validate(noncanonical)


def test_affine_factor_bounds_and_zero_power_are_rejected():
    with pytest.raises(ValidationError, match="proper_hypergeometric_constant_factor"):
        IntegerAffineFactorial(n_coefficient=0, k_coefficient=0, offset=3, power=1)
    with pytest.raises(ValidationError, match="proper_hypergeometric_zero_power"):
        IntegerAffineFactorial(n_coefficient=1, k_coefficient=0, offset=0, power=0)
    with pytest.raises(ValidationError):
        IntegerAffineFactorial(n_coefficient=129, k_coefficient=0, offset=0, power=1)


def test_large_offset_is_accepted_without_increasing_carrier_size():
    factor = IntegerAffineFactorial(
        n_coefficient=1, k_coefficient=0, offset=129, power=1
    )
    assert factor.offset == 129
    assert factor.model_dump()["offset"] == 129


def test_affine_offset_uses_the_exact_integer_wire_codec():
    # Offsets past the interoperable JSON integer range must not be emitted as
    # bare JSON numbers, which JavaScript consumers would round.
    offset = (1 << 53) + 1
    factor = IntegerAffineFactorial(
        n_coefficient=1, k_coefficient=0, offset=offset, power=1
    )
    wire = json.loads(factor.model_dump_json())
    assert wire["offset"] == str(offset)
    assert (
        IntegerAffineFactorial.model_validate_json(factor.model_dump_json()) == factor
    )


def test_affine_offset_digit_bound_is_enforced():
    bound = 10**MAX_CANONICAL_INTEGER_DIGITS
    with pytest.raises(ValidationError, match=r"exact_integer\.digit_bound"):
        IntegerAffineFactorial(n_coefficient=1, k_coefficient=0, offset=bound, power=1)
    admitted = IntegerAffineFactorial(
        n_coefficient=1, k_coefficient=0, offset=bound - 1, power=1
    )
    assert admitted.offset == bound - 1
