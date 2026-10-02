"""Grouped factor growth, source closure, and controlled integer PRS admission."""

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import prod
from typing import Any

import pytest
from sympy import QQ, Poly, symbols

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import _square_free_bounds as bounds
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import PolynomialSquareFreeDecompositionResult
from jacobian.math.polynomials._square_free import admit
from jacobian.math.polynomials.operations import (
    polynomial_square_free_decomposition,
    verify_polynomial_square_free_decomposition,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def polynomial(
    terms: Mapping[tuple[int, ...], int | Fraction], variables: tuple[str, ...]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    exponents=e,
                    coefficient=CanonicalRational.from_fraction(Fraction(c)),
                )
                for e, c in sorted(terms.items(), reverse=True)
                if c
            )
        ),
    )


def family(n: int, variables: tuple[str, ...] = ("x", "y", "z")) -> RationalPolynomial:
    local = {0: 1, 1: -1, n: -1, n + 1: 1}
    return polynomial(
        {
            tuple(e for e, c in row): prod(c for e, c in row)
            for row in product(local.items(), repeat=len(variables))
        },
        variables,
    )


@pytest.mark.parametrize("n", [10, 11, 14])
@pytest.mark.parametrize("variables", [("x", "y", "z"), ("z", "y", "x")])
def test_grouped_factors_match_integer_identity_and_feed_back(
    n: int, variables: tuple[str, ...]
) -> None:
    source = family(n, variables)
    result = polynomial_square_free_decomposition(source)
    expected = polynomial(dict.fromkeys(product(range(n), repeat=3), 1), variables)
    repeated = polynomial(
        {e: (-1) ** (3 - sum(e)) for e in product((0, 1), repeat=3)}, variables
    )
    assert [
        (r.multiplicity, len(r.factor.polynomial.terms)) for r in result.factors
    ] == [(1, n**3), (2, 8)]
    assert result.factors[0].factor == expected
    assert result.factors[1].factor == repeated
    assert result.reconstructed == source
    # Independent coefficient convolution, with no square-free backend.
    reconstructed: dict[tuple[int, ...], int] = {(0,) * 3: 1}
    for factor in (expected, repeated, repeated):
        next_terms: dict[tuple[int, ...], int] = {}
        for (a, c), term in product(reconstructed.items(), factor.polynomial.terms):
            e = tuple(x + y for x, y in zip(a, term.exponents, strict=True))
            next_terms[e] = next_terms.get(e, 0) + c * term.coefficient.num
        reconstructed = {e: c for e, c in next_terms.items() if c}
    assert polynomial(reconstructed, variables) == source
    derived = polynomial_square_free_decomposition(result.factors[0].factor)
    assert derived.reconstructed == expected and len(derived.factors) == 1
    assert verify_polynomial_square_free_decomposition(result)
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate_json(
            result.model_dump_json()
        )
        == result
    )


def test_excessive_grouped_support_refuses_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _square_free_kernel

    original = _square_free_kernel.grouped_factors
    calls = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_square_free_kernel, "grouped_factors", observed)
    with pytest.raises(OperationResourceAdmissionError, match=r"grouped.*support"):
        polynomial_square_free_decomposition(family(17))
    assert calls == []
    claim = PolynomialSquareFreeDecompositionResult(
        polynomial=family(17),
        coefficient=CanonicalRational(num=1, den=1),
        factors=(),
        reconstructed=family(17),
    )
    with pytest.raises(OperationResourceAdmissionError):
        verify_polynomial_square_free_decomposition(claim)


@pytest.mark.parametrize("change", ["missing", "coefficient"])
def test_near_cartesian_source_takes_controlled_generic_path(change: str) -> None:
    source = family(2, ("x", "y"))
    terms = {
        t.exponents: Fraction(t.coefficient.num, t.coefficient.den)
        for t in source.polynomial.terms
    }
    if change == "missing":
        del terms[(1, 1)]
    else:
        terms[(1, 1)] += 1
    source = polynomial(terms, source.variables)
    assert admit(source).kind == "general"
    result = polynomial_square_free_decomposition(source)
    coefficient, oracle = rational_polynomial_to_sympy(source).sqf_list()
    assert result.coefficient.as_fraction() == Fraction(coefficient)
    assert [
        (rational_polynomial_to_sympy(row.factor), row.multiplicity)
        for row in result.factors
    ] == [(p.monic(), m) for p, m in oracle]
    assert result.reconstructed == source


@pytest.mark.parametrize(
    "expression",
    ["(x+y)**3*(x-y)**2", "(2*x+3*y)**2*(x*x+y+1)", "2*(x+y)**2", "x*x+y*y"],
)
def test_coupled_factors_match_independent_backend(expression: str) -> None:
    # Static authored expressions only; no user text reaches this test oracle.
    x, y = symbols("x y")
    expressions = {
        "(x+y)**3*(x-y)**2": (x + y) ** 3 * (x - y) ** 2,
        "(2*x+3*y)**2*(x*x+y+1)": (2 * x + 3 * y) ** 2 * (x * x + y + 1),
        "2*(x+y)**2": 2 * (x + y) ** 2,
        "x*x+y*y": x * x + y * y,
    }
    p = Poly(expressions[expression], x, y, domain=QQ)
    source = rational_polynomial_from_sympy(p, ("x", "y"))
    result = polynomial_square_free_decomposition(source)
    c, oracle = p.sqf_list()
    expected_c = c
    for q, m in oracle:
        expected_c *= q.LC() ** m
    assert result.coefficient.as_fraction() == Fraction(expected_c)
    assert [
        (rational_polynomial_to_sympy(row.factor), row.multiplicity)
        for row in result.factors
    ] == [(q.monic(), m) for q, m in oracle]


def test_eight_axis_monomial_and_mixed_axis_groups_stay_sparse() -> None:
    variables = tuple(f"x{i}" for i in range(8))
    source = polynomial({(64,) * 8: Fraction(-7, 11)}, variables)
    result = polynomial_square_free_decomposition(source)
    assert result.coefficient.as_fraction() == Fraction(-7, 11)
    assert len(result.factors) == 1 and result.factors[0].multiplicity == 64
    assert result.factors[0].factor == polynomial({(1,) * 8: 1}, variables)
    mixed = polynomial_square_free_decomposition(
        polynomial({(2, 3, 2, 0): 5}, ("a", "b", "c", "unused"))
    )
    assert [row.multiplicity for row in mixed.factors] == [2, 3]
    assert [row.factor for row in mixed.factors] == [
        polynomial({(1, 0, 1, 0): 1}, mixed.polynomial.variables),
        polynomial({(0, 1, 0, 0): 1}, mixed.polynomial.variables),
    ]


@pytest.mark.parametrize("variables", [(), ("unused",), ("z", "unused", "x")])
@pytest.mark.parametrize("value", [Fraction(), Fraction(-7, 11)])
def test_zero_and_constants_keep_the_declared_ring(
    variables: tuple[str, ...], value: Fraction
) -> None:
    source = polynomial({(0,) * len(variables): value}, variables)
    result = polynomial_square_free_decomposition(source)
    assert result.reconstructed == source and result.factors == ()
    assert result.coefficient.as_fraction() == value
    assert verify_polynomial_square_free_decomposition(result)


def test_rational_axis_scaling_and_compact_output_closure() -> None:
    x, y = symbols("x y")
    p = Poly(
        Fraction(-7, 11)
        * (Fraction(2, 3) * x + Fraction(5, 7)) ** 2
        * (Fraction(3, 5) * y - Fraction(1, 13)) ** 3,
        x,
        y,
        domain=QQ,
    )
    result = polynomial_square_free_decomposition(
        rational_polynomial_from_sympy(p, ("x", "y"))
    )
    assert [row.multiplicity for row in result.factors] == [2, 3]
    assert result.factors[0].factor == polynomial(
        {(1, 0): 1, (0, 0): Fraction(15, 14)}, ("x", "y")
    )
    assert result.factors[1].factor == polynomial(
        {(0, 1): 1, (0, 0): Fraction(-5, 39)}, ("x", "y")
    )
    assert result.coefficient.as_fraction() == Fraction(-84, 1375)
    assert result.reconstructed == result.polynomial
    large = 10**256 - 1
    compact = polynomial({(32768,): Fraction(1, large), (0,): large}, ("x",))
    factor = polynomial_square_free_decomposition(compact).factors[0].factor
    assert polynomial_square_free_decomposition(factor).factors[0].factor == factor
    assert (
        polynomial_square_free_decomposition(
            polynomial({(65,): 10**256, (0,): 1}, ("x",))
        )
        .reconstructed.polynomial.terms[0]
        .coefficient.num
        == 10**256
    )


def test_normalization_overflow_is_operational_before_compact_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _sympy

    original = _sympy.polynomial_square_free_decomposition
    calls = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_sympy, "polynomial_square_free_decomposition", observed)
    huge = 10**20000 + 1
    source = polynomial({(65,): Fraction(1, huge), (0,): huge}, ("x",))
    with pytest.raises(OperationResourceAdmissionError, match="normalization"):
        polynomial_square_free_decomposition(source)
    assert calls == []


@pytest.mark.parametrize("quantity", ["work", "bits", "storage"])
def test_recursive_envelope_exact_boundaries(
    quantity: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = bounds.decomposition_envelope((12,), 2)
    constant = {
        "work": "MAX_SQUARE_FREE_WORK",
        "bits": "MAX_SQUARE_FREE_SCRATCH_BITS",
        "storage": "MAX_SQUARE_FREE_STORAGE_BITS",
    }[quantity]
    value = getattr(profile, quantity)
    monkeypatch.setattr(bounds, constant, value)
    assert bounds.decomposition_envelope((12,), 2) == profile
    monkeypatch.setattr(bounds, constant, value - 1)
    with pytest.raises(OperationResourceAdmissionError):
        bounds.decomposition_envelope((12,), 2)


def test_forged_factors_reject_without_replaying_during_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _square_free_kernel

    result = polynomial_square_free_decomposition(family(11))
    original = _square_free_kernel.grouped_factors
    calls = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_square_free_kernel, "grouped_factors", observed)
    decoded = PolynomialSquareFreeDecompositionResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result and calls == []
    forged = result.model_copy(update={"coefficient": CanonicalRational(num=2, den=1)})
    assert not verify_polynomial_square_free_decomposition(forged)


@pytest.mark.parametrize("update", [{"variables": ("x", "x", "z")}, {"domain": "RR"}])
def test_invalid_native_source_stays_a_domain_error(update: dict[str, Any]) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        polynomial_square_free_decomposition(family(11).model_copy(update=update))
    assert not isinstance(caught.value, OperationResourceAdmissionError)


def test_generic_reconstruction_charges_source_scale_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = 10**32767 + 1
    terms: dict[tuple[int, ...], Fraction] = {
        (2, 2): Fraction(1, huge),
        (1, 1): Fraction(1, huge),
        (0, 0): Fraction(1, huge),
    }
    source = polynomial(terms, ("x", "y"))
    clearing = bounds.clear_source(terms)
    assert (
        bounds.decomposition_envelope((2, 2), clearing.height).bits
        < clearing.envelope.bits
    )
    assert polynomial_square_free_decomposition(source).reconstructed == source
    monkeypatch.setattr(bounds, "MAX_SQUARE_FREE_SCRATCH_BITS", clearing.envelope.bits)
    with pytest.raises(OperationResourceAdmissionError, match="scratch_height"):
        polynomial_square_free_decomposition(source)


def test_axis_clearing_and_retained_scales_are_aggregated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = 10**32767 + 1
    source = polynomial(
        {(64,) * 8: Fraction(1, huge)}, tuple(f"x{i}" for i in range(8))
    )
    assert polynomial_square_free_decomposition(source).coefficient.den == huge
    single = bounds.clearing_envelope({(64,): Fraction(1, huge)})
    assert single.storage < 1 << 20
    monkeypatch.setattr(bounds, "MAX_SQUARE_FREE_STORAGE_BITS", 1 << 20)
    with pytest.raises(OperationResourceAdmissionError, match="coefficient_storage"):
        polynomial_square_free_decomposition(source)


def test_exact_compact_canonical_component_boundary_is_admitted() -> None:
    largest = 10**32768 - 1
    source = polynomial({(65,): 1, (0,): largest}, ("x",))
    result = polynomial_square_free_decomposition(source)
    assert result.factors[0].factor == source
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate_json(
            result.model_dump_json()
        )
        == result
    )
    assert (
        polynomial_square_free_decomposition(result.factors[0].factor).factors[0].factor
        == source
    )


def test_actual_4096_term_factor_boundary_remains_composable() -> None:
    source = polynomial(dict.fromkeys(product(range(16), repeat=3), 1), ("x", "y", "z"))
    result = polynomial_square_free_decomposition(source)
    assert len(source.polynomial.terms) == 4096
    assert len(result.factors) == 1 and result.factors[0].multiplicity == 1
    assert result.factors[0].factor == source
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate_json(
            result.model_dump_json()
        )
        == result
    )


def test_oversized_native_axes_stop_before_term_validation() -> None:
    source = family(11).model_copy(
        update={
            "variables": tuple(f"x{i}" for i in range(9)),
            "polynomial": SparseRationalPolynomial.model_construct(terms=None),
        }
    )
    with pytest.raises(OperationDomainValidationError, match="named axes"):
        polynomial_square_free_decomposition(source)


@pytest.mark.parametrize(
    "case",
    ["three_square", "two_power64", "three_power64", "rational_affine", "unused_axis"],
)
def test_affine_powers_keep_cheap_coupled_sources_admitted(case: str) -> None:
    x, y, z = symbols("x y z")
    linear, multiplicity, scalar = {
        "three_square": (x + y + z, 2, Fraction(1)),
        "two_power64": (x + y, 64, Fraction(1)),
        "three_power64": (x + y + z, 64, Fraction(1)),
        "rational_affine": (2 * x - 3 * y + 5, 5, Fraction(-7, 11)),
        "unused_axis": (y + z, 2, Fraction(1)),
    }[case]
    p = Poly(scalar * linear**multiplicity, x, y, z, domain=QQ)
    source = rational_polynomial_from_sympy(p, ("x", "y", "z"))
    result = polynomial_square_free_decomposition(source)
    expected = Poly(linear, x, y, z, domain=QQ).monic()
    assert len(result.factors) == 1
    assert result.factors[0].multiplicity == multiplicity
    assert rational_polynomial_to_sympy(result.factors[0].factor) == expected
    assert result.coefficient.as_fraction() == Fraction(p.LC())
    assert result.reconstructed == source
    if case == "three_power64":
        assert len(source.polynomial.terms) == 2145
    assert verify_polynomial_square_free_decomposition(result)


@pytest.mark.parametrize("change", ["constant", "missing", "coefficient"])
def test_quadratic_affine_power_mismatch_is_square_free(change: str) -> None:
    x, y, z = symbols("x y z")
    p = Poly((x + y + z) ** 2, x, y, z, domain=QQ)
    changes = {"constant": 1, "missing": -z * z, "coefficient": -x * y}
    p += Poly(changes[change], x, y, z, domain=QQ)
    source = rational_polynomial_from_sympy(p, ("x", "y", "z"))
    result = polynomial_square_free_decomposition(source)
    assert len(result.factors) == 1 and result.factors[0].multiplicity == 1
    assert result.factors[0].factor == source
    assert p.sqf_list()[1] == [(p, 1)]


def test_quadratic_recognition_resource_refusal_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, z = symbols("x y z")
    source = rational_polynomial_from_sympy(
        Poly((x + y + z) ** 2, x, y, z, domain=QQ), ("x", "y", "z")
    )
    monkeypatch.setattr(bounds, "MAX_SQUARE_FREE_WORK", 1)
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_square_free_decomposition(source)


def test_monomial_partial_certificate_avoids_a_dense_coupled_box() -> None:
    source = polynomial({(64, 0, 0): 1, (0, 64, 64): 1}, ("x", "y", "z"))
    result = polynomial_square_free_decomposition(source)
    assert result.factors[0].factor == source and result.factors[0].multiplicity == 1
    assert result.reconstructed == source


def test_denominator_diversity_is_refused_before_controlled_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _square_free_kernel

    original = _square_free_kernel.grouped_factors
    calls = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_square_free_kernel, "grouped_factors", observed)
    huge = 10**5000
    source = polynomial(
        {
            (2, 2): Fraction(1, huge + 1),
            (1, 1): Fraction(1, huge + 3),
            (0, 0): Fraction(1, huge + 7),
        },
        ("x", "y"),
    )
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_square_free_decomposition(source)
    assert calls == []
