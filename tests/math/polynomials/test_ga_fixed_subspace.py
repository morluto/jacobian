"""Exact fixed vectors of a bounded finite-dimensional polynomial Ga action."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations import _stable_kernels
from jacobian.math.polynomials.derivations._models import (
    PolynomialDerivation,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_kernels import (
    ga_fixed_subspace,
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations.operations import ga_action_from_derivation
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...], terms: tuple[tuple[int, ...], ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": list(exponents)}
                    for exponents in terms
                ]
            },
        }
    )


def _translation_action() -> PolynomialGaAction:
    x = _poly(("x",), ((1,),))
    one = _poly(("x",), ((0,),))
    zero = _poly(("x",), ())
    return ga_action_from_derivation(
        PolynomialDerivation(variables=("x",), images=(one,)),
        ((x, one, zero),),
    )


def test_translation_fixed_space_is_constant_polynomials_with_coordinates() -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    result = ga_fixed_subspace(stable)

    assert tuple(value.as_fraction() for value in result.coordinates[0]) == (
        Fraction(1),
        Fraction(0),
    )
    assert result.basis == (_poly(("x",), ((0,),)),)
    # Independent oracle: the polynomial substitution x -> x+t fixes 1 and
    # does not fix x, so the invariant line in this span is exactly QQ*1.
    assert result.subrepresentation.action_matrix[0][1].polynomial.terms
    assert result.model_validate_json(result.model_dump_json()) == result


def test_shear_fixes_y_and_constants_but_not_x() -> None:
    action = ga_action_from_derivation(
        PolynomialDerivation(
            variables=("x", "y"),
            images=(_poly(("x", "y"), ((0, 1),)), _poly(("x", "y"), ())),
        ),
        (
            (
                _poly(("x", "y"), ((1, 0),)),
                _poly(("x", "y"), ((0, 1),)),
                _poly(("x", "y"), ()),
            ),
            (_poly(("x", "y"), ((0, 1),)), _poly(("x", "y"), ())),
        ),
    )
    basis = (
        _poly(("x", "y"), ((0, 0),)),
        _poly(("x", "y"), ((1, 0),)),
        _poly(("x", "y"), ((0, 1),)),
    )
    result = ga_fixed_subspace(ga_stable_subrepresentation(action, basis))
    assert result.basis == (basis[0], basis[2])
    assert tuple(
        tuple(value.as_fraction() for value in row) for row in result.coordinates
    ) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_zero_action_fixes_the_whole_span() -> None:
    zero_action = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_poly(("x", "t"), ((1, 0),)),),
    )
    basis = (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    result = ga_fixed_subspace(ga_stable_subrepresentation(zero_action, basis))
    assert result.basis == basis
    assert tuple(
        tuple(value.as_fraction() for value in row) for row in result.coordinates
    ) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def test_forged_action_matrix_is_rejected() -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    forged = stable.model_copy(
        update={
            "action_matrix": (
                (stable.action_matrix[0][0], stable.action_matrix[0][0]),
                stable.action_matrix[1],
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ga_fixed_subspace(forged)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.unverified_subrepresentation"
    )


def test_kernel_work_is_admitted_before_exact_elimination(monkeypatch) -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    monkeypatch.setattr(_stable_kernels, "MAX_GA_FIXED_KERNEL_WORK", 1)

    def elimination_must_not_start(_matrix):
        raise AssertionError("kernel elimination ran before work admission")

    monkeypatch.setattr(
        _stable_kernels, "_rational_kernel_basis", elimination_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_fixed_subspace(stable)
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_ga_fixed_subspace.kernel_work"
    )


def test_hadamard_bound_rejects_exact_matrix_before_kernel_elimination() -> None:
    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
    )
    denominators = []
    for prime in primes:
        denominator = prime
        while len(str(denominator * prime)) <= 15:
            denominator *= prime
        denominators.append(denominator)
    row = [Fraction(1, denominator) for denominator in denominators]
    matrix = [row.copy() for _ in primes]

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        _stable_kernels._admit_and_integerize_kernel_matrix(matrix)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.minor_growth"
    )


def test_aggregate_polynomial_support_is_admitted_before_expansion(monkeypatch) -> None:
    zero_action = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_poly(("x", "t"), ((1, 0),)),),
    )
    basis = (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    stable = ga_stable_subrepresentation(zero_action, basis)
    monkeypatch.setattr(_stable_kernels, "MAX_GA_FIXED_SUPPORT_TERMS", 1)

    def expansion_must_not_start(*_args, **_kwargs):
        raise AssertionError(
            "polynomial representatives were expanded before support admission"
        )

    monkeypatch.setattr(
        _stable_kernels, "_fixed_polynomial_basis", expansion_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_fixed_subspace(stable)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.output_terms"
    )


def test_malformed_typed_subrepresentation_is_revalidated() -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    # model_construct skips validation; the malformed matrix entry would raise
    # AttributeError from the term loop if the typed instance were trusted.
    malformed = stable.model_construct(
        action=stable.action,
        basis=stable.basis,
        action_matrix=((("not a polynomial",),), ()),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ga_fixed_subspace(malformed)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.request_shape"
    )


def test_oversized_claimed_matrix_is_rejected_before_reconstruction(
    monkeypatch,
) -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    wide = RationalPolynomial.model_validate(
        {
            "variables": ["t"],
            "polynomial": {
                "terms": [{"coefficient": {"num": 10**200, "den": 1}, "exponents": [0]}]
            },
        }
    )
    forged = stable.model_copy(
        update={
            "action_matrix": (
                (wide, stable.action_matrix[0][1]),
                stable.action_matrix[1],
            )
        }
    )

    def reconstruction_must_not_start(*_args, **_kwargs):
        raise AssertionError("reconstruction ran before matrix width admission")

    monkeypatch.setattr(
        _stable_kernels, "ga_stable_subrepresentation", reconstruction_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_fixed_subspace(forged)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.matrix_coefficient_growth"
    )


def test_oversized_claimed_matrix_degree_is_rejected_before_reconstruction(
    monkeypatch,
) -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    high_degree = RationalPolynomial.model_validate(
        {
            "variables": ["t"],
            "polynomial": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [1_000]}]
            },
        }
    )
    forged = stable.model_copy(
        update={
            "action_matrix": (
                (high_degree, stable.action_matrix[0][1]),
                stable.action_matrix[1],
            )
        }
    )

    def reconstruction_must_not_start(*_args, **_kwargs):
        raise AssertionError("reconstruction ran before matrix degree admission")

    monkeypatch.setattr(
        _stable_kernels, "ga_stable_subrepresentation", reconstruction_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_fixed_subspace(forged)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_fixed_subspace.matrix_degree_growth"
    )


def test_result_cell_admission_counts_retained_entries(monkeypatch) -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    monkeypatch.setattr(_stable_kernels, "MAX_GA_ACTION_OUTPUT_CELLS", 10**9)
    narrow = [(Fraction(1), Fraction(0))]
    wide = [
        (Fraction(10**_stable_kernels.MAX_GA_FIXED_COORDINATE_DIGITS - 1), Fraction(1))
    ]

    # The envelope counts retained entries, so an exact-but-wide coordinate
    # does not by itself change admission.
    assert _stable_kernels._admit_fixed_result_cells(
        stable, narrow, 1
    ) == _stable_kernels._admit_fixed_result_cells(stable, wide, 1)


def test_result_cell_envelope_rejects_before_construction(monkeypatch) -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    # Below the retained-entry envelope the request is refused before any
    # representative is constructed.
    monkeypatch.setattr(_stable_kernels, "MAX_GA_ACTION_OUTPUT_CELLS", 1)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_fixed_subspace(stable)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.output_bytes"
    )

    # Well above it the same request succeeds.
    monkeypatch.setattr(_stable_kernels, "MAX_GA_ACTION_OUTPUT_CELLS", 10**9)
    assert ga_fixed_subspace(stable).basis == (_poly(("x",), ((0,),)),)


def test_parameter_degree_bound_follows_the_action_chain() -> None:
    names = ("x", "y", "z")
    x = _poly(names, ((1, 0, 0),))
    y = _poly(names, ((0, 1, 0),))
    z = _poly(names, ((0, 0, 1),))
    zero = _poly(names, ())
    action = ga_action_from_derivation(
        PolynomialDerivation(variables=names, images=(y, z, zero)),
        ((x, y, z, zero), (y, z, zero), (z, zero)),
    )
    stable = ga_stable_subrepresentation(action, (x, y, z))
    # exp(tD)(x) = x + t y + t^2 z / 2, so the action matrix reaches
    # parameter degree 2. The claimed-matrix degree envelope must follow the
    # action (2) rather than the 64-degree source envelope.
    assert (
        max(
            term.exponents[-1]
            for row in stable.action_matrix
            for entry in row
            for term in entry.polynomial.terms
        )
        == 2
    )
    assert ga_fixed_subspace(stable).basis == (z,)
