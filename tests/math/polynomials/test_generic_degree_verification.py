"""Strict-JSON authored evidence is checked independently of the producer."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest
import sympy

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_polynomial_from_sympy,
)
from jacobian.math.polynomials.maps import RationalPolynomialMap, verify_generic_degree
from jacobian.math.polynomials.maps._models import (
    GenericDegreeOutcome,
    GenericDegreeResult,
    GenericFiberCertificate,
    GenericFiberPolynomial,
    GenericFiberTerm,
)

_X, _Y, _Z, _T, _U, _V = sympy.symbols("x y z t u v")


def _claim(
    outputs: list[Any],
    basis: list[Any],
    transform: list[list[Any]],
    monomials: tuple[tuple[int, ...], ...],
    degree: int | None = 2,
    outcome: GenericDegreeOutcome = "GENERICALLY_FINITE",
    variables: tuple[Any, ...] = (_X,),
    parameters: tuple[Any, ...] = (_T,),
) -> GenericDegreeResult:
    """Construct small independent exact witnesses, then strictly decode JSON."""
    names = tuple(map(str, variables))
    params = tuple(map(str, parameters))

    def fiber(expression: Any) -> GenericFiberPolynomial:
        polynomial = sympy.Poly(
            expression, *variables, domain=sympy.QQ.frac_field(*parameters)
        )
        return GenericFiberPolynomial(
            terms=tuple(
                GenericFiberTerm(
                    source_exponents=monomial,
                    coefficient=rational_function_from_sympy(coefficient, params),
                )
                for monomial, coefficient in polynomial.terms()
                if coefficient
            )
        )

    value = GenericDegreeResult(
        source=RationalPolynomialMap(
            input_variables=names,
            output_polynomials=tuple(
                rational_polynomial_from_sympy(
                    sympy.Poly(f, *variables, domain=sympy.QQ), names
                )
                for f in outputs
            ),
        ),
        outcome=outcome,
        degree=degree,
        evidence=GenericFiberCertificate(
            target_parameters=params,
            source_variable_order=names,
            basis=tuple(map(fiber, basis)),
            basis_from_source=tuple(tuple(map(fiber, row)) for row in transform),
            standard_monomials=monomials,
        ),
    )
    return GenericDegreeResult.model_validate_json(value.model_dump_json(), strict=True)


@pytest.mark.parametrize(
    ("monomials", "degree"),
    [(((0,),), 1), (((0,), (1,), (2,)), 3), (((0,), (2,)), 2), (((0,), (1,)), 3)],
    ids=["incomplete", "extra", "nonstandard", "wrong-count"],
)
def test_finite_claim_requires_exact_complement(
    monomials: tuple[tuple[int, ...], ...], degree: int
) -> None:
    assert not verify_generic_degree(
        _claim([_X**2], [_X**2 - _T], [[1]], monomials, degree)
    )


@pytest.mark.parametrize("monomials", [(), ((0,), (1,))])
def test_finite_fiber_cannot_be_called_positive_dimensional(
    monomials: tuple[tuple[int, ...], ...],
) -> None:
    value = _claim(
        [_X**2], [_X**2 - _T], [[1]], monomials, None, "DOMINANT_NOT_GENERICALLY_FINITE"
    )
    assert not verify_generic_degree(value)


def test_free_variable_cannot_be_called_finite() -> None:
    assert not verify_generic_degree(
        _claim([_X], [_X - _T], [[1]], ((0, 0),), 1, variables=(_X, _Y))
    )


def test_positive_dimensional_fiber_uses_empty_finite_list() -> None:
    value = _claim(
        [_X],
        [_X - _T],
        [[1]],
        (),
        None,
        "DOMINANT_NOT_GENERICALLY_FINITE",
        variables=(_X, _Y),
    )
    assert verify_generic_degree(value)
    assert value.evidence is not None
    assert not verify_generic_degree(
        value.model_copy(
            update={
                "evidence": value.evidence.model_copy(
                    update={"standard_monomials": ((0, 0),)}
                )
            }
        )
    )


def test_proper_subideal_does_not_prove_source_degree() -> None:
    assert not verify_generic_degree(
        _claim([_X], [_X * (_X - _T)], [[_X]], ((0,), (1,)))
    )


def test_nonliteral_unit_cannot_be_called_finite() -> None:
    assert not verify_generic_degree(_claim([0], [-_T], [[1]], ((0,),), 1))


@pytest.mark.parametrize("basis", [[_X**2 - _T, _X * _Y - _U]])
def test_generating_the_source_ideal_is_not_enough_for_groebner_evidence(
    basis: list[Any],
) -> None:
    assert not verify_generic_degree(
        _claim(
            [_X**2, _X * _Y],
            basis,
            [[1, 0], [0, 1]],
            ((0, 0), (1, 0)),
            variables=(_X, _Y),
            parameters=(_T, _U),
        )
    )


def test_spairs_are_needed_even_when_supplied_leading_ideal_is_finite() -> None:
    value = _claim(
        [_X**2, _X * _Y, _Y**2],
        [_X**2 - _T, _X * _Y - _U, _Y**2 - _V],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ((0, 0), (0, 1), (1, 0)),
        3,
        variables=(_X, _Y),
        parameters=(_T, _U, _V),
    )
    oracle = sympy.groebner(
        [_X**2 - _T, _X * _Y - _U, _Y**2 - _V],
        _X,
        _Y,
        domain=sympy.QQ.frac_field(_T, _U, _V),
    )
    assert list(oracle) == [1]
    assert not verify_generic_degree(value)


@pytest.mark.parametrize("power", [1, 2, 3, 8, 64])
def test_monic_univariate_fibers_including_larger_than_producer_envelope(
    power: int,
) -> None:
    assert verify_generic_degree(
        _claim(
            [_X**power],
            [_X**power - _T],
            [[1]],
            tuple((i,) for i in range(power)),
            power,
        )
    )


@pytest.mark.parametrize("scale", [2, -3, _T, (_T + 1) / (_T**2 + 1)])
def test_nonmonic_and_rationally_rescaled_equivalent_witnesses(scale: Any) -> None:
    assert verify_generic_degree(
        _claim([_X**2], [scale * (_X**2 - _T)], [[scale]], ((0,), (1,)))
    )


def test_nonreduced_reordered_redundant_and_zero_generators_are_allowed() -> None:
    f = _X**2 - _T
    assert verify_generic_degree(
        _claim([_X**2], [_X * f, 0, -2 * f, f], [[_X, 0, -2, 1]], ((0,), (1,)))
    )


def test_parameter_leading_coefficients_and_nontrivial_transformations() -> None:
    source = [_X**2 - _T, _X * _Y - _U]
    basis = [_U * _X - _T * _Y, _T * _Y**2 - _U**2]
    transformation = [[_Y, -(_Y**2)], [-_X, _X * _Y + _U]]
    oracle = sympy.groebner(source, _X, _Y, domain=sympy.QQ.frac_field(_T, _U))
    assert oracle.is_zero_dimensional
    assert all(oracle.reduce(g)[1] == 0 for g in basis)
    assert verify_generic_degree(
        _claim(
            [_X**2, _X * _Y],
            basis,
            transformation,
            ((0, 0), (0, 1)),
            variables=(_X, _Y),
            parameters=(_T, _U),
        )
    )


@pytest.mark.parametrize(
    ("basis", "transform"),
    [([1], [[-1 / _T]]), ([-_T], [[1]]), ([-2 * _T, -_T * _X, 0], [[2, _X, 0]])],
)
def test_empty_fibers_and_nonliteral_unit_generators(
    basis: list[Any], transform: list[list[Any]]
) -> None:
    assert verify_generic_degree(
        _claim([0], basis, transform, (), None, "NOT_DOMINANT")
    )


def test_false_source_and_transformation_are_rejected_after_structural_decode() -> None:
    assert not verify_generic_degree(_claim([_X**3], [_X**2 - _T], [[1]], ((0,), (1,))))
    assert not verify_generic_degree(_claim([_X**2], [_X**2 - _T], [[2]], ((0,), (1,))))
    assert not verify_generic_degree(
        _claim([_X], [0], [[0]], (), None, "DOMINANT_NOT_GENERICALLY_FINITE")
    )


def test_source_parameter_names_are_separate_axes() -> None:
    value = _claim([_X**2], [_X**2 - _T], [[1]], ((0,), (1,)))
    payload = value.model_dump(mode="json")
    payload["evidence"]["target_parameters"] = ["x"]
    for polynomial in [
        *payload["evidence"]["basis"],
        *payload["evidence"]["basis_from_source"][0],
    ]:
        for term in polynomial["terms"]:
            term["coefficient"]["variables"] = ["x"]
    assert verify_generic_degree(
        GenericDegreeResult.model_validate_json(json.dumps(payload), strict=True)
    )


def test_maximum_finite_complement_remains_accepted() -> None:
    assert verify_generic_degree(
        _claim(
            [_X**8, _Y**8, _Z**8],
            [_X**8 - _T, _Y**8 - _U, _Z**8 - _V],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            tuple((i, j, k) for i in range(8) for j in range(8) for k in range(8)),
            512,
            variables=(_X, _Y, _Z),
            parameters=(_T, _U, _V),
        )
    )


def test_missing_evidence_and_invalid_native_shape_are_false() -> None:
    value = _claim([_X], [_X - _T], [[1]], ((0,),), 1)
    assert not verify_generic_degree(value.model_copy(update={"evidence": None}))
    assert not verify_generic_degree(value.model_copy(update={"source": None}))


def test_deadline_and_cancellation_are_not_false_mathematical_claims() -> None:
    value = _claim([_X], [_X - _T], [[1]], ((0,),), 1)
    with (
        request_execution(time.monotonic() - 1, outer_deadline=time.monotonic() - 0.5),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        verify_generic_degree(value)
    event = threading.Event()
    event.set()
    with request_cancellation(event), pytest.raises(OperationExecutionCancelledError):
        verify_generic_degree(value)


def _denominator_growth_claim(size: int) -> GenericDegreeResult:
    # The first generator is already a unit. The second valid forward identity
    # still has to be replayed, and its distinct denominators grow concretely.
    extra = sum(_X**i / (_T + i + 1) for i in range(size))
    return _claim([0], [-_T, extra], [[1, -extra / _T]], (), None, "NOT_DOMINANT")


def test_nontrivial_denominator_growth_has_a_useful_accepted_boundary() -> None:
    assert verify_generic_degree(_denominator_growth_claim(32))


def test_exhausted_denominator_growth_does_not_refute_a_true_claim() -> None:
    value = _denominator_growth_claim(48)
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_generic_degree(value)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.generic_degree_verification_budget"
    )


def _pseudoreduction_growth_claim(power: int) -> GenericDegreeResult:
    scale = 1 + _T + _U
    quotient = sum(_X ** (power - 1 - i) * _U**i for i in range(power))
    # F=(x^power+y,x) is a polynomial automorphism. The authored basis is
    # (scale*(x-u), y+u^power-t), with a nontrivial source transformation.
    return _claim(
        [_X**power + _Y, _X],
        [scale * (_X - _U), _Y + _U**power - _T],
        [[0, 1], [scale, -quotient]],
        ((0, 0),),
        1,
        variables=(_X, _Y),
        parameters=(_T, _U),
    )


def test_nontrivial_fraction_free_reduction_remains_admitted() -> None:
    assert verify_generic_degree(_pseudoreduction_growth_claim(32))


def test_pseudoreduction_growth_refusal_is_not_a_false_claim() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        verify_generic_degree(_pseudoreduction_growth_claim(64))


def test_source_axes_and_transformation_rows_are_bound() -> None:
    value = _claim([_X**2], [_X**2 - _T], [[1]], ((0,), (1,)))
    payload = value.model_dump(mode="json")
    payload["source"]["input_variables"] = ["y"]
    payload["source"]["output_polynomials"][0]["variables"] = ["y"]
    assert not verify_generic_degree(
        GenericDegreeResult.model_validate_json(json.dumps(payload), strict=True)
    )
    payload = value.model_dump(mode="json")
    payload["evidence"]["basis_from_source"] *= 2
    assert not verify_generic_degree(
        GenericDegreeResult.model_validate_json(json.dumps(payload), strict=True)
    )
