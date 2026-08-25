"""Exact bounded spectra and inertia over real quadratic fields."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.quadratic_spectral import (
    RealQuadraticInertia,
    RealQuadraticSpectrum,
    inertia,
    singular_spectrum,
    symmetric_spectrum,
)
from jacobian.math.matrices.quadratic_spectral._models import (
    RealQuadraticInertiaRequest,
    RealQuadraticSingularSpectrumRequest,
    RealQuadraticSymmetricSpectrumRequest,
)
from jacobian.math.matrices.quadratic_spectral._tools import TOOLS
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.real_algebraic import compare_real_algebraic
from jacobian.math.real_quadratic import RealQuadraticValue


def _q(
    rational: Fraction | int = 0,
    radical: Fraction | int = 0,
    radicand: int = 3,
) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(Fraction(rational)),
        radical_coefficient=CanonicalRational.from_fraction(Fraction(radical)),
        radicand=radicand,
    )


def _matrix(
    entries: tuple[tuple[RealQuadraticValue, ...], ...],
) -> RealQuadraticMatrix:
    return RealQuadraticMatrix(entries=entries)


def test_pang_spectra_and_proof_critical_order_are_exact() -> None:
    weighted_sum = _matrix(
        (
            (_q(Fraction(1, 2)), _q(0, Fraction(1, 20))),
            (_q(0, Fraction(1, 20)), _q(Fraction(1, 2))),
        )
    )
    projection_product = _matrix(
        (
            (_q(), _q(0, Fraction(3, 8))),
            (_q(), _q()),
        )
    )

    eigenvalues = symmetric_spectrum(weighted_sum)
    singular_values = singular_spectrum(projection_product)

    assert tuple(
        (row.value.polynomial, row.value.real_root_index, row.multiplicity)
        for row in eigenvalues.values
    ) == (
        (("400", "-400", "97"), 1, 1),
        (("400", "-400", "97"), 0, 1),
    )
    assert tuple(
        (row.value.polynomial, row.value.real_root_index, row.multiplicity)
        for row in singular_values.values
    ) == (
        (("64", "0", "-27"), 1, 1),
        (("1", "0"), 0, 1),
    )
    comparison = compare_real_algebraic(
        singular_values.values[0].value,
        eigenvalues.values[0].value,
    )
    assert comparison.order == "GT"
    assert comparison.left_isolating_interval.lower.as_fraction() == Fraction(3, 5)
    assert comparison.right_isolating_interval.upper.as_fraction() == Fraction(3, 5)


def test_symmetric_spectrum_can_return_quartic_values() -> None:
    source = _matrix(
        (
            (_q(0, 1, 2), _q(1, 0, 2)),
            (_q(1, 0, 2), _q(0, 0, 2)),
        )
    )

    result = symmetric_spectrum(source)

    assert tuple(
        (row.value.polynomial, row.value.real_root_index) for row in result.values
    ) == (
        (("1", "0", "-4", "0", "1"), 3),
        (("1", "0", "-4", "0", "1"), 1),
    )


def test_singular_spectrum_can_return_degree_eight_values() -> None:
    source = _matrix(
        (
            (_q(0, 0, 2), _q(1, 0, 2)),
            (_q(1, 1, 2), _q(1, 0, 2)),
        )
    )

    result = singular_spectrum(source)

    polynomial = ("1", "0", "-10", "0", "23", "0", "-14", "0", "1")
    assert tuple(
        (row.value.polynomial, row.value.real_root_index) for row in result.values
    ) == ((polynomial, 7), (polynomial, 5))


def test_repeated_irrational_spectrum_keeps_multiplicity() -> None:
    source = _matrix(
        (
            (_q(0, 2, 6), _q(0, 0, 6)),
            (_q(0, 0, 6), _q(0, 2, 6)),
        )
    )

    result = symmetric_spectrum(source)

    assert len(result.values) == 1
    assert result.values[0].value.polynomial == ("1", "0", "-24")
    assert result.values[0].value.real_root_index == 1
    assert result.values[0].multiplicity == 2


def test_repeated_rational_singular_value_selects_nonnegative_root() -> None:
    identity = _matrix(
        (
            (_q(1, 0, 2), _q(0, 0, 2)),
            (_q(0, 0, 2), _q(1, 0, 2)),
        )
    )

    result = singular_spectrum(identity)

    assert len(result.values) == 1
    assert result.values[0].value.polynomial == ("1", "-1")
    assert result.values[0].value.real_root_index == 0
    assert result.values[0].multiplicity == 2


def test_maxwell_hessian_inertia_is_bound_to_the_exact_source() -> None:
    zero = _q(0, 0, 39)
    source = _matrix(
        (
            (_q(Fraction(19, 8), 0, 39), zero, _q(0, Fraction(-1, 2), 39)),
            (zero, _q(Fraction(-45, 8), 0, 39), zero),
            (_q(0, Fraction(-1, 2), 39), zero, _q(Fraction(13, 4), 0, 39)),
        )
    )

    result = inertia(source)

    assert (result.n_positive, result.n_negative, result.n_zero) == (1, 2, 0)
    assert result.definiteness == "indefinite"


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (
            (
                (_q(0, 0, 2), _q(1, 1, 2), _q(0, 0, 2)),
                (_q(1, 1, 2), _q(0, 0, 2), _q(1, 0, 2)),
                (_q(0, 0, 2), _q(1, 0, 2), _q(0, 0, 2)),
            ),
            (1, 1, 1),
        ),
        (
            (
                (_q(0, 0, 2), _q(1, 0, 2)),
                (_q(1, 0, 2), _q(0, 0, 2)),
            ),
            (1, 1, 0),
        ),
    ],
)
def test_inertia_handles_exact_two_by_two_pivots(
    entries: tuple[tuple[RealQuadraticValue, ...], ...],
    expected: tuple[int, int, int],
) -> None:
    result = inertia(_matrix(entries))

    assert (result.n_positive, result.n_negative, result.n_zero) == expected


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (((_q(),),), (0, 0, 1, "zero")),
        (((_q(), _q()), (_q(), _q())), (0, 0, 2, "zero")),
        (
            (
                (_q(0, 0, 6), _q(0, 0, 6)),
                (_q(0, 0, 6), _q(0, 0, 6)),
            ),
            (0, 0, 2, "zero"),
        ),
        (((_q(1), _q()), (_q(), _q())), (1, 0, 1, "positive_semidefinite")),
        (((_q(-1), _q()), (_q(), _q())), (0, 1, 1, "negative_semidefinite")),
        (((_q(0, 1, 6),),), (1, 0, 0, "positive_definite")),
    ],
)
def test_zero_forms_get_the_explicit_zero_category(
    entries: tuple[tuple[RealQuadraticValue, ...], ...],
    expected: tuple[int, int, int, str],
) -> None:
    result = inertia(_matrix(entries))

    assert (
        result.n_positive,
        result.n_negative,
        result.n_zero,
        result.definiteness,
    ) == expected


def test_serialized_zero_form_inertia_stays_source_bound() -> None:
    source = _matrix(((_q(), _q()), (_q(), _q())))

    payload = inertia(source).model_dump()

    assert RealQuadraticInertia.model_validate(payload) == inertia(source)


def test_matrix_value_requires_one_explicit_quadratic_field() -> None:
    with pytest.raises(ValidationError):
        _matrix(((_q(radicand=2), _q(radicand=3)),))
    with pytest.raises(ValidationError):
        _matrix(((_q(),), (_q(), _q())))


def test_operation_specific_shape_and_work_bounds_are_preflighted() -> None:
    nonsymmetric = _matrix(((_q(), _q(1)), (_q(0), _q())))
    with pytest.raises(ValidationError):
        RealQuadraticSymmetricSpectrumRequest(matrix=nonsymmetric)

    five = tuple(tuple(_q(radicand=2) for _ in range(5)) for _ in range(5))
    with pytest.raises(ValidationError):
        RealQuadraticInertiaRequest(matrix=_matrix(five))

    huge = _q(10**255, 0, 2)
    large_diagonal = _matrix(((huge, _q(radicand=2)), (_q(radicand=2), huge)))
    with pytest.raises(ValidationError):
        RealQuadraticSingularSpectrumRequest(matrix=large_diagonal)


def test_source_bound_results_reject_forged_spectrum_and_inertia() -> None:
    source = _matrix(
        (
            (_q(1, 0, 2), _q(0, 0, 2)),
            (_q(0, 0, 2), _q(2, 0, 2)),
        )
    )
    spectrum_payload = symmetric_spectrum(source).model_dump()
    spectrum_payload["values"] = tuple(reversed(spectrum_payload["values"]))
    with pytest.raises(ValidationError):
        RealQuadraticSpectrum.model_validate(spectrum_payload)

    inertia_payload = inertia(source).model_dump()
    inertia_payload["n_positive"] = 1
    inertia_payload["n_zero"] = 1
    inertia_payload["definiteness"] = "positive_semidefinite"
    with pytest.raises(ValidationError):
        RealQuadraticInertia.model_validate(inertia_payload)


def test_quadratic_spectral_public_api_and_catalog_are_exact() -> None:
    import jacobian.math.matrices.quadratic_spectral as public

    assert tuple(public.__all__) == (
        "RealAlgebraicMultiplicity",
        "RealQuadraticInertia",
        "RealQuadraticSpectrum",
        "inertia",
        "singular_spectrum",
        "symmetric_spectrum",
    )
    assert {tool.operation_id for tool in TOOLS} == {
        "matrix.real_quadratic.inertia.compute",
        "matrix.real_quadratic.singular_spectrum.compute",
        "matrix.real_quadratic.symmetric_spectrum.compute",
    }
