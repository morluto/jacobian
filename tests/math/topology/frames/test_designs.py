"""Spherical/projective designs and algebraic-scalar SIC (#3720)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Poly, Symbol, cyclotomic_poly

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.topology.frames import (
    CyclotomicFrame,
    CyclotomicScalar,
    VectorFamily,
    cyclotomic_sic_povm,
    cyclotomic_sic_profile,
    projective_design_profile,
    spherical_design_verify,
)
from jacobian.math.topology.frames._models import (
    ProjectiveDesignRequest,
    SphericalDesignRequest,
)
from jacobian.math.topology.frames.values import ComplexFrame, euler_phi

R = CanonicalRational
_z = Symbol("z")
_PHI_24 = Poly(cyclotomic_poly(24, _z), _z)


def _zeta_reduce(powers: dict[int, Fraction]) -> CyclotomicScalar:
    """Reduce a zeta_24 exponent dictionary with the maintained backend only."""

    poly = sum(
        coeff * _z**exponent for exponent, coeff in powers.items()
    )
    reduced = Poly(poly, _z).rem(_PHI_24)
    coefficients = [Fraction(0)] * euler_phi(24)
    for monomial, coeff in reduced.as_dict().items():
        coefficients[monomial[0]] = Fraction(coeff)
    return CyclotomicScalar._from_kernel(
        order=24,
        coefficients=tuple(R.from_fraction(value) for value in coefficients),
    )


def _zeta_mul(
    left: dict[int, Fraction], right: dict[int, Fraction]
) -> CyclotomicScalar:
    """Multiply two zeta_24 dictionaries with the maintained backend only."""

    product = sum(
        a * b * _z ** (i + j)
        for i, a in left.items()
        for j, b in right.items()
    )
    reduced = Poly(product, _z).rem(_PHI_24)
    coefficients = [Fraction(0)] * euler_phi(24)
    for monomial, coeff in reduced.as_dict().items():
        coefficients[monomial[0]] = Fraction(coeff)
    return CyclotomicScalar._from_kernel(
        order=24,
        coefficients=tuple(R.from_fraction(value) for value in coefficients),
    )


def _octahedron() -> VectorFamily:
    return VectorFamily(
        dimension=3,
        vectors=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
    )


def _uniform(count: int) -> tuple[R, ...]:
    return tuple(R.from_fraction(Fraction(1, count)) for _ in range(count))


def test_octahedron_is_spherical_3_design() -> None:
    result = spherical_design_verify(_octahedron(), _uniform(6), 3)
    assert result.is_design is True
    assert result.first_failure is None
    assert result.moment_count == 3 + 6 + 10


def test_octahedron_is_not_4_design() -> None:
    """The x^4 moment separates: 1/3 observed against 1/5 required."""
    result = spherical_design_verify(_octahedron(), _uniform(6), 4)
    assert result.is_design is False
    assert result.first_failure is not None
    assert sum(result.first_failure) == 4


def test_weights_must_normalize() -> None:
    with pytest.raises(OperationDomainValidationError, match="sum to one"):
        spherical_design_verify(
            _octahedron(),
            tuple(R.from_fraction(Fraction(1, 3)) for _ in range(6)),
            1,
        )


def test_qubit_basis_is_projective_1_design() -> None:
    one = GaussianRational(real=R(num=1, den=1), imaginary=R(num=0, den=1))
    zero = GaussianRational(real=R(num=0, den=1), imaginary=R(num=0, den=1))
    frame = ComplexFrame(dimension=2, vectors=((one, zero), (zero, one)))
    result = projective_design_profile(
        frame, (R.from_fraction(Fraction(1, 2)),) * 2, 1
    )
    assert result.is_design is True
    assert result.welch_value.as_fraction() == Fraction(1, 2)
    assert result.welch_target.as_fraction() == Fraction(1, 2)


def test_qubit_basis_is_not_2_design() -> None:
    one = GaussianRational(real=R(num=1, den=1), imaginary=R(num=0, den=1))
    zero = GaussianRational(real=R(num=0, den=1), imaginary=R(num=0, den=1))
    frame = ComplexFrame(dimension=2, vectors=((one, zero), (zero, one)))
    result = projective_design_profile(
        frame, (R.from_fraction(Fraction(1, 2)),) * 2, 2
    )
    assert result.is_design is False
    assert result.welch_value.as_fraction() == Fraction(1, 2)
    assert result.welch_target.as_fraction() == Fraction(1, 3)


def _tetrahedron() -> CyclotomicFrame:
    """Regular-tetrahedron SIC lines: sqrt(3), sqrt(2), and cube roots of unity."""

    sqrt3 = _zeta_reduce({2: Fraction(1), 22: Fraction(1)})
    sqrt2 = _zeta_reduce({3: Fraction(1), 21: Fraction(1)})
    one = _zeta_reduce({0: Fraction(1)})
    sqrt2_zeta3 = _zeta_mul({3: Fraction(1), 21: Fraction(1)}, {8: Fraction(1)})
    sqrt2_zeta3_sq = _zeta_mul({3: Fraction(1), 21: Fraction(1)}, {16: Fraction(1)})
    return CyclotomicFrame(
        order=24,
        dimension=2,
        vectors=(
            (sqrt3, _zeta_reduce({})),
            (one, sqrt2),
            (one, sqrt2_zeta3),
            (one, sqrt2_zeta3_sq),
        ),
    )


def test_zero_scalar_encodes_cleanly() -> None:
    assert _zeta_reduce({}).coefficients == tuple(
        R.from_fraction(Fraction(0)) for _ in range(euler_phi(24))
    )


def test_tetrahedron_is_cyclotomic_sic() -> None:
    assert cyclotomic_sic_profile(_tetrahedron()).is_sic is True


def test_sic_profile_is_phase_invariant() -> None:
    """Projective status ignores vector phases; vector data does not."""

    frame = _tetrahedron()
    phased = CyclotomicFrame(
        order=frame.order,
        dimension=frame.dimension,
        vectors=(
            (
                _zeta_reduce({12: Fraction(1)}),
                frame.vectors[0][1],
            ),
            *frame.vectors[1:],
        ),
    )
    assert cyclotomic_sic_profile(frame).is_sic is True
    assert cyclotomic_sic_profile(phased).is_sic is True
    assert phased.vectors[0] != frame.vectors[0]


def test_broken_tetrahedron_is_not_sic() -> None:
    frame = _tetrahedron()
    broken = CyclotomicFrame(
        order=frame.order,
        dimension=frame.dimension,
        vectors=(*frame.vectors[:3], (_zeta_reduce({0: Fraction(2)}), frame.vectors[3][1])),
    )
    assert cyclotomic_sic_profile(broken).is_sic is False


def test_sic_povm_effects_are_scaled_projectors() -> None:
    """Retained numerators satisfy P^2 = D P and Tr P = D with D = Prod n_j."""

    from jacobian.math.topology.frames.operations import (
        _cyclo_add,
        _cyclo_is_zero,
        _cyclo_mul,
        _cyclotomic_modulus_coefficients,
        _scalar_fractions,
    )
    from jacobian.math.topology.frames.values import euler_phi as _phi

    profile = cyclotomic_sic_profile(_tetrahedron())
    result = cyclotomic_sic_povm(_tetrahedron())
    order = result.frame.order
    modulus = _cyclotomic_modulus_coefficients(order)
    degree = _phi(order)
    assert len(result.effect_numerators) == 4
    assert result.effect_denominator.order == order
    common = [Fraction(1)] + [Fraction(0)] * (degree - 1)
    for norm in (_scalar_fractions(entry) for entry in profile.norms):
        common = _cyclo_mul(common, norm, modulus, degree)
    for matrix in result.effect_numerators:
        plain = [[_scalar_fractions(entry) for entry in row] for row in matrix]
        for row in range(2):
            for column in range(2):
                square = [Fraction(0)] * degree
                for k in range(2):
                    square = _cyclo_add(
                        square,
                        _cyclo_mul(plain[row][k], plain[k][column], modulus, degree),
                    )
                scaled = _cyclo_mul(plain[row][column], common, modulus, degree)
                assert _cyclo_is_zero(
                    _cyclo_add(square, [-value for value in scaled])
                )
        trace = [Fraction(0)] * degree
        for row in range(2):
            trace = _cyclo_add(trace, plain[row][row])
        assert _cyclo_is_zero(_cyclo_add(trace, [-value for value in common]))


def test_povm_effects_replay_resolution() -> None:
    """An independent replay of effect/DEN - delta over the retained data."""

    from fractions import Fraction

    from jacobian.math.topology.frames.operations import (
        _cyclo_add,
        _cyclo_is_zero,
        _scalar_fractions,
    )

    result = cyclotomic_sic_povm(_tetrahedron())
    order = result.frame.order
    degree = euler_phi(order)
    denominator = _scalar_fractions(result.effect_denominator)
    for row in range(2):
        for column in range(2):
            total = [Fraction(0)] * degree
            for matrix in result.effect_numerators:
                total = _cyclo_add(total, _scalar_fractions(matrix[row][column]))
            target = (
                list(denominator)
                if row == column
                else [Fraction(0)] * degree
            )
            assert _cyclo_is_zero(
                _cyclo_add(total, [-value for value in target])
            )


def test_forged_overlap_ledger_rejected() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.topology.frames._models import CyclotomicSicResult

    result = cyclotomic_sic_profile(_tetrahedron())
    payload = result.model_dump(mode="json")
    payload["norms"] = payload["norms"][:2]
    with pytest.raises(ValidationError):
        CyclotomicSicResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


def test_design_requests_validate_axes() -> None:
    with pytest.raises(ValidationError):
        SphericalDesignRequest(
            family=_octahedron(), weights=_uniform(5), strength=2
        )
    one = GaussianRational(real=R(num=1, den=1), imaginary=R(num=0, den=1))
    zero = GaussianRational(real=R(num=0, den=1), imaginary=R(num=0, den=1))
    with pytest.raises(ValidationError):
        ProjectiveDesignRequest(
            frame=ComplexFrame(dimension=2, vectors=((one, zero),)),
            weights=(R.from_fraction(Fraction(1)),),
            strength=9,
        )
