"""Independent exact checks for root--critical distance profiles."""

from __future__ import annotations

import time

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.root_critical._models import (
    RootCriticalDistanceProfileRequest,
)
from jacobian.math.polynomials.root_critical._tools import TOOLS
from jacobian.math.polynomials.root_critical.operations import (
    root_critical_distance_profile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(*coefficients: tuple[int, int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=value, den=1),
                    exponents=(exponent,),
                )
                for exponent, value in sorted(coefficients, reverse=True)
                if value
            )
        ),
    )


def test_cubic_profile_has_complete_rows_and_unit_distances() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -1)))

    assert len(result.roots) == 3
    assert len(result.critical_points) == 1
    assert len(result.pairs) == 3
    assert all(row.distance_squared.polynomial == (1, -1) for row in result.pairs)
    assert all(row.kind == "POSITIVE" for row in result.pairs)
    assert all(row.isolating_interval.lower.as_fraction() == 1 for row in result.pairs)


def test_repeated_root_and_zero_distance_are_retained() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (1, -3), (0, 2)))

    assert [root.multiplicity for root in result.roots] == [2, 1]
    assert [critical.multiplicity for critical in result.critical_points] == [1, 1]
    zero_rows = [row for row in result.pairs if row.kind == "ZERO_DISTANCE"]
    assert len(zero_rows) == 1
    assert zero_rows[0].distance_squared.polynomial == (1, 0)


def test_nonreal_roots_use_exact_nonnegative_squared_distances() -> None:
    # (z^2+1)(z-2), with derivative roots 1 and 1/3.  The expected values
    # are obtained independently from the explicit roots ±i and 2.
    result = root_critical_distance_profile(
        _polynomial((3, 1), (2, -2), (1, 1), (0, -2))
    )

    assert len(result.pairs) == 6
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert values == {(1, -1), (1, -2), (9, -10), (9, -25)}
    assert all(row.isolating_interval.lower.as_fraction() >= 0 for row in result.pairs)
    assert any(
        root.rectangle.imaginary_lower.as_fraction() < 0 for root in result.roots
    )
    assert any(
        root.rectangle.imaginary_upper.as_fraction() > 0 for root in result.roots
    )


def test_admission_rejects_incomplete_pair_budget_before_expansion() -> None:
    request = RootCriticalDistanceProfileRequest(
        polynomial=_polynomial((3, 1), (0, -1)),
        max_pair_rows=2,
    )
    with pytest.raises(OperationResourceAdmissionError):
        TOOLS[0].run(request)


def test_constant_polynomial_is_outside_profile_domain() -> None:
    with pytest.raises(OperationDomainValidationError):
        root_critical_distance_profile(_polynomial((0, 3)))


def test_operation_is_published_with_stable_id() -> None:
    assert TOOLS[0].operation_id == "polynomial.root_critical_distance_profile.compute"


def test_request_wire_model_is_not_a_native_export() -> None:
    import jacobian.math.polynomials.root_critical as package

    assert "RootCriticalDistanceProfileRequest" not in package.__all__
    profile = root_critical_distance_profile(_polynomial((4, 1), (0, -2)))
    assert profile.pairs


def test_primitive_derivative_factor_height_is_admitted() -> None:
    scale = 10**128 - 1
    source = RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=scale, den=1),
                    exponents=(2,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=scale),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(0,),
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(source)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.root_critical.factor_coefficient_bound"
    )


def test_cube_root_cubic_profile_is_admitted() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -2)))
    assert len(result.roots) == 3
    assert len(result.critical_points) == 1
    assert len(result.pairs) == 3
    assert all(row.kind == "POSITIVE" for row in result.pairs)


def test_conjugate_distance_degree_is_admitted() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(_polynomial((4, 1), (1, 1), (0, 1)))
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.root_critical.distance_degree_bound"
    )


def test_factor_coefficients_are_canonical_decimal_integers() -> None:
    import json

    from jacobian.math.number_theory.algebraic_numbers.complex import (
        ComplexAlgebraicValue,
    )
    from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue

    profile = root_critical_distance_profile(_polynomial((3, 1), (0, -1)))
    dumped = json.loads(profile.model_dump_json())
    for root in (*dumped["roots"], *dumped["critical_points"]):
        assert "polynomial" in root["value"]
        assert "factor" not in root
    assert isinstance(profile.critical_points[0].value, RealAlgebraicValue)
    assert any(isinstance(root.value, ComplexAlgebraicValue) for root in profile.roots)


def test_cube_root_cubic_returns_exact_squared_distances() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -2)))
    assert len(result.pairs) == 3
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert values == {(1, 0, 0, -4)}


def test_quartic_with_conjugates_is_refused_before_minpoly() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="distance"):
        root_critical_distance_profile(_polynomial((4, 1), (1, 1), (0, 1)))


def test_quartic_crootof_with_rational_critical_points_has_exact_distances() -> None:
    """A CRootOf source can use the bounded rational-critical regime.

    For ``z^4 - 2 z^2 + 2`` the derivative roots are ``-1, 0, 1``.  The
    distance kernel works from the source factor and a certified root box;
    it never asks SymPy to take a minimal polynomial of two opaque CRootOf
    values.
    """
    result = root_critical_distance_profile(_polynomial((4, 1), (2, -2), (0, 2)))

    assert len(result.roots) == 4
    assert len(result.critical_points) == 3
    assert len(result.pairs) == 12
    assert {tuple(row.distance_squared.polynomial) for row in result.pairs} == {
        (1, 0, -2),
        (1, -4, -2, -4, 1),
    }
    assert all(row.isolating_interval.lower.as_fraction() >= 0 for row in result.pairs)


def test_distance_selector_refines_endpoint_overlap_before_selection() -> None:
    """An unrefined ``[1, 2]`` isolator must not match ``[0, 1.0001]``."""
    from fractions import Fraction

    import sympy

    from jacobian.math.polynomials.root_critical.operations import (
        _select_eliminated_distance_root,
    )

    distance = sympy.Symbol("distance")
    with pytest.raises(OperationDomainValidationError, match="isolate"):
        _select_eliminated_distance_root(
            [sympy.Poly(distance**2 - 2, distance, domain=sympy.QQ)],
            Fraction(0),
            Fraction(10001, 10000),
        )


def test_pair_budget_is_rejected_before_exact_root_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(_polynomial((4, 1), (0, -2)), max_pair_rows=0)
    assert (
        error.value.errors()[0]["type"] == "polynomial.root_critical.pair_output_bound"
    )


def test_shifted_quadratic_surd_profile_selects_square_root_distances() -> None:
    # (z-5/2)(z^2-2) = z^3 - (5/2)z^2 - 2z + 5
    source = RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(3,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-5, den=2),
                    exponents=(2,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-2, den=1),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=5, den=1),
                    exponents=(0,),
                ),
            )
        ),
    )
    result = root_critical_distance_profile(source)
    assert len(result.pairs) == 6
    assert all(row.kind == "POSITIVE" for row in result.pairs)
    assert {(1, -12, 4), (4, -1)}.issubset(
        {tuple(row.distance_squared.polynomial) for row in result.pairs}
    )


def test_shifted_quadratic_square_root_pair_selects_the_distance() -> None:
    # (z - 5/2)(z^2 - 2) stays in the square-root grammar; the pair √2 and 2
    # has squared distance 6 - 4√2 whose coarse box is wider than one isolating
    # interval, but unique intersection still selects the smaller root.
    result = root_critical_distance_profile(
        _polynomial((3, 2), (2, -5), (1, -4), (0, 10))
    )
    assert len(result.pairs) >= 1
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert (1, -12, 4) in values


def test_zero_pair_budget_rejects_before_root_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="row budget"):
        root_critical_distance_profile(_polynomial((4, 1), (0, -5)), max_pair_rows=0)


def test_native_pair_budget_matches_catalog_range() -> None:
    polynomial = _polynomial((3, 1), (0, -1))
    with pytest.raises(OperationDomainValidationError, match=r"0\.\.64"):
        root_critical_distance_profile(polynomial, max_pair_rows=65)
    with pytest.raises(OperationDomainValidationError, match="non-boolean"):
        root_critical_distance_profile(polynomial, max_pair_rows="64")  # type: ignore[arg-type]


def test_root_rectangles_ignore_crootof_cache_refinement() -> None:
    import sympy
    from sympy.polys.rootoftools import CRootOf

    CRootOf.clear_cache()
    try:
        variable = sympy.Symbol("z")
        cached = sympy.CRootOf(variable**2 - 2, 0)
        cached.refine()
        cached.eval_rational(n=80)
        result = root_critical_distance_profile(_polynomial((2, 1), (0, -2)))
        digits = [
            max(len(str(abs(component.num))), len(str(component.den)))
            for root in (*result.roots, *result.critical_points)
            for component in (
                root.rectangle.real_lower,
                root.rectangle.real_upper,
                root.rectangle.imaginary_lower,
                root.rectangle.imaginary_upper,
            )
        ]
        assert digits
        assert max(digits) <= 256
        again = root_critical_distance_profile(_polynomial((2, 1), (0, -2)))
        assert again.model_dump() == result.model_dump()
    finally:
        CRootOf.clear_cache()


def test_catalog_example_states_the_univariate_qq_precondition() -> None:
    description = TOOLS[0].examples[0].description
    assert "bounded nonconstant univariate" in description
    assert "QQ" in description


def test_expired_owner_deadline_stops_before_the_sympy_kernel() -> None:
    with (
        request_execution(time.monotonic() - 61.0),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        root_critical_distance_profile(_polynomial((3, 1), (0, -1)))


def test_rectangle_component_rounds_negative_upper_endpoint_up() -> None:
    """A negative upper endpoint must round toward +infinity to stay an upper bound."""
    from fractions import Fraction

    from jacobian.math.polynomials.root_critical._models import (
        MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS,
    )
    from jacobian.math.polynomials.root_critical.operations import (
        _fit_rectangle_component,
    )

    scale = 10 ** (MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS - 1)
    # The scaled value has a nonzero remainder and a negative numerator, so the
    # upper endpoint must round up (toward +infinity) to contain ``value``.
    value = Fraction(-(10**300) * scale * 3 + 1, scale * 3)
    fitted = _fit_rectangle_component(value, round_up=True)
    assert fitted.as_fraction() >= value


def test_evalf_containing_box_scales_with_component_magnitude() -> None:
    """The fallback enclosure must not use a fixed 1e-8 absolute pad."""
    import sympy

    from jacobian.math.polynomials.root_critical.operations import (
        _evalf_containing_box,
    )

    root = sympy.sqrt(1 + 10**200)
    real_lo, real_hi, imag_lo, imag_hi = _evalf_containing_box(root)
    # The magnitude is about 1e100; the enclosure must be far wider than the
    # old fixed 1e-8 pad and must contain the true value.
    assert real_hi - real_lo > 10**50
    assert real_lo <= int(sympy.sqrt(1 + 10**200).evalf(80)) <= real_hi
    assert imag_lo <= 0 <= imag_hi


def test_complex_square_root_enclosure_contains_the_principal_root() -> None:
    from fractions import Fraction

    import sympy

    from jacobian.math.polynomials.root_critical.operations import _enclose_sympy

    for expression in (sympy.sqrt(1 + sympy.I), sympy.sqrt(3 - 2 * sympy.I)):
        real_lo, real_hi, imag_lo, imag_hi = _enclose_sympy(expression)
        true_real, true_imag = expression.as_real_imag()
        # Fifty-digit evaluation error (~1e-50) is far below the enclosure's
        # measured ~1e-31 gap, so these exact comparisons are decisive. A
        # 53-bit float reference is not: the platform libm behind it can round
        # the final ulp either way and straddle a tight endpoint.
        reference_real = Fraction(str(true_real.evalf(50)))
        reference_imag = Fraction(str(true_imag.evalf(50)))
        assert real_lo <= reference_real <= real_hi
        assert imag_lo <= reference_imag <= imag_hi


def test_short_deadline_kills_the_blocking_sympy_kernel() -> None:
    """The blocking SymPy phases run in a killable worker, not in-process.

    A fresh execution with a sub-second deadline must stop during the kernel
    rather than waiting for the unbounded factorization to finish.
    """
    import time as time_module

    started = time_module.monotonic()
    with (
        request_execution(started, outer_deadline=started + 0.05),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        root_critical_distance_profile(_polynomial((3, 1), (0, -1)))
    assert time_module.monotonic() - started < 5.0


def test_fourth_root_radical_is_certified_by_the_enclose_grammar() -> None:
    """z^4-2 roots are fourth roots and must enclose exactly, not heuristically."""
    import sympy

    from jacobian.math.polynomials.root_critical.operations import _enclose_sympy

    z = sympy.Symbol("z")
    roots = sympy.Poly(z**4 - 2, z).all_roots()
    assert len(roots) == 4
    for root in roots:
        real_lo, real_hi, imag_lo, imag_hi = _enclose_sympy(root)
        value = root.evalf(50)
        real, imag = value.as_real_imag()
        assert float(real_lo) <= float(real) <= float(real_hi)
        assert float(imag_lo) <= float(imag) <= float(imag_hi)


def test_native_arguments_are_validated_before_worker_serialization() -> None:
    with pytest.raises(OperationDomainValidationError, match="rational polynomial"):
        root_critical_distance_profile("not a polynomial")  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError, match="non-boolean"):
        root_critical_distance_profile(
            _polynomial((3, 1), (0, -1)),
            max_pair_rows="64",  # type: ignore[arg-type]
        )


def test_worker_decoder_rejects_source_unbound_distance(monkeypatch) -> None:
    """A structurally valid forged row cannot replace the worker result."""
    import json

    from jacobian.canonical import encode_strict_json
    from jacobian.math.polynomials.root_critical import operations

    polynomial = _polynomial((3, 1), (0, -1))
    valid = root_critical_distance_profile(polynomial)
    payload = json.loads(valid.model_dump_json())
    payload["pairs"][0]["distance_squared"] = {
        "polynomial": ["1", "-6"],
        "real_root_index": 0,
    }
    payload["pairs"][0]["isolating_interval"] = {
        "lower": {"num": "6", "den": "1"},
        "upper": {"num": "7", "den": "1"},
        "interval_type": "OPEN",
    }
    forged_output = encode_strict_json({"ok": True, "profile": json.dumps(payload)})
    monkeypatch.setattr(
        operations,
        "run_profile_worker_process",
        lambda *args, **kwargs: forged_output,
    )
    with pytest.raises(
        (OperationDomainValidationError, RuntimeError), match="distance"
    ):
        root_critical_distance_profile(polynomial)


def test_cleared_coefficient_height_is_bounded_before_factorization() -> None:
    """A source whose cleared coefficients exceed the envelope is refused.

    The input components are within the 128-digit source bound, but clearing
    denominators grows the primitive coefficients past the 256-digit factor
    envelope, so the request must fail on the cleared height rather than after
    the expensive factorization.
    """
    import random

    import sympy

    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy

    random.seed(1)
    primes = [sympy.nextprime(random.getrandbits(300)) for _ in range(4)]
    z = sympy.Symbol("z")
    source = sympy.Poly(
        sympy.Rational(1, primes[0]) * z**3
        + sympy.Rational(1, primes[1]) * z**2
        + sympy.Rational(1, primes[2]) * z
        + sympy.Rational(1, primes[3]),
        z,
        domain="QQ",
    )
    primitive = source.clear_denoms(convert=True)[1]
    assert max(len(str(abs(int(c)))) for c in primitive.all_coeffs()) > 256
    polynomial = rational_polynomial_from_sympy(source, ("z",))
    with pytest.raises(OperationResourceAdmissionError, match="cleared primitive"):
        root_critical_distance_profile(polynomial)


def test_cancellation_signal_is_forwarded_to_the_kernel_worker() -> None:
    """A cancellation during the kernel is reported as cancellation."""
    from threading import Event

    from jacobian._execution import (
        OperationExecutionCancelledError,
        request_execution,
    )

    imported = Event()
    imported.set()
    with (
        request_execution(time.monotonic(), cancellation_signal=imported),
        pytest.raises(OperationExecutionCancelledError),
    ):
        root_critical_distance_profile(_polynomial((3, 1), (0, -1)))


def test_nth_root_enclosure_scales_with_the_radicand_magnitude() -> None:
    """A fixed 2**-32 grid cannot separate roots of a large-coefficient radical."""
    from fractions import Fraction

    from jacobian.math.polynomials.root_critical.operations import _nth_root_bounds

    radicand = Fraction(10**80 - 4)
    lower, upper = _nth_root_bounds(radicand, 2)
    # ``sqrt(10**80 - 4)`` sits within 2/10**40 *below* 10**40, so an enclosure
    # of width 2**-32 (about 2.3e-10) would be far too coarse to certify it.
    assert upper - lower < Fraction(1, 10**60)
    assert lower * lower <= radicand <= upper * upper
    # The root is 10**40 - 2 * 10**-40 + O(10**-120), so a usable enclosure
    # stays strictly below the neighbouring value 10**40.
    assert lower < Fraction(10**40) < upper + Fraction(1, 10**39)
    assert upper < Fraction(10**40)


def test_large_coefficient_cubic_rectangles_are_pairwise_isolating() -> None:
    """The small root of z(z**2 - A z + 1) must not share a box with root 0."""
    from fractions import Fraction

    scale = 10**40
    profile = root_critical_distance_profile(_polynomial((3, 1), (2, -scale), (1, 1)))
    rectangles = [
        (row.rectangle.real_lower.as_fraction(), row.rectangle.real_upper.as_fraction())
        for row in profile.roots
    ]
    for index, (lower, upper) in enumerate(rectangles):
        assert lower <= upper
        for other_index, (other_lower, _other_upper) in enumerate(rectangles):
            if other_index == index:
                continue
            assert not (lower <= other_lower <= upper), (
                f"rectangle {index} contains rectangle {other_index}"
            )
    # The two nonzero roots bracket 10**40 and 1/10**40; only the origin root
    # may be pinned at zero.
    zero_rows = [
        index for index, (lower, upper) in enumerate(rectangles) if lower == 0 == upper
    ]
    assert len(zero_rows) == 1
    small = min(
        (
            upper
            for index, (lower, upper) in enumerate(rectangles)
            if index not in zero_rows
        ),
        key=abs,
    )
    assert Fraction(1, 10**41) < small <= Fraction(1, 10**39)


def test_rectangles_are_isolating_across_factors() -> None:
    """A near-rational irrational root excludes the distinct rational sibling.

    ``(z - r)(z^2 - 2)`` with a 40-digit rational approximation ``r`` of
    ``sqrt(2)`` puts ``r`` inside ``sqrt(2)``'s naive rectangle; the refinement
    must shrink it so each rectangle contains exactly one root.
    """
    from fractions import Fraction

    import sympy

    from jacobian.math.polynomials.root_critical.operations import _family

    rational = Fraction(14142135623730950488016887242096980785696, 10**40)
    z = sympy.Symbol("z")
    polynomial = sympy.Poly(
        (z - sympy.Rational(rational.numerator, rational.denominator)) * (z**2 - 2),
        z,
        domain="QQ",
    )
    records, _values = _family(polynomial)
    for record in records:
        real_lower = record.rectangle.real_lower.as_fraction()
        real_upper = record.rectangle.real_upper.as_fraction()
        imaginary_lower = record.rectangle.imaginary_lower.as_fraction()
        imaginary_upper = record.rectangle.imaginary_upper.as_fraction()
        contains_rational = (
            real_lower <= rational <= real_upper
            and imaginary_lower <= 0 <= imaginary_upper
        )
    # Only the rational root's own singleton rectangle may contain it.
    if contains_rational:
        assert real_lower == real_upper == rational


def test_rectangles_separate_irrational_siblings() -> None:
    """Refinement must exclude irrational siblings, not only rational values.

    ``(z**2 - 2)(z**2 - (2 + 1/q))`` with ``q = 10**127`` puts the distinct
    root ``sqrt(2 + 1/q)`` only about ``3.5e-128`` above ``sqrt(2)``, inside
    ``sqrt(2)``'s naive roughly ``2**-97`` wide enclosure. A comprehension that
    retained only rational sibling values would never detect this collision and
    would publish a non-isolating rectangle.
    """
    import sympy

    from jacobian.math.polynomials.root_critical.operations import _family

    q = 10**127
    z = sympy.Symbol("z")
    polynomial = sympy.Poly(
        (z**2 - 2) * (z**2 - (2 + sympy.Rational(1, q))), z, domain="QQ"
    )
    records, _values = _family(polynomial)
    assert len(records) == 4
    intervals = [
        (
            record.rectangle.real_lower.as_fraction(),
            record.rectangle.real_upper.as_fraction(),
        )
        for record in records
    ]
    for lower, upper in intervals:
        assert lower <= upper
    for index, (lower, upper) in enumerate(intervals):
        for other_index, (other_lower, other_upper) in enumerate(intervals):
            if other_index <= index:
                continue
            assert upper < other_lower or other_upper < lower, (
                f"rectangle {index} overlaps rectangle {other_index}"
            )
    # The two close roots are about 3.5e-128 apart, so a separating rectangle
    # must be narrower than that gap.
    gap = sympy.N(sympy.sqrt(2 + sympy.Rational(1, q)) - sympy.sqrt(2), 150)
    assert 0 < gap < sympy.Float("1e-127")


def test_public_operation_isolates_irrational_siblings() -> None:
    """The public profile refuses to publish a non-isolating rectangle.

    ``(z**2 - 2)(z**2 - (2 + 1/q))`` with ``q = 10**127`` is run through the
    final public operation; every published axis rectangle must contain exactly
    one support root, so the near-degenerate ``sqrt(2)``/``sqrt(2 + 1/q)`` pair
    cannot share a rectangle.
    """
    import sympy
    from tests.math.polynomials.root_critical._isolation_invariants import (
        require_axis_rectangles_are_isolating,
    )

    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy

    q = 10**127
    z = sympy.Symbol("z")
    source = sympy.Poly(
        (z**2 - 2) * (z**2 - (2 + sympy.Rational(1, q))), z, domain="QQ"
    )
    profile = root_critical_distance_profile(
        rational_polynomial_from_sympy(source, ("z",))
    )
    require_axis_rectangles_are_isolating(
        source.sqf_part(), tuple(root.rectangle for root in profile.roots)
    )
    derivative_support = source.diff().sqf_part()
    require_axis_rectangles_are_isolating(
        derivative_support,
        tuple(point.rectangle for point in profile.critical_points),
    )


def test_simplest_rational_between_uses_minimal_denominators() -> None:
    """The sibling-separation helper returns the simplest interior rational."""
    from fractions import Fraction

    from jacobian.math.polynomials.root_critical.operations import (
        _simplest_rational_between,
    )

    assert _simplest_rational_between(Fraction(0), Fraction(10)) == Fraction(1)
    assert _simplest_rational_between(Fraction(1, 3), Fraction(1, 2)) == Fraction(2, 5)
    assert _simplest_rational_between(Fraction(2), Fraction(3)) == Fraction(5, 2)
    assert _simplest_rational_between(Fraction(-5, 2), Fraction(-9, 4)) == Fraction(
        -7, 3
    )


def test_rectangles_separate_pell_siblings_after_fitting() -> None:
    """A 256-digit Pell gap survives carrier-grid fitting as isolating.

    ``(z - p/q)(z^2 - 2)`` with the Pell solution ``p^2 - 2q^2 = 1`` puts
    ``p/q`` about ``5.72e-256`` above ``sqrt(2)``, inside one ``10**-255``
    carrier cell. The refined interval excludes ``p/q`` but grid fitting
    would republish the shared cell, so the separation must use
    carrier-representable bounds instead.
    """
    from fractions import Fraction

    import sympy

    from jacobian.math.polynomials.root_critical.operations import _family

    numerator = int(
        "3516000330154850977384610988889313749623219658947581531941618646"
        "4662997079480091099971328100179573232051463624273585519792514883"
    )
    denominator = int(
        "2486187676106635063536074971212188444452320723524864312416871281"
        "6086617787865585526449362118108604080589062492969360100219935438"
    )
    assert numerator * numerator - 2 * denominator * denominator == 1
    rational = Fraction(numerator, denominator)
    z = sympy.Symbol("z")
    polynomial = sympy.Poly(
        (z - sympy.Rational(numerator, denominator)) * (z**2 - 2),
        z,
        domain="QQ",
    )
    records, _values = _family(polynomial)
    assert len(records) == 3
    for record in records:
        real_lower = record.rectangle.real_lower.as_fraction()
        real_upper = record.rectangle.real_upper.as_fraction()
        imaginary_lower = record.rectangle.imaginary_lower.as_fraction()
        imaginary_upper = record.rectangle.imaginary_upper.as_fraction()
        contains_rational = (
            real_lower <= rational <= real_upper
            and imaginary_lower <= 0 <= imaginary_upper
        )
        # Only the rational root's own singleton rectangle may contain it.
        if contains_rational:
            assert real_lower == real_upper == rational
