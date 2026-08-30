"""Exact conversion of QEPCAD sample descriptions."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_samples import (
    QepcadSampleLimitError,
    parse_qepcad_plane_sample,
)


def test_primitive_rational_sample_becomes_a_canonical_plane_point() -> None:
    point = parse_qepcad_plane_sample(
        """The sample point is in a PRIMITIVE representation.

alpha = the unique root of x between 0 and 0
      = 0.0000000000

Coordinate 1 = 0
             = 0.0000000000
Coordinate 2 = -1/2
             = -0.5000000000""",
        axis=("u", "v"),
    )

    assert point.isolating_box.intervals[0].lower.as_fraction() == 0
    assert point.isolating_box.intervals[1].lower.as_fraction() == Fraction(-1, 2)
    assert point.coordinate_polynomials[1].polynomial.terms[0].coefficient.num == "2"


def test_primitive_algebraic_coordinate_uses_its_minimal_polynomial() -> None:
    point = parse_qepcad_plane_sample(
        """The sample point is in a PRIMITIVE representation.

alpha = the unique root of x^2 - 3 between 3/2 and 2
      = 1.7320508076-

Coordinate 1 = 0
             = 0.0000000000
Coordinate 2 = alpha
             = 1.7320508076-""",
        axis=("x", "y"),
    )

    terms = point.coordinate_polynomials[1].polynomial.terms
    assert tuple((term.coefficient.num, term.exponents) for term in terms) == (
        ("1", (0, 2)),
        ("-3", (0, 0)),
    )
    assert point.isolating_box.intervals[1].lower.as_fraction() > 0


def test_extended_sample_uses_the_univariate_coordinate_projection() -> None:
    point = parse_qepcad_plane_sample(
        """The sample point is in an EXTENDED representation.

alpha = the unique root of x^2 - 2 between 181/128 and 1449/1024
      = 1.4142135624-

Coordinate 1 = alpha
             = 1.4142135624-
Coordinate 2 = the unique root of x - alpha between 0 and 4
             = the unique root of x^2 - 2 between 0 and 4
             = 1.4142135624-""",
        axis=("x", "y"),
    )

    assert point.coordinate_polynomials[0].polynomial.terms[1].coefficient.num == "-2"
    assert point.coordinate_polynomials[1].polynomial.terms[1].coefficient.num == "-2"
    assert point.isolating_box.intervals[0].lower.as_fraction() > 0
    assert point.isolating_box.intervals[1].lower.as_fraction() > 0


def test_large_isolator_denominator_uses_the_bounded_decimal_parser() -> None:
    denominator = "9" * 5_000
    point = parse_qepcad_plane_sample(
        f"""The sample point is in a PRIMITIVE representation.

alpha = the unique root of x between 0/{denominator} and 0/{denominator}
      = 0.0000000000

Coordinate 1 = 0
             = 0.0000000000
Coordinate 2 = 0
             = 0.0000000000""",
        axis=("x", "y"),
    )

    assert point.isolating_box.intervals[0].lower.as_fraction() == 0


def test_isolator_past_the_declared_digit_envelope_is_a_typed_limit() -> None:
    denominator = "9" * (MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS + 1)

    with pytest.raises(QepcadSampleLimitError, match="digit count"):
        parse_qepcad_plane_sample(
            f"""The sample point is in a PRIMITIVE representation.

alpha = the unique root of x between 0/{denominator} and 0/{denominator}
      = 0.0000000000

Coordinate 1 = 0
             = 0.0000000000
Coordinate 2 = 0
             = 0.0000000000""",
            axis=("x", "y"),
        )
