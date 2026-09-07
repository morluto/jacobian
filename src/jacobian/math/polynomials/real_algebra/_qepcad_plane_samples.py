"""Exact conversion of QEPCAD plane-cell samples to canonical point values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_POINT_DEGREE,
    MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
    MAX_PLANE_COMPONENT_POINT_TERMS,
    IsolatedRealPlanePoint,
)
from jacobian.math.polynomials.values import PolynomialVariable

_RATIONAL = r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?"
_ROOT_DESCRIPTION = re.compile(
    rf"the unique root of (?P<polynomial>.*?) between "
    rf"(?P<lower>{_RATIONAL}) and (?P<upper>{_RATIONAL})",
    re.DOTALL,
)
_COORDINATE = re.compile(r"^Coordinate (?P<index>[12]) = (?P<value>.+)$", re.MULTILINE)
_TOKEN = re.compile(r"\s*(?:(?P<number>[0-9]+)|(?P<name>[A-Za-z]+)|(?P<op>[+\-*/^()]))")
_MAX_EXPRESSION_TOKENS = 2_048
_MAX_REFINEMENT_BITS = 32_768


class QepcadSampleError(RuntimeError):
    """QEPCAD did not return a recognized exact sample description."""


class QepcadSampleLimitError(RuntimeError):
    """An exact QEPCAD sample exceeded the admitted point carrier."""


@dataclass(frozen=True, slots=True)
class _RootDescription:
    polynomial: Any
    lower: Fraction
    upper: Fraction


def _parse_bounded_integer(
    source: str,
    *,
    maximum_digits: int,
    label: str,
) -> int:
    """Parse backend decimal syntax without CPython's integer-string ceiling."""

    negative = source.startswith("-")
    digits = source[1:] if negative else source
    if not digits or not digits.isdigit():
        raise QepcadSampleError(f"QEPCAD {label} was not a decimal integer")
    canonical_digits = digits.lstrip("0") or "0"
    if len(canonical_digits) > maximum_digits:
        raise QepcadSampleLimitError(f"QEPCAD {label} digit count exceeded")
    canonical = (
        f"-{canonical_digits}"
        if negative and canonical_digits != "0"
        else canonical_digits
    )
    return parse_canonical_integer(canonical)


class _PolynomialParser:
    def __init__(self, source: str, *, variable: str) -> None:
        self._tokens = self._tokenize(source)
        self._position = 0
        self._variable = variable

    @staticmethod
    def _tokenize(source: str) -> tuple[str, ...]:
        tokens: list[str] = []
        position = 0
        while position < len(source):
            match = _TOKEN.match(source, position)
            if match is None:
                raise QepcadSampleError("QEPCAD polynomial used unsupported syntax")
            token = match.group("number") or match.group("name") or match.group("op")
            tokens.append(token)
            position = match.end()
            if len(tokens) > _MAX_EXPRESSION_TOKENS:
                raise QepcadSampleLimitError("QEPCAD polynomial token count exceeded")
        if not tokens:
            raise QepcadSampleError("QEPCAD returned an empty polynomial")
        return tuple(tokens)

    def _peek(self) -> str | None:
        return (
            self._tokens[self._position] if self._position < len(self._tokens) else None
        )

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise QepcadSampleError("QEPCAD polynomial ended unexpectedly")
        self._position += 1
        return token

    @staticmethod
    def _bounded(
        polynomial: dict[int, Fraction],
    ) -> dict[int, Fraction]:
        if any(exponent > MAX_PLANE_COMPONENT_POINT_DEGREE for exponent in polynomial):
            raise QepcadSampleLimitError("QEPCAD sample degree exceeded")
        if any(
            len(format_canonical_integer(abs(component)))
            > MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
            for coefficient in polynomial.values()
            for component in (coefficient.numerator, coefficient.denominator)
        ):
            raise QepcadSampleLimitError(
                "QEPCAD sample intermediate coefficient height exceeded"
            )
        return polynomial

    @classmethod
    def _add(
        cls,
        left: dict[int, Fraction],
        right: dict[int, Fraction],
        *,
        sign: int = 1,
    ) -> dict[int, Fraction]:
        result = dict(left)
        for exponent, coefficient in right.items():
            value = result.get(exponent, Fraction()) + sign * coefficient
            if value:
                result[exponent] = value
            else:
                result.pop(exponent, None)
        return cls._bounded(result)

    @classmethod
    def _multiply(
        cls,
        left: dict[int, Fraction],
        right: dict[int, Fraction],
    ) -> dict[int, Fraction]:
        result: dict[int, Fraction] = {}
        for left_exponent, left_coefficient in left.items():
            for right_exponent, right_coefficient in right.items():
                exponent = left_exponent + right_exponent
                result[exponent] = (
                    result.get(exponent, Fraction())
                    + left_coefficient * right_coefficient
                )
        return cls._bounded(
            {exponent: value for exponent, value in result.items() if value}
        )

    @classmethod
    def _power(
        cls, polynomial: dict[int, Fraction], exponent: int
    ) -> dict[int, Fraction]:
        if exponent > MAX_PLANE_COMPONENT_POINT_DEGREE:
            raise QepcadSampleLimitError("QEPCAD sample degree exceeded")
        result = {0: Fraction(1)}
        factor = polynomial
        power = exponent
        while power:
            if power & 1:
                result = cls._multiply(result, factor)
            power >>= 1
            if power:
                factor = cls._multiply(factor, factor)
        return result

    def _primary(self) -> dict[int, Fraction]:
        token = self._take()
        if token == "(":
            value = self._expression()
            if self._take() != ")":
                raise QepcadSampleError("QEPCAD polynomial has unmatched parentheses")
        elif token.isdigit():
            value = {
                0: Fraction(
                    _parse_bounded_integer(
                        token,
                        maximum_digits=MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS,
                        label="sample coefficient",
                    )
                )
            }
        elif token == self._variable:
            value = {1: Fraction(1)}
        else:
            raise QepcadSampleError("QEPCAD polynomial used an unknown identifier")
        if self._peek() == "^":
            self._take()
            exponent = self._take()
            if not exponent.isdigit():
                raise QepcadSampleError("QEPCAD polynomial exponent is not natural")
            value = self._power(
                value,
                _parse_bounded_integer(
                    exponent,
                    maximum_digits=len(str(MAX_PLANE_COMPONENT_POINT_DEGREE)),
                    label="sample exponent",
                ),
            )
        return value

    def _factor(self) -> dict[int, Fraction]:
        sign = 1
        while self._peek() in {"+", "-"}:
            if self._take() == "-":
                sign *= -1
        value = self._primary()
        return {exponent: sign * coefficient for exponent, coefficient in value.items()}

    def _term(self) -> dict[int, Fraction]:
        value = self._factor()
        while True:
            token = self._peek()
            if token in {None, "+", "-", ")"}:
                return value
            if token == "*":
                self._take()
                value = self._multiply(value, self._factor())
                continue
            if token == "/":
                self._take()
                divisor = self._factor()
                if tuple(divisor) != (0,) or divisor[0] == 0:
                    raise QepcadSampleError(
                        "QEPCAD polynomial denominator was not a nonzero rational"
                    )
                value = self._bounded(
                    {
                        exponent: coefficient / divisor[0]
                        for exponent, coefficient in value.items()
                    }
                )
                continue
            if token == "(" or token.isdigit() or token == self._variable:
                value = self._multiply(value, self._factor())
                continue
            raise QepcadSampleError("QEPCAD polynomial term used unsupported syntax")

    def _expression(self) -> dict[int, Fraction]:
        value = self._term()
        while self._peek() in {"+", "-"}:
            operator = self._take()
            value = self._add(value, self._term(), sign=-1 if operator == "-" else 1)
        return value

    def parse(self, *, allow_zero: bool = False) -> Any:
        import sympy

        coefficients = self._expression()
        if self._peek() is not None:
            raise QepcadSampleError("QEPCAD polynomial has trailing syntax")
        variable = sympy.Symbol(self._variable)
        polynomial = sympy.Poly.from_dict(
            {
                (exponent,): sympy.Rational(value.numerator, value.denominator)
                for exponent, value in coefficients.items()
            },
            variable,
            domain=sympy.QQ,
        )
        if polynomial.is_zero and not allow_zero:
            raise QepcadSampleError("QEPCAD root polynomial was zero")
        return polynomial


def _parse_fraction(source: str) -> Fraction:
    numerator, separator, denominator = source.partition("/")
    return Fraction(
        _parse_bounded_integer(
            numerator,
            maximum_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
            label="isolating endpoint",
        ),
        _parse_bounded_integer(
            denominator,
            maximum_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
            label="isolating endpoint denominator",
        )
        if separator
        else 1,
    )


def _root_description(match: re.Match[str]) -> _RootDescription:
    polynomial_text = " ".join(match.group("polynomial").split())
    return _RootDescription(
        polynomial=_PolynomialParser(polynomial_text, variable="x").parse(),
        lower=_parse_fraction(match.group("lower")),
        upper=_parse_fraction(match.group("upper")),
    )


def _primitive_integer_polynomial(polynomial: Any) -> Any:
    import sympy

    _denominator, integral = polynomial.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    return sympy.Poly(primitive, primitive.gens[0], domain=sympy.ZZ)


def _contains_one_root(polynomial: Any, lower: Fraction, upper: Fraction) -> bool:
    import sympy

    left = sympy.Rational(lower.numerator, lower.denominator)
    right = sympy.Rational(upper.numerator, upper.denominator)
    if left == right:
        return bool(polynomial.eval(left) == 0)
    if polynomial.eval(left) == 0 or polynomial.eval(right) == 0:
        return False
    return int(polynomial.count_roots(left, right)) == 1


def _canonical_root(description: _RootDescription) -> tuple[Any, int, Any]:
    import sympy

    source = _primitive_integer_polynomial(description.polynomial)
    factors = tuple(
        _primitive_integer_polynomial(sympy.Poly(factor, *source.gens, domain=sympy.QQ))
        for factor, _multiplicity in sympy.factor_list(source.as_expr(), *source.gens)[
            1
        ]
    )
    owners = tuple(
        factor
        for factor in factors
        if _contains_one_root(factor, description.lower, description.upper)
    )
    if len(owners) != 1:
        raise QepcadSampleError("QEPCAD root description did not select one factor")
    polynomial = owners[0]
    if polynomial.degree() > MAX_PLANE_COMPONENT_POINT_DEGREE:
        raise QepcadSampleLimitError("QEPCAD point degree exceeded")
    coefficients = tuple(int(coefficient) for coefficient in polynomial.all_coeffs())
    if len(coefficients) > MAX_PLANE_COMPONENT_POINT_TERMS or any(
        len(format_canonical_integer(abs(coefficient)))
        > MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
        for coefficient in coefficients
    ):
        raise QepcadSampleLimitError("QEPCAD point coefficient height exceeded")

    left = sympy.Rational(description.lower.numerator, description.lower.denominator)
    root_index = int(polynomial.count_roots(-sympy.oo, left))
    if polynomial.eval(left) == 0:
        root_index -= 1
    real_roots = polynomial.real_roots()
    if not 0 <= root_index < len(real_roots):
        raise QepcadSampleError("QEPCAD root index was inconsistent")
    return polynomial, root_index, real_roots[root_index]


def _interval_multiply(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _polynomial_interval(
    polynomial: Any,
    interval: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    value = (Fraction(), Fraction())
    for coefficient in polynomial.all_coeffs():
        value = _interval_multiply(value, interval)
        rational = Fraction(int(coefficient.p), int(coefficient.q))
        value = value[0] + rational, value[1] + rational
    return value


def _root_index_for_expression(
    *,
    alpha_polynomial: Any,
    alpha_root_index: int,
    expression: Any,
    target_polynomial: Any,
) -> int:
    import sympy

    target_intervals = tuple(
        interval for interval, _multiplicity in target_polynomial.intervals()
    )
    alpha_interval, _multiplicity = alpha_polynomial.intervals()[alpha_root_index]
    lower, upper = alpha_interval
    if lower == upper:
        raise QepcadSampleError(
            "a nonconstant primitive expression used rational alpha"
        )
    refinement_bits = 16
    while refinement_bits <= _MAX_REFINEMENT_BITS:
        refined_lower, refined_upper = alpha_polynomial.refine_root(
            lower,
            upper,
            eps=sympy.Rational(1, 1 << refinement_bits),
        )
        value_lower, value_upper = _polynomial_interval(
            expression,
            (
                Fraction(int(refined_lower.p), int(refined_lower.q)),
                Fraction(int(refined_upper.p), int(refined_upper.q)),
            ),
        )
        matches = tuple(
            index
            for index, (target_lower, target_upper) in enumerate(target_intervals)
            if target_lower
            < sympy.Rational(value_lower.numerator, value_lower.denominator)
            and sympy.Rational(value_upper.numerator, value_upper.denominator)
            < target_upper
        )
        if len(matches) == 1:
            return matches[0]
        refinement_bits *= 2
    raise QepcadSampleLimitError(
        "QEPCAD point root selection exceeded refinement bound"
    )


def _canonical_expression_root(
    *,
    alpha_polynomial: Any,
    alpha_root_index: int,
    alpha_root: Any,
    expression: Any,
) -> tuple[Any, int]:
    import sympy

    alpha_variable = expression.gens[0]
    alpha_modulus = sympy.Poly.from_list(
        alpha_polynomial.all_coeffs(),
        gens=alpha_variable,
        domain=sympy.QQ,
    )
    reduced = expression.rem(alpha_modulus)
    if reduced.degree() <= 0:
        value = sympy.Rational(reduced.nth(0))
        polynomial = sympy.Poly(
            value.q * sympy.Symbol("x") - value.p,
            sympy.Symbol("x"),
            domain=sympy.ZZ,
        )
        return polynomial, 0

    alpha = reduced.gens[0]
    target = sympy.Symbol("x")
    value = reduced.as_expr().subs(alpha, alpha_root)
    polynomial = _primitive_integer_polynomial(
        sympy.Poly(sympy.minpoly(value, target), target, domain=sympy.QQ)
    )
    if polynomial.degree() > MAX_PLANE_COMPONENT_POINT_DEGREE:
        raise QepcadSampleLimitError("QEPCAD coordinate minimal degree exceeded")
    coefficients = tuple(int(coefficient) for coefficient in polynomial.all_coeffs())
    if any(
        len(format_canonical_integer(abs(coefficient)))
        > MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
        for coefficient in coefficients
    ):
        raise QepcadSampleLimitError("QEPCAD coordinate height exceeded")
    return polynomial, _root_index_for_expression(
        alpha_polynomial=alpha_polynomial,
        alpha_root_index=alpha_root_index,
        expression=reduced,
        target_polynomial=polynomial,
    )


def _canonical_interval(polynomial: Any, root_index: int) -> ClosedRationalInterval:
    if polynomial.degree() == 1:
        root = Fraction(-int(polynomial.nth(0)), int(polynomial.nth(1)))
        bounds = root, root
    else:
        (lower, upper), _multiplicity = polynomial.intervals()[root_index]
        bounds = (
            Fraction(int(lower.p), int(lower.q)),
            Fraction(int(upper.p), int(upper.q)),
        )
    if any(
        len(format_canonical_integer(abs(component)))
        > MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS
        for bound in bounds
        for component in (bound.numerator, bound.denominator)
    ):
        raise QepcadSampleLimitError("QEPCAD point isolator height exceeded")
    lower_endpoint = CanonicalRational.from_fraction(bounds[0])
    upper_endpoint = CanonicalRational.from_fraction(bounds[1])
    return ClosedRationalInterval(lower=lower_endpoint, upper=upper_endpoint)


def _point_from_coordinate_data(
    coordinate_data: tuple[tuple[Any, int], tuple[Any, int]],
    *,
    axis: tuple[PolynomialVariable, PolynomialVariable],
) -> IsolatedRealPlanePoint:
    coordinates = tuple(
        RealAlgebraicValue._from_admitted_polynomial(
            polynomial=tuple(
                int(coefficient) for coefficient in polynomial.all_coeffs()
            ),
            real_root_index=root_index,
        )
        for polynomial, root_index in coordinate_data
    )
    return IsolatedRealPlanePoint(
        axis=axis,
        coordinates=(coordinates[0], coordinates[1]),
        isolating_box=RationalBox(
            domain="QQ",
            variables=axis,
            intervals=(
                _canonical_interval(coordinate_data[0][0], coordinate_data[0][1]),
                _canonical_interval(coordinate_data[1][0], coordinate_data[1][1]),
            ),
        ),
    )


def _point_coordinate_data(
    point: IsolatedRealPlanePoint,
) -> tuple[tuple[Any, int, Any], tuple[Any, int, Any]]:
    import sympy

    coordinates: list[tuple[Any, int, Any]] = []
    for coordinate, interval in zip(
        point.coordinates,
        point.isolating_box.intervals,
        strict=True,
    ):
        variable = sympy.Symbol("x")
        coordinate_polynomial = sympy.Poly.from_list(
            coordinate.polynomial,
            gens=variable,
            domain=sympy.ZZ,
        )
        recognized = _canonical_root(
            _RootDescription(
                polynomial=coordinate_polynomial,
                lower=interval.lower.as_fraction(),
                upper=interval.upper.as_fraction(),
            )
        )
        recognized_polynomial, recognized_root_index, _root = recognized
        recognized_coefficients = tuple(
            int(coefficient) for coefficient in recognized_polynomial.all_coeffs()
        )
        if (
            recognized_coefficients != coordinate.polynomial
            or recognized_root_index != coordinate.real_root_index
        ):
            raise QepcadSampleError(
                "plane sample box does not isolate its declared canonical coordinate"
            )
        coordinates.append(recognized)
    return coordinates[0], coordinates[1]


def canonicalize_isolated_plane_point(
    point: IsolatedRealPlanePoint,
) -> IsolatedRealPlanePoint:
    """Recognize a supplied point and return its canonical minimal system."""

    coordinates = _point_coordinate_data(point)
    return _point_from_coordinate_data(
        (
            (coordinates[0][0], coordinates[0][1]),
            (coordinates[1][0], coordinates[1][1]),
        ),
        axis=point.axis,
    )


def isolated_plane_point_coordinates(
    point: IsolatedRealPlanePoint,
) -> tuple[Any, Any]:
    """Return the two exact selected SymPy roots after bounded recognition."""

    coordinates = _point_coordinate_data(point)
    return coordinates[0][2], coordinates[1][2]


def parse_qepcad_plane_sample(
    sample: str,
    *,
    axis: tuple[PolynomialVariable, PolynomialVariable],
) -> IsolatedRealPlanePoint:
    """Parse QEPCAD's documented exact sample grammar without evaluation."""

    root_matches = tuple(_ROOT_DESCRIPTION.finditer(sample))
    if not root_matches:
        raise QepcadSampleError("QEPCAD sample omitted its primitive root")
    alpha_polynomial, alpha_root_index, alpha_root = _canonical_root(
        _root_description(root_matches[0])
    )
    coordinate_matches = {
        int(match.group("index")): match.group("value").strip()
        for match in _COORDINATE.finditer(sample)
    }
    if set(coordinate_matches) != {1, 2}:
        raise QepcadSampleError("QEPCAD sample omitted a plane coordinate")

    coordinate_data: tuple[tuple[Any, int], tuple[Any, int]]
    if "EXTENDED representation" in sample:
        if len(root_matches) < 3:
            raise QepcadSampleError("QEPCAD extended sample omitted a coordinate root")
        second_polynomial, second_root_index, _second_root = _canonical_root(
            _root_description(root_matches[-1])
        )
        coordinate_data = (
            (alpha_polynomial, alpha_root_index),
            (second_polynomial, second_root_index),
        )
    elif "PRIMITIVE representation" in sample:
        primitive_coordinate_data = tuple(
            _canonical_expression_root(
                alpha_polynomial=alpha_polynomial,
                alpha_root_index=alpha_root_index,
                alpha_root=alpha_root,
                expression=_PolynomialParser(
                    coordinate_matches[index], variable="alpha"
                ).parse(allow_zero=True),
            )
            for index in (1, 2)
        )
        coordinate_data = (
            primitive_coordinate_data[0],
            primitive_coordinate_data[1],
        )
    else:
        raise QepcadSampleError("QEPCAD sample representation was not recognized")

    return _point_from_coordinate_data(coordinate_data, axis=axis)


__all__ = [
    "QepcadSampleError",
    "QepcadSampleLimitError",
    "canonicalize_isolated_plane_point",
    "isolated_plane_point_coordinates",
    "parse_qepcad_plane_sample",
]
