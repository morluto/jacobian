"""Canonical exact nonreal algebraic values and rational isolation evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cmp_to_key, lru_cache
from math import gcd
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
    _real_polynomial_validation,
    _sympy_polynomial_from_coefficients,
)

MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS = 4_096
MAX_COMPLEX_ORDERING_INTERMEDIATE_DIGITS = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"complex_algebraic.{reason}", message)


def _integer_coefficients(
    coefficients: Sequence[CanonicalInteger],
) -> tuple[int, ...]:
    return tuple(parse_canonical_integer(coefficient) for coefficient in coefficients)


def _sympy_polynomial(coefficients: tuple[CanonicalInteger, ...]) -> Any:
    return _sympy_polynomial_from_coefficients(coefficients)


def algebraic_root_separation_denominator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Return ``B`` such that distinct roots have distance greater than ``1/B``.

    For a square-free integer polynomial of degree ``n``, Mignotte's bound is

    ``sep(f) > sqrt(3) / (n^((n+2)/2) * ||f||_2^(n-1))``.

    With coefficient height ``H``, ``||f||_2 <= sqrt(n+1) H``.  Replacing both
    square roots by larger integer factors gives the deliberately conservative
    denominator below.  The public carriers independently require
    irreducibility, hence square-freeness in characteristic zero.
    """

    integers = _integer_coefficients(coefficients)
    degree = len(integers) - 1
    if degree <= 1:
        return 1
    height: int = max(abs(coefficient) for coefficient in integers)
    degree_factor: int = degree ** ((degree + 3) // 2)
    norm_factor: int = ((degree + 1) * height) ** (degree - 1)
    return int(degree_factor * norm_factor)


@lru_cache(maxsize=128)
def _real_part_elimination_polynomial(
    coefficients: tuple[int, ...],
) -> tuple[int, ...]:
    """Eliminate the imaginary coordinate from ``f(u + i*v) = 0`` exactly."""

    import sympy

    real_coordinate, imaginary_coordinate = sympy.symbols("u v", real=True)
    degree = len(coefficients) - 1
    argument = real_coordinate + sympy.I * imaginary_coordinate
    expression = sum(
        coefficient * argument ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    real_part, imaginary_part = map(sympy.expand, expression.as_real_imag())
    resultant = sympy.resultant(real_part, imaginary_part, imaginary_coordinate)
    polynomial = sympy.Poly(resultant, real_coordinate, domain=sympy.ZZ)
    _content, primitive = polynomial.primitive()
    square_free = primitive.sqf_part()
    if square_free.LC() < 0:
        square_free = -square_free
    return tuple(int(coefficient) for coefficient in square_free.all_coeffs())


def algebraic_real_part_separation_denominator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Bound separation between distinct real coordinates of roots of ``f``."""

    integer_coefficients = _integer_coefficients(coefficients)
    elimination = _real_part_elimination_polynomial(integer_coefficients)
    degree = len(elimination) - 1
    if degree <= 1:
        return 1
    height: int = max(abs(coefficient) for coefficient in elimination)
    bound: int = degree ** ((degree + 3) // 2) * ((degree + 1) * height) ** (degree - 1)
    return int(bound)


def algebraic_root_magnitude_numerator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Return an integer strictly above every root magnitude (Cauchy's bound)."""

    integers = _integer_coefficients(coefficients)
    leading = abs(integers[0])
    tail_height = max((abs(coefficient) for coefficient in integers[1:]), default=0)
    return 2 + tail_height // leading


def complex_isolator_component_digit_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Bound decimal component digits of the dyadic evidence constructed here."""

    separation_denominator = algebraic_root_separation_denominator_bound(coefficients)
    grid_bits = separation_denominator.bit_length() + 4
    magnitude_bits = algebraic_root_magnitude_numerator_bound(coefficients).bit_length()
    component_bits = grid_bits + magnitude_bits + 3
    # 0.30103 is a strict decimal upper bound for log10(2).
    return (component_bits * 30_103) // 100_000 + 2


def _sympy_fraction(value: Any) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _indexed_root_approximation(
    polynomial: Any,
    root_index: int,
    error: Fraction,
) -> tuple[Fraction, Fraction]:
    import sympy

    root = sympy.CRootOf(polynomial.as_expr(), root_index, radicals=False)
    tolerance = sympy.Rational(error.numerator, error.denominator)
    approximation = root.eval_rational(dx=tolerance, dy=tolerance)
    real_part, imaginary_part = approximation.as_real_imag()
    return _sympy_fraction(real_part), _sympy_fraction(imaginary_part)


def _root_evidence_parameters(
    coefficients: Sequence[CanonicalInteger],
) -> tuple[int, Fraction]:
    separation_denominator = algebraic_root_separation_denominator_bound(coefficients)
    grid_denominator = 1 << (separation_denominator.bit_length() + 4)
    return grid_denominator, Fraction(1, 16 * grid_denominator)


@lru_cache(maxsize=128)
def _public_to_backend_root_indices_from_count(
    coefficients: tuple[CanonicalInteger, ...],
    real_count: int,
) -> tuple[int, ...]:
    """Map pair-ordered public indexes to SymPy indexes using exact bounds."""

    import sympy

    polynomial = _sympy_polynomial(coefficients)
    degree = polynomial.degree()
    if real_count == degree:
        return tuple(range(degree))

    root_separation = algebraic_root_separation_denominator_bound(coefficients)
    pair_count = (degree - real_count) // 2
    real_part_separation = (
        algebraic_real_part_separation_denominator_bound(coefficients)
        if pair_count > 1
        else 1
    )
    real_part_digits = (real_part_separation.bit_length() * 30_103) // 100_000 + 1
    if real_part_digits > MAX_COMPLEX_ORDERING_INTERMEDIATE_DIGITS:
        raise _validation_error(
            "ordering_intermediate_bound",
            "exact conjugate-pair ordering exceeds the "
            f"{MAX_COMPLEX_ORDERING_INTERMEDIATE_DIGITS:,}-digit "
            "real-coordinate separation bound",
        )
    root_error = Fraction(1, 16 * root_separation)
    real_part_error = Fraction(1, 16 * real_part_separation)
    ordering_error = min(root_error, real_part_error)
    backend_roots = [
        sympy.CRootOf(polynomial.as_expr(), index, radicals=False)
        for index in range(degree)
    ]
    unused = set(range(real_count, degree))
    positive_representatives: list[tuple[int, Fraction, Fraction]] = []
    negative_for_positive: dict[int, int] = {}
    while unused:
        backend_index = min(unused)
        root = backend_roots[backend_index]
        conjugate = sympy.conjugate(root)
        # Real coefficients make the nonreal roots disjoint conjugate pairs.
        # SymPy's exact CRootOf equality therefore guarantees this lookup.
        partner = next(
            candidate
            for candidate in unused
            if candidate != backend_index and backend_roots[candidate] == conjugate
        )

        _left_real, left_imaginary = _indexed_root_approximation(
            polynomial, backend_index, root_error
        )
        positive_index = backend_index if left_imaginary > 0 else partner
        negative_index = partner if positive_index == backend_index else backend_index
        positive_real, positive_imaginary = _indexed_root_approximation(
            polynomial, positive_index, ordering_error
        )
        positive_representatives.append(
            (positive_index, positive_real, positive_imaginary)
        )
        negative_for_positive[positive_index] = negative_index
        unused.remove(backend_index)
        unused.remove(partner)

    def compare(
        left: tuple[int, Fraction, Fraction],
        right: tuple[int, Fraction, Fraction],
    ) -> int:
        _left_index, left_real, left_imaginary = left
        _right_index, right_real, right_imaginary = right
        if left_real + real_part_error < right_real - real_part_error:
            return -1
        if right_real + real_part_error < left_real - real_part_error:
            return 1
        # Mignotte separation for the real-coordinate elimination polynomial
        # proves that overlapping error intervals represent the same real
        # coordinate.  Original-root separation then orders their positive
        # imaginary coordinates.
        if left_imaginary + root_error < right_imaginary - root_error:
            return -1
        if right_imaginary + root_error < left_imaginary - root_error:
            return 1
        return -1 if left_imaginary < right_imaginary else 1

    positive_representatives.sort(key=cmp_to_key(compare))
    public_to_backend = list(range(real_count))
    for positive_index, _real_part, _imaginary_part in positive_representatives:
        public_to_backend.extend(
            (negative_for_positive[positive_index], positive_index)
        )
    return tuple(public_to_backend)


def _public_to_backend_root_indices(
    coefficients: tuple[CanonicalInteger, ...],
) -> tuple[int, ...]:
    _is_irreducible, real_count = _real_polynomial_validation(coefficients)
    return _public_to_backend_root_indices_from_count(coefficients, real_count)


class RationalComplexIsolatingRectangle(StrictModel):
    """A closed rational rectangle carrying one certified nonreal root.

    All four boundary segments are included.  Embedding records additionally
    prove that a rational error box for their selected exact root lies strictly
    inside this rectangle, so boundary-root ambiguities in backend root counts
    cannot affect the public identity.
    """

    real_lower: CanonicalRational
    real_upper: CanonicalRational
    imaginary_lower: CanonicalRational
    imaginary_upper: CanonicalRational
    boundary: Literal["CLOSED"] = "CLOSED"

    @model_validator(mode="before")
    @classmethod
    def require_raw_component_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        for name in (
            "real_lower",
            "real_upper",
            "imaginary_lower",
            "imaginary_upper",
        ):
            component = data.get(name)
            if not isinstance(component, Mapping):
                continue
            for part in ("num", "den"):
                raw = component.get(part)
                if isinstance(raw, str) and len(raw.lstrip("-")) > (
                    MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
                ):
                    raise _validation_error(
                        "isolator_component_bound",
                        "complex isolator components exceed the "
                        f"{MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
                    )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_nonempty_rectangle(self) -> Self:
        if any(
            len(component.num.lstrip("-")) > MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
            or len(component.den) > MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
            for component in (
                self.real_lower,
                self.real_upper,
                self.imaginary_lower,
                self.imaginary_upper,
            )
        ):
            raise _validation_error(
                "isolator_component_bound",
                "complex isolator components exceed the "
                f"{MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
            )
        if self.real_lower.as_fraction() >= self.real_upper.as_fraction():
            raise _validation_error(
                "real_axis_order",
                "complex isolator real bounds must be strictly increasing",
            )
        if self.imaginary_lower.as_fraction() >= self.imaginary_upper.as_fraction():
            raise _validation_error(
                "imaginary_axis_order",
                "complex isolator imaginary bounds must be strictly increasing",
            )
        return self

    def conjugate(self) -> RationalComplexIsolatingRectangle:
        """Reflect this rectangle exactly across the real axis."""

        return RationalComplexIsolatingRectangle(
            real_lower=self.real_lower,
            real_upper=self.real_upper,
            imaginary_lower=CanonicalRational.from_fraction(
                -self.imaginary_upper.as_fraction()
            ),
            imaginary_upper=CanonicalRational.from_fraction(
                -self.imaginary_lower.as_fraction()
            ),
        )


class ComplexAlgebraicValue(StrictModel):
    """One nonreal algebraic number in canonical indexed-root form.

    ``polynomial`` is primitive irreducible in ``ZZ[x]`` with positive leading
    coefficient.  ``root_index`` uses the mathematical global order: all real
    roots increasingly first; then conjugate pairs ordered lexicographically
    by their positive-imaginary representative ``(Re(z), Im(z))``; and within
    each pair the negative root precedes the positive root.  Thus the
    polynomial and index, rather than any one of infinitely many valid
    isolating rectangles, are the value's identity.
    """

    polynomial: tuple[CanonicalInteger, ...] = Field(
        min_length=2,
        max_length=MAX_REAL_ALGEBRAIC_DEGREE + 1,
        description=(
            "Primitive irreducible ZZ[x] coefficients in descending degree, "
            "with positive leading coefficient."
        ),
    )
    root_index: StrictInt = Field(
        ge=0,
        le=MAX_REAL_ALGEBRAIC_DEGREE - 1,
        description=(
            "Index in the real-first, positive-representative conjugate-pair "
            "order declared by ComplexAlgebraicValue."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_polynomial_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        polynomial = data.get("polynomial")
        if isinstance(polynomial, (list, tuple)):
            if len(polynomial) > MAX_REAL_ALGEBRAIC_DEGREE + 1:
                raise _validation_error(
                    "degree_bound",
                    "complex algebraic degree exceeds the bounded root envelope",
                )
            if any(
                isinstance(value, str)
                and len(value.lstrip("-")) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
                for value in polynomial
            ):
                raise _validation_error(
                    "coefficient_bound",
                    "complex algebraic polynomial coefficients exceed the "
                    f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit bound",
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_nonreal_root(self) -> Self:
        coefficients = _integer_coefficients(self.polynomial)
        if coefficients[0] <= 0:
            raise _validation_error(
                "leading_sign",
                "complex algebraic minimal polynomial must have positive leading coefficient",
            )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise _validation_error(
                "not_primitive",
                "complex algebraic minimal polynomial must be primitive over ZZ",
            )

        is_irreducible, real_root_count = _real_polynomial_validation(self.polynomial)
        if not is_irreducible:
            raise _validation_error(
                "not_irreducible",
                "complex algebraic minimal polynomial must be irreducible over QQ",
            )
        if not real_root_count <= self.root_index < len(self.polynomial) - 1:
            raise _validation_error(
                "root_index",
                "root_index must select a nonreal root in the declared global order",
            )
        return self

    @classmethod
    def _from_admitted_polynomial(
        cls,
        *,
        polynomial: tuple[CanonicalInteger, ...],
        root_index: int,
    ) -> ComplexAlgebraicValue:
        """Construct after an owner has admitted the canonical polynomial/root."""

        return cls.model_construct(polynomial=polynomial, root_index=root_index)


def _isolate_complex_algebraic(
    value: ComplexAlgebraicValue,
    *,
    backend_root_index: int,
    candidates: tuple[RationalComplexIsolatingRectangle, ...],
) -> RationalComplexIsolatingRectangle:
    """Select one deterministic admitted rectangle for an indexed root."""

    polynomial = _sympy_polynomial(value.polynomial)
    _grid_denominator, error = _root_evidence_parameters(value.polynomial)
    real_part, imaginary_part = _indexed_root_approximation(
        polynomial, backend_root_index, error
    )
    return next(
        rectangle
        for rectangle in candidates
        if rectangle.real_lower.as_fraction() < real_part - error
        and real_part + error < rectangle.real_upper.as_fraction()
        and rectangle.imaginary_lower.as_fraction() < imaginary_part - error
        and imaginary_part + error < rectangle.imaginary_upper.as_fraction()
    )


def _require_rectangle_selects_root(
    value: ComplexAlgebraicValue,
    rectangle: RationalComplexIsolatingRectangle,
) -> Literal["NEGATIVE_IMAGINARY", "POSITIVE_IMAGINARY"]:
    """Replay exact root count and indexed-root containment for one rectangle."""

    import sympy

    polynomial = _sympy_polynomial(value.polynomial)
    _grid_denominator, error = _root_evidence_parameters(value.polynomial)
    backend_root_index = _public_to_backend_root_indices(value.polynomial)[
        value.root_index
    ]
    real_part, imaginary_part = _indexed_root_approximation(
        polynomial, backend_root_index, error
    )
    if not (
        rectangle.real_lower.as_fraction() < real_part - error
        and real_part + error < rectangle.real_upper.as_fraction()
        and rectangle.imaginary_lower.as_fraction() < imaginary_part - error
        and imaginary_part + error < rectangle.imaginary_upper.as_fraction()
    ):
        raise _validation_error(
            "root_identity",
            "complex isolator does not certify the selected indexed root",
        )

    lower = sympy.Rational(*rectangle.real_lower.as_integer_ratio()) + sympy.I * (
        sympy.Rational(*rectangle.imaginary_lower.as_integer_ratio())
    )
    upper = sympy.Rational(*rectangle.real_upper.as_integer_ratio()) + sympy.I * (
        sympy.Rational(*rectangle.imaginary_upper.as_integer_ratio())
    )
    try:
        root_count = int(polynomial.count_roots(lower, upper))
    except NotImplementedError as exc:
        raise _validation_error(
            "boundary_root",
            "complex isolator boundaries must contain no additional polynomial root",
        ) from exc
    if root_count != 1:
        raise _validation_error(
            "root_count",
            "a complex algebraic isolator must contain exactly one root",
        )

    imaginary_lower = rectangle.imaginary_lower.as_fraction()
    imaginary_upper = rectangle.imaginary_upper.as_fraction()
    if imaginary_upper < 0:
        return "NEGATIVE_IMAGINARY"
    if imaginary_lower > 0:
        return "POSITIVE_IMAGINARY"
    raise _validation_error(
        "half_plane",
        "a nonreal root isolator must lie wholly in one open half-plane",
    )


__all__ = [
    "MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS",
    "MAX_COMPLEX_ORDERING_INTERMEDIATE_DIGITS",
    "ComplexAlgebraicValue",
    "RationalComplexIsolatingRectangle",
    "algebraic_real_part_separation_denominator_bound",
    "algebraic_root_magnitude_numerator_bound",
    "algebraic_root_separation_denominator_bound",
    "complex_isolator_component_digit_bound",
]
