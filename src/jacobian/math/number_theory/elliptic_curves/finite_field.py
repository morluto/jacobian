"""Exact short-Weierstrass discriminant over one declared finite field.

Only odd characteristic ``p > 3`` is admitted: characteristics 2 and 3 need
generalized Weierstrass models with different formulas and are rejected
structurally before any arithmetic.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"elliptic_curve.finite_field.{reason}", message)


class FiniteFieldShortWeierstrassCurve(StrictModel):
    """A short Weierstrass curve ``y^2 = x^3 + A x + B`` over one finite field.

    Coefficients are bound to exactly the declared field presentation; the
    model tag records that short-Weierstrass formulas apply only in odd
    characteristic above 3. A valid curve value additionally requires
    nonzero discriminant, which the discriminant operation reports rather
    than this structural carrier.
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement
    coefficient_b: FiniteFieldElement
    model: Literal["SHORT_WEIERSTRASS_ODD_CHAR_GT_3"] = (
        "SHORT_WEIERSTRASS_ODD_CHAR_GT_3"
    )

    @model_validator(mode="after")
    def require_shared_presentation(self) -> Self:
        if (
            self.coefficient_a.presentation != self.field
            or self.coefficient_b.presentation != self.field
        ):
            raise _validation_error(
                "coefficient_presentation_mismatch",
                "curve coefficients must use the declared field presentation",
            )
        return self


class FiniteFieldDiscriminantRequest(StrictModel):
    """Inspect one short-Weierstrass coefficient pair, singular or not.

    Structural admission (characteristic above 3, prime characteristic,
    irreducible modulus, shared presentation) runs in
    ``finite_field_discriminant`` before any field arithmetic.
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement = Field(
        description="Coefficient A bound to the declared field presentation."
    )
    coefficient_b: FiniteFieldElement = Field(
        description="Coefficient B bound to the declared field presentation."
    )

    @model_validator(mode="after")
    def require_shared_presentation(self) -> Self:
        if (
            self.coefficient_a.presentation != self.field
            or self.coefficient_b.presentation != self.field
        ):
            raise _validation_error(
                "coefficient_presentation_mismatch",
                "discriminant coefficients must use the declared field presentation",
            )
        return self


class FiniteFieldDiscriminantResult(StrictModel):
    """Source-bound discriminant data of one finite-field coefficient pair.

    ``four_a_cubed`` is ``4A^3``, ``twentyseven_b_squared`` is ``27B^2``,
    and ``discriminant`` is ``Delta = -16(4A^3 + 27B^2)``. The ``j``-invariant
    ``1728 * 4A^3 / (4A^3 + 27B^2)`` is present exactly when the pair is
    nonsingular (``Delta != 0``).
    """

    field: FiniteFieldPresentation
    coefficient_a: FiniteFieldElement
    coefficient_b: FiniteFieldElement
    four_a_cubed: FiniteFieldElement
    twentyseven_b_squared: FiniteFieldElement
    discriminant: FiniteFieldElement
    is_nonsingular: bool
    j_invariant: FiniteFieldElement | None = Field(
        default=None,
        description="j-invariant, present exactly for nonsingular pairs.",
    )

    @model_validator(mode="after")
    def require_discriminant_branches(self) -> Self:
        for label in (
            "coefficient_a",
            "coefficient_b",
            "four_a_cubed",
            "twentyseven_b_squared",
            "discriminant",
        ):
            if getattr(self, label).presentation != self.field:
                raise _validation_error(
                    "result_presentation_mismatch",
                    "discriminant data must use the declared field presentation",
                )
        if self.j_invariant is not None and self.j_invariant.presentation != self.field:
            raise _validation_error(
                "result_presentation_mismatch",
                "discriminant data must use the declared field presentation",
            )
        if self.is_nonsingular == self.discriminant.is_zero:
            raise _validation_error(
                "singularity_mismatch",
                "is_nonsingular must be the negation of discriminant vanishing",
            )
        if (self.j_invariant is None) == self.is_nonsingular:
            raise _validation_error(
                "j_invariant_branch_mismatch",
                "j is present exactly for nonsingular coefficient pairs",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FiniteFieldDiscriminantRequest,
        *,
        four_a_cubed: FiniteFieldElement,
        twentyseven_b_squared: FiniteFieldElement,
        discriminant: FiniteFieldElement,
        is_nonsingular: bool,
        j_invariant: FiniteFieldElement | None,
    ) -> Self:
        """Build one result after the admitted kernel established its values."""

        return cls.model_construct(
            field=request.field,
            coefficient_a=request.coefficient_a,
            coefficient_b=request.coefficient_b,
            four_a_cubed=four_a_cubed,
            twentyseven_b_squared=twentyseven_b_squared,
            discriminant=discriminant,
            is_nonsingular=is_nonsingular,
            j_invariant=j_invariant,
        )


def _coordinates(element: FiniteFieldElement) -> tuple[int, ...]:
    return tuple(element.coordinates)


def _element(
    field: FiniteFieldPresentation, coordinates: tuple[int, ...]
) -> FiniteFieldElement:
    return FiniteFieldElement(presentation=field, coordinates=coordinates)


def _add(
    modulus: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple((a + b) % modulus for a, b in zip(left, right, strict=True))


def _multiply(
    presentation: FiniteFieldPresentation,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Multiply power-basis coordinates modulo the presentation modulus."""

    modulus = presentation.characteristic
    degree = presentation.degree
    modulus_coefficients = presentation.modulus_coefficients
    raw = [0] * (2 * degree - 1)
    for i, a in enumerate(left):
        if a:
            for j, b in enumerate(right):
                raw[i + j] = (raw[i + j] + a * b) % modulus
    for position in range(2 * degree - 2, degree - 1, -1):
        factor = raw[position] % modulus
        if factor:
            for offset in range(degree + 1):
                raw[position - degree + offset] = (
                    raw[position - degree + offset]
                    - factor * modulus_coefficients[offset]
                ) % modulus
    return tuple(raw[:degree])


def _power(
    presentation: FiniteFieldPresentation,
    value: tuple[int, ...],
    exponent: int,
) -> tuple[int, ...]:
    result = (1,) + (0,) * (presentation.degree - 1)
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(presentation, result, base)
        base = _multiply(presentation, base, base)
        remaining >>= 1
    return result


def _inverse(
    presentation: FiniteFieldPresentation, value: tuple[int, ...]
) -> tuple[int, ...]:
    """Invert a nonzero element by the extended Euclidean algorithm over Fp."""

    modulus = presentation.characteristic
    modulus_poly = list(presentation.modulus_coefficients)

    def degree(poly: list[int]) -> int:
        index = len(poly) - 1
        while index > 0 and poly[index] % modulus == 0:
            index -= 1
        return index

    def normalize(poly: list[int]) -> list[int]:
        trimmed = [c % modulus for c in poly[: degree(poly) + 1]]
        while len(trimmed) > 1 and trimmed[-1] == 0:
            trimmed.pop()
        return trimmed

    remainder, current = normalize(modulus_poly), normalize(list(value))
    old, new = [0], [1]
    while not (len(current) == 1 and current[0] % modulus == 0):
        quotient, leftover = _poly_divmod(remainder, current, modulus)
        remainder, current = current, leftover
        old, new = new, _poly_sub(old, _poly_mul(quotient, new, modulus), modulus)
    if len(remainder) != 1 or remainder[0] % modulus == 0:
        raise ArithmeticError("element is not invertible")
    scale = pow(remainder[0] % modulus, -1, modulus)
    inverse = [(c * scale) % modulus for c in old]
    return tuple(
        inverse[i] if i < len(inverse) else 0 for i in range(presentation.degree)
    )


def _poly_mul(left: list[int], right: list[int], modulus: int) -> list[int]:
    if not left or not right:
        return [0]
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = (result[i + j] + a * b) % modulus
    return result


def _poly_sub(left: list[int], right: list[int], modulus: int) -> list[int]:
    width = max(len(left), len(right))
    return [
        ((left[i] if i < len(left) else 0) - (right[i] if i < len(right) else 0))
        % modulus
        for i in range(width)
    ]


def _poly_divmod(
    dividend: list[int], divisor: list[int], modulus: int
) -> tuple[list[int], list[int]]:
    remainder = [c % modulus for c in dividend]
    divisor = [c % modulus for c in divisor]
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    divisor_degree = len(divisor) - 1
    leading_inverse = pow(divisor[-1], -1, modulus)
    quotient = [0] * max(len(remainder) - divisor_degree, 0)
    while len(remainder) - 1 >= divisor_degree and any(c % modulus for c in remainder):
        shift = len(remainder) - 1 - divisor_degree
        factor = remainder[-1] * leading_inverse % modulus
        if shift < len(quotient):
            quotient[shift] = (quotient[shift] + factor) % modulus
        for i, c in enumerate(divisor):
            remainder[shift + i] = (remainder[shift + i] - factor * c) % modulus
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    return quotient, remainder


def _scale(modulus: int, scalar: int, value: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((scalar * a) % modulus for a in value)


def require_discriminant_admission(
    field: FiniteFieldPresentation,
    coefficient_a: FiniteFieldElement,
    coefficient_b: FiniteFieldElement,
) -> tuple[FiniteFieldPresentation, FiniteFieldElement, FiniteFieldElement]:
    """Admit the finite-field discriminant domain once per call."""

    if not isinstance(field, FiniteFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="elliptic_curve.finite_field.field_type",
            message="field must be a finite-field presentation value",
        )
    try:
        field = FiniteFieldPresentation.model_validate(field.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("field",),
            code="elliptic_curve.finite_field.invalid_field",
            message="field has malformed finite-field presentation data",
        ) from exc
    canonical_coefficients: list[FiniteFieldElement] = []
    for label, coefficient in (
        ("coefficient_a", coefficient_a),
        ("coefficient_b", coefficient_b),
    ):
        if not isinstance(coefficient, FiniteFieldElement):
            raise OperationDomainValidationError(
                location=(label,),
                code="elliptic_curve.finite_field.coefficient_type",
                message="curve coefficients must be finite-field element values",
            )
        try:
            coefficient = FiniteFieldElement.model_validate(coefficient.model_dump())
        except (
            ValidationError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise OperationDomainValidationError(
                location=(label,),
                code="elliptic_curve.finite_field.invalid_coefficient",
                message="curve coefficient has malformed presentation or coordinates",
            ) from exc
        if coefficient.presentation != field:
            raise OperationDomainValidationError(
                location=(label,),
                code="elliptic_curve.finite_field.coefficient_presentation_mismatch",
                message="curve coefficients must use the declared field presentation",
            )
        coordinates = getattr(coefficient, "coordinates", None)
        if (
            type(coordinates) is not tuple
            or len(coordinates) != field.degree
            or any(
                type(value) is not int or not 0 <= value < field.characteristic
                for value in coordinates
            )
        ):
            raise OperationDomainValidationError(
                location=(label, "coordinates"),
                code="elliptic_curve.finite_field.coefficient_coordinates",
                message="coefficient coordinates must be canonical field residues",
            )
        canonical_coefficients.append(coefficient)
    if field.characteristic <= 3:
        raise OperationDomainValidationError(
            location=("field", "characteristic"),
            code="elliptic_curve.finite_field.characteristic_above_three",
            message=(
                "short Weierstrass discriminant requires odd characteristic "
                "above 3; characteristics 2 and 3 need generalized models"
            ),
        )
    # Establish the caller-authored field claim (prime characteristic and
    # irreducible modulus) before any quotient-ring arithmetic relies on it.
    require_field(field)
    return field, canonical_coefficients[0], canonical_coefficients[1]


def finite_field_discriminant(
    field: FiniteFieldPresentation,
    coefficient_a: FiniteFieldElement,
    coefficient_b: FiniteFieldElement,
) -> FiniteFieldDiscriminantResult:
    """Compute 4A^3, 27B^2, Delta, singularity, and j over one finite field."""

    field, coefficient_a, coefficient_b = require_discriminant_admission(
        field, coefficient_a, coefficient_b
    )
    modulus = field.characteristic
    a = _coordinates(coefficient_a)
    b = _coordinates(coefficient_b)
    four_a_cubed = _scale(modulus, 4, _power(field, a, 3))
    twentyseven_b_squared = _scale(modulus, 27, _multiply(field, b, b))
    total = _add(modulus, four_a_cubed, twentyseven_b_squared)
    discriminant = _scale(modulus, modulus - 16, total)
    is_nonsingular = any(discriminant)
    j_invariant: FiniteFieldElement | None = None
    if is_nonsingular:
        ratio = _multiply(
            field, _scale(modulus, 1728, four_a_cubed), _inverse(field, total)
        )
        j_invariant = _element(field, ratio)
    return FiniteFieldDiscriminantResult._from_kernel(
        FiniteFieldDiscriminantRequest(
            field=field, coefficient_a=coefficient_a, coefficient_b=coefficient_b
        ),
        four_a_cubed=_element(field, four_a_cubed),
        twentyseven_b_squared=_element(field, twentyseven_b_squared),
        discriminant=_element(field, discriminant),
        is_nonsingular=is_nonsingular,
        j_invariant=j_invariant,
    )


class FiniteFieldEllipticPoint(StrictModel):
    """A projective point bound to one nonsingular short-Weierstrass curve."""

    curve: FiniteFieldShortWeierstrassCurve
    at_infinity: bool = False
    x: FiniteFieldElement | None = None
    y: FiniteFieldElement | None = None

    @model_validator(mode="after")
    def require_point_shape(self) -> Self:
        if self.at_infinity:
            if self.x is not None or self.y is not None:
                raise _validation_error(
                    "infinity_coordinates", "infinity has no affine coordinates"
                )
        elif self.x is None or self.y is None:
            raise _validation_error(
                "affine_coordinates", "an affine point requires x and y"
            )
        elif (
            self.x.presentation != self.curve.field
            or self.y.presentation != self.curve.field
        ):
            raise _validation_error(
                "point_presentation_mismatch",
                "point coordinates must use the curve field",
            )
        return self

    @classmethod
    def infinity(cls, curve: FiniteFieldShortWeierstrassCurve) -> Self:
        return cls.model_construct(curve=curve, at_infinity=True, x=None, y=None)

    @classmethod
    def affine(
        cls,
        curve: FiniteFieldShortWeierstrassCurve,
        x: FiniteFieldElement,
        y: FiniteFieldElement,
    ) -> Self:
        return cls.model_construct(curve=curve, at_infinity=False, x=x, y=y)


class FiniteFieldCurveRequest(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve


class FiniteFieldPointRequest(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint


class FiniteFieldPointAdditionRequest(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    first: FiniteFieldEllipticPoint
    second: FiniteFieldEllipticPoint


class FiniteFieldScalarRequest(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint
    scalar: int = Field(ge=-1_000_000, le=1_000_000)


class FiniteFieldPointResult(StrictModel):
    point: FiniteFieldEllipticPoint


class FiniteFieldPointCheckResult(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint
    on_curve: bool


class FiniteFieldPointSet(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    points: tuple[FiniteFieldEllipticPoint, ...]
    complete: Literal["EXHAUSTIVE_OVER_F_Q"] = "EXHAUSTIVE_OVER_F_Q"


class FiniteFieldCardinalityResult(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    cardinality: int = Field(ge=1)
    trace: int
    frobenius_polynomial: tuple[int, int, int]


def _curve_admit(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldShortWeierstrassCurve:
    if not isinstance(curve, FiniteFieldShortWeierstrassCurve):
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.curve_type",
            message="curve must be a finite-field short-Weierstrass value",
        )
    try:
        curve = FiniteFieldShortWeierstrassCurve.model_validate(curve.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.invalid_curve",
            message="curve has malformed field or coefficient data",
        ) from exc
    require_discriminant_admission(
        curve.field, curve.coefficient_a, curve.coefficient_b
    )
    data = finite_field_discriminant(
        curve.field, curve.coefficient_a, curve.coefficient_b
    )
    if not data.is_nonsingular:
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.singular_curve",
            message="point arithmetic requires a nonsingular curve",
        )
    return curve


def _canonical_point(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldEllipticPoint:
    """Re-admit a point carrier before inspecting its infinity branch."""

    if not isinstance(point, FiniteFieldEllipticPoint):
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_type",
            message="point must be a finite-field elliptic point value",
        )
    # bool is intentionally checked separately: Pydantic's ordinary bool field
    # accepts values such as ``"yes"``, while native authored values must retain
    # the same strict contract as serialized values.
    if type(getattr(point, "at_infinity", None)) is not bool:
        raise OperationDomainValidationError(
            location=("point", "at_infinity"),
            code="elliptic_curve.finite_field.point_infinity_type",
            message="point at_infinity must be a strict boolean",
        )
    try:
        validated = FiniteFieldEllipticPoint.model_validate(point.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        affine_shape = point.at_infinity is False and (
            getattr(point, "x", None) is not None
            or getattr(point, "y", None) is not None
        )
        raise OperationDomainValidationError(
            location=("point", "coordinates") if affine_shape else ("point",),
            code=(
                "elliptic_curve.finite_field.point_coordinates"
                if affine_shape
                else "elliptic_curve.finite_field.invalid_point"
            ),
            message=(
                "point coordinates must be canonical elements of the curve field"
                if affine_shape
                else "point has malformed curve, infinity, or coordinate data"
            ),
        ) from exc
    if validated.curve != curve:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.parent_curve_mismatch",
            message="point must carry the supplied curve",
        )
    # Rebind the validated coordinates to the exact curve object supplied by the
    # consumer.  Results therefore never retain a nested forged parent carrier.
    return FiniteFieldEllipticPoint.model_construct(
        curve=curve,
        at_infinity=validated.at_infinity,
        x=validated.x,
        y=validated.y,
    )


def _point_admit(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldEllipticPoint:
    curve = _curve_admit(curve)
    point = _canonical_point(curve, point)
    if point.at_infinity:
        return point
    if point.x is None or point.y is None:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_coordinates",
            message="an affine point must carry both field coordinates",
        )
    if (
        point.x.presentation != curve.field
        or point.y.presentation != curve.field
        or type(point.x.coordinates) is not tuple
        or type(point.y.coordinates) is not tuple
        or len(point.x.coordinates) != curve.field.degree
        or len(point.y.coordinates) != curve.field.degree
        or any(
            type(value) is not int or not 0 <= value < curve.field.characteristic
            for value in (*point.x.coordinates, *point.y.coordinates)
        )
    ):
        raise OperationDomainValidationError(
            location=("point", "coordinates"),
            code="elliptic_curve.finite_field.point_coordinates",
            message="point coordinates must be canonical elements of the curve field",
        )
    lhs = _multiply(curve.field, _coordinates(point.y), _coordinates(point.y))
    rhs = _add(
        curve.field.characteristic,
        _multiply(
            curve.field,
            _multiply(curve.field, _coordinates(point.x), _coordinates(point.x)),
            _coordinates(point.x),
        ),
        _add(
            curve.field.characteristic,
            _multiply(
                curve.field, _coordinates(curve.coefficient_a), _coordinates(point.x)
            ),
            _coordinates(curve.coefficient_b),
        ),
    )
    if lhs != rhs:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_off_curve",
            message="point must lie on the curve",
        )
    return point


def _negate_point(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldEllipticPoint:
    point = _point_admit(curve, point)
    curve = point.curve
    if point.at_infinity:
        return FiniteFieldEllipticPoint.infinity(curve)
    assert point.x is not None and point.y is not None
    p = curve.field.characteristic
    return FiniteFieldEllipticPoint.affine(
        curve,
        point.x,
        _element(curve.field, tuple((-v) % p for v in point.y.coordinates)),
    )


def _add_points(
    curve: FiniteFieldShortWeierstrassCurve,
    first: FiniteFieldEllipticPoint,
    second: FiniteFieldEllipticPoint,
) -> FiniteFieldEllipticPoint:
    first = _point_admit(curve, first)
    second = _point_admit(first.curve, second)
    curve = first.curve
    if first.at_infinity:
        return second
    if second.at_infinity:
        return first
    assert (
        first.x is not None
        and first.y is not None
        and second.x is not None
        and second.y is not None
    )
    p = curve.field.characteristic
    f = curve.field
    x1, y1, x2, y2 = (
        first.x.coordinates,
        first.y.coordinates,
        second.x.coordinates,
        second.y.coordinates,
    )
    if (
        x1 == x2
        and tuple((a + b) % p for a, b in zip(y1, y2, strict=True)) == (0,) * f.degree
    ):
        return FiniteFieldEllipticPoint.infinity(curve)
    if first == second:
        numerator = _add(
            p,
            _scale(p, 3, _multiply(f, _coordinates(first.x), _coordinates(first.x))),
            _coordinates(curve.coefficient_a),
        )
        denominator = _scale(p, 2, _coordinates(first.y))
    else:
        numerator = _add(
            p, _coordinates(second.y), tuple(-v % p for v in _coordinates(first.y))
        )
        denominator = _add(
            p, _coordinates(second.x), tuple(-v % p for v in _coordinates(first.x))
        )
    slope = _multiply(f, numerator, _inverse(f, denominator))
    x3 = _add(
        p,
        _add(
            p, _multiply(f, slope, slope), tuple(-v % p for v in _coordinates(first.x))
        ),
        tuple(-v % p for v in _coordinates(second.x)),
    )
    y3 = _add(
        p,
        _multiply(f, slope, _add(p, _coordinates(first.x), tuple(-v % p for v in x3))),
        tuple(-v % p for v in _coordinates(first.y)),
    )
    return FiniteFieldEllipticPoint.affine(curve, _element(f, x3), _element(f, y3))


def finite_field_point_check(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldPointCheckResult:
    curve = _curve_admit(curve)
    canonical_point = _canonical_point(curve, point)
    if canonical_point.at_infinity:
        return FiniteFieldPointCheckResult(
            curve=curve, point=canonical_point, on_curve=True
        )
    try:
        _point_admit(curve, canonical_point)
    except OperationDomainValidationError as exc:
        if exc.errors()[0]["type"] == "elliptic_curve.finite_field.point_off_curve":
            return FiniteFieldPointCheckResult(
                curve=curve, point=canonical_point, on_curve=False
            )
        raise
    return FiniteFieldPointCheckResult(
        curve=curve, point=canonical_point, on_curve=True
    )


def finite_field_point_negate(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldPointResult:
    return FiniteFieldPointResult(point=_negate_point(curve, point))


def finite_field_point_add(
    curve: FiniteFieldShortWeierstrassCurve,
    first: FiniteFieldEllipticPoint,
    second: FiniteFieldEllipticPoint,
) -> FiniteFieldPointResult:
    return FiniteFieldPointResult(point=_add_points(curve, first, second))


def finite_field_point_scalar(
    curve: FiniteFieldShortWeierstrassCurve,
    point: FiniteFieldEllipticPoint,
    scalar: int,
) -> FiniteFieldPointResult:
    if type(scalar) is not int:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="elliptic_curve.finite_field.scalar_type",
            message="scalar must be an integer",
        )
    if abs(scalar) > 1_000_000:
        raise OperationResourceAdmissionError(
            location=("scalar",),
            code="elliptic_curve.finite_field.scalar_bound",
            message="scalar magnitude exceeds the admitted double-and-add envelope",
        )
    point = _point_admit(curve, point)
    curve = point.curve
    if scalar < 0:
        return finite_field_point_scalar(curve, _negate_point(curve, point), -scalar)
    result = FiniteFieldEllipticPoint.infinity(curve)
    addend = point
    n = scalar
    while n:
        if n & 1:
            result = _add_points(curve, result, addend)
        n >>= 1
        if n:
            addend = _add_points(curve, addend, addend)
    return FiniteFieldPointResult(point=result)


def finite_field_points(curve: FiniteFieldShortWeierstrassCurve) -> FiniteFieldPointSet:
    curve = _curve_admit(curve)
    q = curve.field.characteristic**curve.field.degree
    if q > 4096:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.enumeration_bound",
            message="exhaustive point enumeration admits field order at most 4096",
        )
    points = [FiniteFieldEllipticPoint.infinity(curve)]
    # Exact enumeration is intentionally the first bounded cardinality regime.
    for encoded in range(q):
        coords = []
        value = encoded
        for _ in range(curve.field.degree):
            coords.append(value % curve.field.characteristic)
            value //= curve.field.characteristic
        x = _element(curve.field, tuple(coords))
        rhs = _add(
            curve.field.characteristic,
            _add(
                curve.field.characteristic,
                _multiply(
                    curve.field,
                    _multiply(curve.field, tuple(coords), tuple(coords)),
                    tuple(coords),
                ),
                _multiply(
                    curve.field, _coordinates(curve.coefficient_a), tuple(coords)
                ),
            ),
            _coordinates(curve.coefficient_b),
        )
        for y_encoded in range(q):
            ycoords = []
            value = y_encoded
            for _ in range(curve.field.degree):
                ycoords.append(value % curve.field.characteristic)
                value //= curve.field.characteristic
            if _multiply(curve.field, tuple(ycoords), tuple(ycoords)) == rhs:
                points.append(
                    FiniteFieldEllipticPoint.affine(
                        curve, x, _element(curve.field, tuple(ycoords))
                    )
                )
    return FiniteFieldPointSet(curve=curve, points=tuple(points))


def finite_field_cardinality(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldCardinalityResult:
    points = finite_field_points(curve)
    q = curve.field.characteristic**curve.field.degree
    n = len(points.points)
    trace = q + 1 - n
    if trace * trace > 4 * q:
        raise RuntimeError("Hasse identity failed")
    return FiniteFieldCardinalityResult(
        curve=curve, cardinality=n, trace=trace, frobenius_polynomial=(q, -trace, 1)
    )


__all__ = [
    "FiniteFieldCardinalityResult",
    "FiniteFieldCurveRequest",
    "FiniteFieldDiscriminantRequest",
    "FiniteFieldDiscriminantResult",
    "FiniteFieldEllipticPoint",
    "FiniteFieldPointAdditionRequest",
    "FiniteFieldPointCheckResult",
    "FiniteFieldPointRequest",
    "FiniteFieldPointResult",
    "FiniteFieldPointSet",
    "FiniteFieldScalarRequest",
    "FiniteFieldShortWeierstrassCurve",
    "finite_field_cardinality",
    "finite_field_discriminant",
    "finite_field_point_add",
    "finite_field_point_check",
    "finite_field_point_negate",
    "finite_field_point_scalar",
    "finite_field_points",
    "require_discriminant_admission",
]
