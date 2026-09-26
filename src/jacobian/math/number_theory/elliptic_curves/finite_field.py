"""Exact short-Weierstrass discriminant over one declared finite field.

Only odd characteristic ``p > 3`` is admitted: characteristics 2 and 3 need
generalized Weierstrass models with different formulas and are rejected
structurally before any arithmetic.
"""

from __future__ import annotations

from math import isqrt
from typing import Literal, Self

import rfc8785
from pydantic import Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields._algebraic_sets import (
    FieldEmbedding,
    embed_field_element,
)
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.groups.abelian._models import AbelianPresentation
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
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


class FiniteFieldCurveBaseChangeRequest(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    embedding: FieldEmbedding
    point: FiniteFieldEllipticPoint | None = None


class FiniteFieldCurveBaseChangeResult(StrictModel):
    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint | None = None


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


class FiniteFieldPointOrderRequest(StrictModel):
    """Find one point's exact order using the certified group cardinality."""

    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint


class FiniteFieldPointOrderPrimeWitness(StrictModel):
    """Nonidentity point proving a prime cannot be removed from the order."""

    prime: int = Field(ge=2)
    reduced_scalar: int = Field(ge=1)
    reduced_multiple: FiniteFieldEllipticPoint


class FiniteFieldPointOrderResult(StrictModel):
    """Exact point order with an annihilator and prime-divisor witnesses."""

    curve: FiniteFieldShortWeierstrassCurve
    point: FiniteFieldEllipticPoint
    group_cardinality: int = Field(ge=1)
    order: int = Field(ge=1)
    annihilating_multiple: FiniteFieldEllipticPoint
    prime_divisor_witnesses: tuple[FiniteFieldPointOrderPrimeWitness, ...]

    @model_validator(mode="after")
    def require_order_witness_shape(self) -> Self:
        if (
            self.point.curve != self.curve
            or self.annihilating_multiple.curve != self.curve
        ):
            raise _validation_error(
                "point_order_curve_mismatch",
                "point-order witnesses must retain the supplied curve",
            )
        if not self.annihilating_multiple.at_infinity:
            raise _validation_error(
                "point_order_annihilator",
                "the reported order must annihilate the point",
            )
        if self.order > self.group_cardinality or self.group_cardinality % self.order:
            raise _validation_error(
                "point_order_group_divisibility",
                "the point order must divide the exact curve cardinality",
            )
        expected = _prime_divisors(self.order)
        if tuple(witness.prime for witness in self.prime_divisor_witnesses) != expected:
            raise _validation_error(
                "point_order_prime_witnesses",
                "one ordered minimality witness is required for every prime divisor",
            )
        for witness in self.prime_divisor_witnesses:
            if (
                witness.reduced_scalar != self.order // witness.prime
                or witness.reduced_multiple.curve != self.curve
                or witness.reduced_multiple.at_infinity
            ):
                raise _validation_error(
                    "point_order_minimality",
                    "prime-divisor witnesses must show each reduced multiple is nonidentity",
                )
        return self


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


class FiniteFieldFrobeniusResult(StrictModel):
    """Exact Frobenius polynomial and ordinary/supersingular class."""

    curve: FiniteFieldShortWeierstrassCurve
    cardinality: int = Field(ge=1)
    trace: int
    determinant: int = Field(ge=1)
    characteristic_polynomial: IntegerPolynomial
    discriminant: int
    classification: Literal["ORDINARY", "SUPERSINGULAR"]

    @model_validator(mode="after")
    def require_claim_consistency(self) -> Self:
        q = self.curve.field.characteristic**self.curve.field.degree
        if (
            self.determinant != q
            or self.trace != q + 1 - self.cardinality
            or self.characteristic_polynomial.coefficients != (q, -self.trace, 1)
            or self.discriminant != self.trace * self.trace - 4 * q
            or self.classification
            != (
                "SUPERSINGULAR"
                if self.trace % self.curve.field.characteristic == 0
                else "ORDINARY"
            )
        ):
            raise _validation_error(
                "frobenius_claim_mismatch",
                "Frobenius data must agree with the curve field and trace",
            )
        return self


class FiniteFieldZetaPolynomialResult(StrictModel):
    """Numerator of the zeta function of one finite-field elliptic curve.

    ``numerator`` is the dense integral polynomial ``1 - a*T + q*T^2``;
    ``IntegerPolynomial`` stores coefficients in descending degree order.
    The curve and count fields retain the source context and state the exact
    relation from which the numerator was obtained.
    """

    curve: FiniteFieldShortWeierstrassCurve
    cardinality: int = Field(ge=1)
    trace: int
    numerator: IntegerPolynomial

    @model_validator(mode="after")
    def require_numerator_identity(self) -> Self:
        q = int(self.curve.field.characteristic**self.curve.field.degree)
        if self.trace != q + 1 - self.cardinality or self.numerator.coefficients != (
            q,
            -self.trace,
            1,
        ):
            raise _validation_error(
                "zeta_polynomial_identity",
                "zeta numerator must be 1 - trace*T + q*T^2 for the exact curve count",
            )
        return self


def _zeta_rational_function(q: int, trace: int) -> RationalFunction:
    """Build the normalized QQ(T) form of the finite-field zeta function."""

    def rational_term(
        numerator: int, denominator: int, degree: int
    ) -> RationalPolynomialTerm:
        return RationalPolynomialTerm(
            coefficient=CanonicalRational.from_integer_ratio(numerator, denominator),
            exponents=(degree,),
        )

    numerator_terms = [rational_term(1, 1, 2)]
    if trace:
        numerator_terms.append(rational_term(-trace, q, 1))
    numerator_terms.append(rational_term(1, q, 0))
    denominator_terms = (
        rational_term(1, 1, 2),
        rational_term(-(q + 1), q, 1),
        rational_term(1, q, 0),
    )
    return RationalFunction._from_kernel(
        variables=("T",),
        numerator=SparseRationalPolynomial(terms=tuple(numerator_terms)),
        denominator=SparseRationalPolynomial(terms=denominator_terms),
    )


class FiniteFieldZetaFunctionResult(StrictModel):
    """The exact elliptic zeta function bound to its finite-field curve."""

    curve: FiniteFieldShortWeierstrassCurve
    cardinality: int = Field(ge=1)
    trace: int
    zeta_function: RationalFunction

    @model_validator(mode="after")
    def require_source_bound_zeta_function(self) -> Self:
        q = int(self.curve.field.characteristic**self.curve.field.degree)
        if self.trace != q + 1 - self.cardinality or self.zeta_function != (
            _zeta_rational_function(q, self.trace)
        ):
            raise _validation_error(
                "zeta_function_identity",
                "zeta function must equal (1 - trace*T + q*T^2)/((1-T)(1-q*T)) "
                "for the bound curve count",
            )
        return self


class FiniteFieldGroupStructureResult(StrictModel):
    """Invariant factors with one curve-bound generator for each factor."""

    curve: FiniteFieldShortWeierstrassCurve
    group: AbelianPresentation
    generators: tuple[FiniteFieldEllipticPoint, ...]

    @model_validator(mode="after")
    def require_source_bound_generators(self) -> Self:
        if len(self.generators) != len(self.group.invariant_factors):
            raise _validation_error(
                "group_generator_rank",
                "one point generator is required for each invariant factor",
            )
        if any(generator.curve != self.curve for generator in self.generators):
            raise _validation_error(
                "group_generator_curve_mismatch",
                "group generators must retain the exact source curve",
            )
        return self


class FiniteFieldQuadraticTwistRelation(StrictModel):
    """A canonical nonsquare parameter and its source-bound twist model.

    The producer establishes that ``parameter`` is a nonsquare and that the
    target coefficients are ``d^2*A`` and ``d^3*B``. This carrier preserves
    those exact inputs for consumers that need to use the twist relation; its
    structural validation intentionally does not replay field arithmetic.
    """

    source_curve: FiniteFieldShortWeierstrassCurve
    twisted_curve: FiniteFieldShortWeierstrassCurve
    parameter: FiniteFieldElement

    @model_validator(mode="after")
    def require_one_field_presentation(self) -> Self:
        if (
            self.source_curve.field != self.twisted_curve.field
            or self.parameter.presentation != self.source_curve.field
        ):
            raise _validation_error(
                "twist_relation_field_mismatch",
                "twist relation curves and parameter must share one field presentation",
            )
        return self


MAX_FROBENIUS_EXTENSION_DEGREE = 64
MAX_FROBENIUS_EXTENSION_INTEGER_DIGITS = 4096
MAX_FROBENIUS_CHARACTER_SUM_WORK = 4_000_000
MAX_ISOGENY_PAIR_CHARACTER_SUM_WORK = 8_000_000
MAX_FINITE_FIELD_POINT_ENUMERATION_WORK = 20_000_000
MAX_FINITE_FIELD_TWIST_ORDER = 4096
MAX_FINITE_FIELD_TWIST_WORK = 4_000_000
MAX_FINITE_FIELD_ISOMORPHISM_ORDER = 4096
MAX_FINITE_FIELD_ISOMORPHISM_WORK = 4_000_000
MAX_FINITE_FIELD_GROUP_STRUCTURE_WORK = 50_000_000
MAX_POINT_ORDER_SCALAR_WORK = 100_000
MAX_POINT_ORDER_WITNESSES = 8


class FiniteFieldExtensionCountsRequest(StrictModel):
    """Compute extension-field counts from one exhaustively counted curve."""

    curve: FiniteFieldShortWeierstrassCurve
    max_degree: int = Field(ge=1, le=MAX_FROBENIUS_EXTENSION_DEGREE)


class FiniteFieldExtensionCount(StrictModel):
    """One exact power sum and point count over an extension field."""

    degree: int = Field(ge=1, le=MAX_FROBENIUS_EXTENSION_DEGREE)
    frobenius_power_sum: int
    cardinality: int = Field(ge=1)


class FiniteFieldExtensionCountsResult(StrictModel):
    """Exact counts ``#E(F_(q^n))`` derived from the base Frobenius trace."""

    curve: FiniteFieldShortWeierstrassCurve
    base_cardinality: int = Field(ge=1)
    base_trace: int
    counts: tuple[FiniteFieldExtensionCount, ...]


class FiniteFieldIsogenyClassRequest(StrictModel):
    """Compare two curves over the same exact finite-field presentation."""

    first: FiniteFieldShortWeierstrassCurve
    second: FiniteFieldShortWeierstrassCurve


class FiniteFieldIsogenyClassResult(StrictModel):
    """Exact isogeny-class decision from the two Frobenius polynomials."""

    first: FiniteFieldShortWeierstrassCurve
    second: FiniteFieldShortWeierstrassCurve
    first_cardinality: int = Field(ge=1)
    second_cardinality: int = Field(ge=1)
    first_trace: int
    second_trace: int
    first_frobenius_polynomial: tuple[int, int, int]
    second_frobenius_polynomial: tuple[int, int, int]
    same_isogeny_class: bool

    @model_validator(mode="after")
    def require_frobenius_consistency(self) -> Self:
        if self.first.field != self.second.field:
            raise _validation_error(
                "isogeny_field_mismatch",
                "isogeny result curves must use the same exact field presentation",
            )
        q = self.first.field.characteristic**self.first.field.degree
        expected_first = (q, -(q + 1 - self.first_cardinality), 1)
        expected_second = (q, -(q + 1 - self.second_cardinality), 1)
        if (
            self.first_trace != q + 1 - self.first_cardinality
            or self.second_trace != q + 1 - self.second_cardinality
            or self.first_frobenius_polynomial != expected_first
            or self.second_frobenius_polynomial != expected_second
            or self.same_isogeny_class != (expected_first == expected_second)
        ):
            raise _validation_error(
                "isogeny_frobenius_mismatch",
                "isogeny decision must agree with the exact Frobenius polynomials",
            )
        return self


class FiniteFieldIsomorphismRequest(StrictModel):
    """Compare short models over one identical finite-field presentation."""

    source: FiniteFieldShortWeierstrassCurve
    target: FiniteFieldShortWeierstrassCurve


class FiniteFieldIsomorphismResult(StrictModel):
    """Complete scaling decision for the map (x,y) -> (u^2*x,u^3*y)."""

    source: FiniteFieldShortWeierstrassCurve
    target: FiniteFieldShortWeierstrassCurve
    isomorphic: bool
    scaling: FiniteFieldElement | None = None

    @model_validator(mode="after")
    def require_scaling_witness(self) -> Self:
        if self.source.field != self.target.field:
            raise _validation_error(
                "isomorphism_field_mismatch",
                "curves must use the same exact field presentation",
            )
        if self.isomorphic != (self.scaling is not None):
            raise _validation_error(
                "isomorphism_witness_mismatch",
                "scaling is present exactly for isomorphic curves",
            )
        if self.scaling is not None and self.scaling.presentation != self.source.field:
            raise _validation_error(
                "isomorphism_field_mismatch", "scaling must use the common curve field"
            )
        return self


def _admit_extension_count_growth(
    curve: FiniteFieldShortWeierstrassCurve, degree: int
) -> int:
    q = int(curve.field.characteristic**curve.field.degree)
    if q > 4096:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.enumeration_bound",
            message="extension counts require exhaustive base-field order at most 4096",
        )
    character_sum_work = q * curve.field.degree**2 * (8 + 2 * q.bit_length())
    if character_sum_work > MAX_FROBENIUS_CHARACTER_SUM_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.character_sum_work_bound",
            message="quadratic-character sum exceeds the admitted exact-work envelope",
        )
    # A Hasse-bound trace and the recurrence give a conservative integer-only
    # upper bound for every intermediate power sum, before point enumeration.
    trace_bound = 2 * (isqrt(q) + 1)
    previous_previous, previous = 2, trace_bound
    q_power = q
    for n in range(1, degree + 1):
        if n == 1:
            power_sum_bound = trace_bound
        else:
            current = trace_bound * previous + q * previous_previous
            previous_previous, previous = previous, current
            power_sum_bound = current
            q_power *= q
        count_bound = q_power + 1 + power_sum_bound
        if len(str(count_bound)) > MAX_FROBENIUS_EXTENSION_INTEGER_DIGITS:
            raise OperationResourceAdmissionError(
                location=("max_degree",),
                code="elliptic_curve.finite_field.extension_count_output_bound",
                message="extension count integers exceed the admitted digit envelope",
            )
    return q


def _enumerate_admitted_points(
    curve: FiniteFieldShortWeierstrassCurve, q: int
) -> FiniteFieldPointSet:
    """Enumerate after curve identity and field-order admission have passed."""
    field = curve.field
    characteristic = field.characteristic
    square_roots: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    # Preserve encoded y order within each bucket, so x-major/y-major output
    # ordering is unchanged from the direct q-by-q scan.
    for encoded_y in range(q):
        y = _decode_field_element(field, encoded_y)
        square = _multiply(field, y, y)
        square_roots.setdefault(square, []).append(y)

    points = [FiniteFieldEllipticPoint.infinity(curve)]
    for encoded in range(q):
        coords = _decode_field_element(field, encoded)
        x = _element(field, coords)
        rhs = _add(
            characteristic,
            _add(
                characteristic,
                _multiply(
                    field,
                    _multiply(field, coords, coords),
                    coords,
                ),
                _multiply(field, _coordinates(curve.coefficient_a), coords),
            ),
            _coordinates(curve.coefficient_b),
        )
        for y in square_roots.get(rhs, ()):
            points.append(FiniteFieldEllipticPoint.affine(curve, x, _element(field, y)))
    return FiniteFieldPointSet(curve=curve, points=tuple(points))


def _decode_field_element(
    field: FiniteFieldPresentation, encoded: int
) -> tuple[int, ...]:
    coordinates = []
    value = encoded
    for _ in range(field.degree):
        coordinates.append(value % field.characteristic)
        value //= field.characteristic
    return tuple(coordinates)


def _admit_point_enumeration(
    curve: FiniteFieldShortWeierstrassCurve,
) -> int:
    """Bound field arithmetic and serialized output before expansion."""
    field = curve.field
    q = int(field.characteristic**field.degree)
    # One square per field element and three products per x; each product uses
    # fewer than 2*degree^2 coefficient multiply/reduction steps.
    work = 8 * q * field.degree**2
    if work > MAX_FINITE_FIELD_POINT_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.enumeration_work_bound",
            message="complete point enumeration exceeds its exact-work envelope",
        )

    # Hasse gives #E(F_q) <= q + 1 + floor(2 sqrt(q)).  Every affine point
    # repeats the curve and field in source-bound JSON, so size the worst case
    # before materializing the point tuple.
    max_points = q + 1 + isqrt(4 * q)
    maximum = _element(field, (field.characteristic - 1,) * field.degree)
    sample_point = FiniteFieldEllipticPoint.affine(curve, maximum, maximum)
    point_bytes = len(rfc8785.dumps(sample_point.model_dump(mode="json")))
    empty_result = FiniteFieldPointSet(curve=curve, points=()).model_dump(mode="json")
    output_bytes = (
        len(rfc8785.dumps(empty_result)) + max_points * point_bytes + max_points - 1
    )
    if output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.enumeration_output_bound",
            message="complete point set exceeds the canonical output-byte envelope",
        )
    return q


def _finite_group_factorization(value: int) -> tuple[tuple[int, int], ...]:
    """Return the prime factorization of one already bounded group order."""
    factors: list[tuple[int, int]] = []
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            exponent = 0
            while remaining % divisor == 0:
                remaining //= divisor
                exponent += 1
            factors.append((divisor, exponent))
        divisor = 3 if divisor == 2 else divisor + 2
    if remaining > 1:
        factors.append((remaining, 1))
    return tuple(factors)


def _finite_point_key(point: FiniteFieldEllipticPoint) -> tuple[object, ...]:
    if point.at_infinity:
        return (True,)
    assert point.x is not None and point.y is not None
    return (False, *point.x.coordinates, *point.y.coordinates)


def _point_order_from_admitted_cardinality(
    point: FiniteFieldEllipticPoint,
    cardinality: int,
    prime_factors: tuple[tuple[int, int], ...],
) -> int:
    """Reduce a Lagrange bound to the exact order using scalar multiplication."""
    order = cardinality
    for prime, exponent in prime_factors:
        for _ in range(exponent):
            reduced = order // prime
            if not _scalar_multiply_admitted(point, reduced).at_infinity:
                break
            order = reduced
    return order


def _admit_group_structure_work(curve: FiniteFieldShortWeierstrassCurve) -> int:
    """Preflight enumeration, all point-order reductions, and generator search."""
    q = _admit_point_enumeration(curve)
    maximum_cardinality = q + 1 + 2 * (isqrt(q) + 1)
    if maximum_cardinality > 4096:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.group_structure_order_bound",
            message="elliptic group structure requires Hasse bound at most 4096",
        )
    bits = maximum_cardinality.bit_length()
    degree = curve.field.degree
    # Each point order uses at most log2(N) scalar reductions, each costing at
    # most 2*log2(N) group additions. The independent generator search has the
    # same upper envelope; a group addition is charged for field-coordinate
    # products, inversions, and reductions.
    work_bound = 32 * maximum_cardinality * bits**2 * degree**2 + maximum_cardinality
    if work_bound > MAX_FINITE_FIELD_GROUP_STRUCTURE_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.group_structure_work_bound",
            message="group structure point-order and generator work exceeds its envelope",
        )
    max_element = _element(curve.field, (curve.field.characteristic - 1,) * degree)
    max_point = FiniteFieldEllipticPoint.affine(curve, max_element, max_element)
    sample = FiniteFieldGroupStructureResult.model_construct(
        curve=curve,
        group=AbelianPresentation(invariant_factors=(maximum_cardinality,)),
        generators=(max_point,),
    )
    output_bound = (
        len(rfc8785.dumps(sample.model_dump(mode="json")))
        + len(rfc8785.dumps(max_point.model_dump(mode="json")))
        + 128
    )
    if output_bound > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("curve",),
            code="elliptic_curve.finite_field.group_structure_output_bound",
            message="group structure result exceeds the canonical output-byte envelope",
        )
    return q


def _cardinality_from_points(
    curve: FiniteFieldShortWeierstrassCurve, points: FiniteFieldPointSet, q: int
) -> FiniteFieldCardinalityResult:
    cardinality = len(points.points)
    trace = q + 1 - cardinality
    if trace * trace > 4 * q:
        raise RuntimeError("Hasse identity failed")
    return FiniteFieldCardinalityResult(
        curve=curve,
        cardinality=cardinality,
        trace=trace,
        frobenius_polynomial=(q, -trace, 1),
    )


def _cardinality_from_character_sum(
    curve: FiniteFieldShortWeierstrassCurve, q: int
) -> FiniteFieldCardinalityResult:
    """Count by the quadratic-character sum without materializing points."""
    field = curve.field
    p = field.characteristic
    zero = (0,) * field.degree
    one = (1,) + (0,) * (field.degree - 1)
    trace = 0
    for encoded in range(q):
        coordinates = []
        value = encoded
        for _ in range(field.degree):
            coordinates.append(value % p)
            value //= p
        x = tuple(coordinates)
        x_cubed = _multiply(field, _multiply(field, x, x), x)
        ax = _multiply(field, _coordinates(curve.coefficient_a), x)
        rhs = _add(p, _add(p, x_cubed, ax), _coordinates(curve.coefficient_b))
        if rhs == zero:
            continue
        character = _power(field, rhs, (q - 1) // 2)
        if character == one:
            trace -= 1
        else:
            # Euler's criterion in the admitted finite field says the only
            # other nonzero result is -1. Treat a violated field invariant as
            # an internal error rather than returning a fabricated count.
            minus_one = (p - 1,) + (0,) * (field.degree - 1)
            if character != minus_one:
                raise RuntimeError("quadratic character identity failed")
            trace += 1
    cardinality = q + 1 - trace
    if trace * trace > 4 * q:
        raise RuntimeError("Hasse identity failed")
    return FiniteFieldCardinalityResult(
        curve=curve,
        cardinality=cardinality,
        trace=trace,
        frobenius_polynomial=(q, -trace, 1),
    )


def finite_field_extension_counts(
    curve: FiniteFieldShortWeierstrassCurve, max_degree: int
) -> FiniteFieldExtensionCountsResult:
    """Count ``E(F_(q^n))`` by the exact Frobenius recurrence.

    The base trace comes from the exact quadratic-character sum over the
    admitted base field; no point set is constructed for a count-only request.
    """
    if (
        type(max_degree) is not int
        or not 1 <= max_degree <= MAX_FROBENIUS_EXTENSION_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="elliptic_curve.finite_field.extension_degree_bound",
            message=f"max_degree must be an integer in [1, {MAX_FROBENIUS_EXTENSION_DEGREE}]",
        )
    curve = _curve_admit(curve)
    q = _admit_extension_count_growth(curve, max_degree)
    base = _cardinality_from_character_sum(curve, q)
    # S_0 = 2, S_1 = t, S_n = t*S_(n-1) - q*S_(n-2).
    power_sums = [2]
    if max_degree:
        power_sums.append(base.trace)
    for _ in range(2, max_degree + 1):
        power_sums.append(base.trace * power_sums[-1] - q * power_sums[-2])
    counts = tuple(
        FiniteFieldExtensionCount(
            degree=n,
            frobenius_power_sum=power_sums[n],
            cardinality=q**n + 1 - power_sums[n],
        )
        for n in range(1, max_degree + 1)
    )
    return FiniteFieldExtensionCountsResult(
        curve=base.curve,
        base_cardinality=base.cardinality,
        base_trace=base.trace,
        counts=counts,
    )


def finite_field_isogeny_class(
    first: FiniteFieldShortWeierstrassCurve,
    second: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldIsogenyClassResult:
    """Decide isogeny over a common finite field by exact Frobenius data.

    For elliptic curves over one finite field, Tate's isogeny theorem makes
    equality of Frobenius characteristic polynomials equivalent to being
    isogenous over that field. Traces are computed from exact quadratic
    character sums; this operation does not construct an isogeny.
    """
    first = _curve_admit(first)
    second = _curve_admit(second)
    if first.field != second.field:
        raise OperationDomainValidationError(
            location=("second", "field"),
            code="elliptic_curve.finite_field.isogeny_field_mismatch",
            message="isogeny comparison requires the same exact finite-field presentation",
        )
    output_bound = (
        len(rfc8785.dumps(first.model_dump(mode="json")))
        + len(rfc8785.dumps(second.model_dump(mode="json")))
        + 512
    )
    if output_bound > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("first", "second"),
            code="elliptic_curve.finite_field.isogeny_output_bound",
            message="isogeny comparison result exceeds the canonical output-byte envelope",
        )
    # Admit both full-field character sums before doing either one. The curves
    # share q and degree, so this bounds the complete pair request.
    q = _admit_extension_count_growth(first, 1)
    second_q = _admit_extension_count_growth(second, 1)
    if second_q != q:
        raise RuntimeError("common field presentations have different orders")
    pair_work = 2 * q * first.field.degree**2 * (8 + 2 * q.bit_length())
    if pair_work > MAX_ISOGENY_PAIR_CHARACTER_SUM_WORK:
        raise OperationResourceAdmissionError(
            location=("first", "field"),
            code="elliptic_curve.finite_field.isogeny_work_bound",
            message="paired Frobenius character sums exceed the admitted exact-work envelope",
        )
    first_count = _cardinality_from_character_sum(first, q)
    second_count = _cardinality_from_character_sum(second, q)
    first_poly = (q, -first_count.trace, 1)
    second_poly = (q, -second_count.trace, 1)
    return FiniteFieldIsogenyClassResult(
        first=first_count.curve,
        second=second_count.curve,
        first_cardinality=first_count.cardinality,
        second_cardinality=second_count.cardinality,
        first_trace=first_count.trace,
        second_trace=second_count.trace,
        first_frobenius_polynomial=first_poly,
        second_frobenius_polynomial=second_poly,
        same_isogeny_class=first_poly == second_poly,
    )


def finite_field_isomorphism(
    source: FiniteFieldShortWeierstrassCurve,
    target: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldIsomorphismResult:
    """Decide model isomorphism by complete scaling search over the common field.

    For characteristic greater than three, every isomorphism between these
    short models has the form ``(x,y) -> (u^2*x,u^3*y)``. The full nonzero
    field search proves a negative result as well as finding a positive map.
    """
    source = _curve_admit(source)
    target = _curve_admit(target)
    if source.field != target.field:
        raise OperationDomainValidationError(
            location=("target", "field"),
            code="elliptic_curve.finite_field.isomorphism_field_mismatch",
            message="isomorphism decision requires the same exact field presentation",
        )
    field = source.field
    q = field.characteristic**field.degree
    if q > MAX_FINITE_FIELD_ISOMORPHISM_ORDER:
        raise OperationResourceAdmissionError(
            location=("source", "field"),
            code="elliptic_curve.finite_field.isomorphism_order_bound",
            message="complete model-isomorphism search requires field order at most 4096",
        )
    work = q * field.degree**2 * (4 + 4 * q.bit_length())
    if work > MAX_FINITE_FIELD_ISOMORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "field"),
            code="elliptic_curve.finite_field.isomorphism_work_bound",
            message="complete model-isomorphism search exceeds its exact-work envelope",
        )
    largest_element = _element(field, (field.characteristic - 1,) * field.degree)
    output_bound = (
        len(rfc8785.dumps(source.model_dump(mode="json")))
        + len(rfc8785.dumps(target.model_dump(mode="json")))
        + len(rfc8785.dumps(largest_element.model_dump(mode="json")))
        + 256
    )
    if output_bound > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("source", "target"),
            code="elliptic_curve.finite_field.isomorphism_output_bound",
            message="model-isomorphism result exceeds the canonical output-byte envelope",
        )
    zero = (0,) * field.degree
    source_a, source_b = (
        _coordinates(source.coefficient_a),
        _coordinates(source.coefficient_b),
    )
    target_a, target_b = (
        _coordinates(target.coefficient_a),
        _coordinates(target.coefficient_b),
    )
    for encoded in range(1, q):
        u = _decode_field_element(field, encoded)
        u2 = _multiply(field, u, u)
        u4 = _multiply(field, u2, u2)
        u6 = _multiply(field, u4, u2)
        if (
            _multiply(field, u4, source_a) == target_a
            and _multiply(field, u6, source_b) == target_b
            and u != zero
        ):
            return FiniteFieldIsomorphismResult(
                source=source,
                target=target,
                isomorphic=True,
                scaling=_element(field, u),
            )
    return FiniteFieldIsomorphismResult(
        source=source, target=target, isomorphic=False, scaling=None
    )


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


def _finite_field_quadratic_twist_components(
    curve: FiniteFieldShortWeierstrassCurve,
    *,
    include_relation: bool,
) -> tuple[
    FiniteFieldShortWeierstrassCurve,
    FiniteFieldShortWeierstrassCurve,
    FiniteFieldElement,
]:
    """Compute shared twist components after operation-specific admission."""

    if not isinstance(curve, FiniteFieldShortWeierstrassCurve):
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.curve_type",
            message="curve must be a finite-field short-Weierstrass value",
        )
    try:
        admitted = FiniteFieldShortWeierstrassCurve.model_validate(
            curve.model_dump(), strict=True
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.invalid_curve",
            message="curve has malformed field or coefficient data",
        ) from exc
    field, coefficient_a, coefficient_b = require_discriminant_admission(
        admitted.field, admitted.coefficient_a, admitted.coefficient_b
    )
    q = field.characteristic**field.degree
    if q > MAX_FINITE_FIELD_TWIST_ORDER:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.twist_order_bound",
            message=(
                "canonical quadratic twist search requires field order at most "
                f"{MAX_FINITE_FIELD_TWIST_ORDER}"
            ),
        )
    work_bound = q * field.degree**2 * (2 * q.bit_length() + 8)
    if work_bound > MAX_FINITE_FIELD_TWIST_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.twist_work_bound",
            message="canonical quadratic twist search exceeds its exact work envelope",
        )
    max_element = _element(field, (field.characteristic - 1,) * field.degree)
    max_curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=max_element, coefficient_b=max_element
    )
    if include_relation:
        maximum_result = FiniteFieldQuadraticTwistRelation.model_construct(
            source_curve=admitted,
            twisted_curve=max_curve,
            parameter=max_element,
        )
    else:
        maximum_result = max_curve
    output_bound = len(rfc8785.dumps(maximum_result.model_dump(mode="json")))
    if not include_relation:
        # Retain the original curve-only admission margin for transport
        # overhead; the relation operation admits its complete exact shape.
        output_bound += 64
    if output_bound > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("curve",),
            code="elliptic_curve.finite_field.twist_output_bound",
            message="quadratic twist value exceeds the canonical output-byte envelope",
        )

    modulus = field.characteristic
    a, b = _coordinates(coefficient_a), _coordinates(coefficient_b)
    discriminant_core = _add(
        modulus,
        _scale(modulus, 4, _power(field, a, 3)),
        _scale(modulus, 27, _multiply(field, b, b)),
    )
    if not any(discriminant_core):
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.finite_field.singular_curve",
            message="quadratic twisting requires a nonsingular elliptic curve",
        )

    nonsquare = None
    minus_one = (modulus - 1,) + (0,) * (field.degree - 1)
    for encoded in range(1, q):
        candidate = _decode_field_element(field, encoded)
        if _power(field, candidate, (q - 1) // 2) == minus_one:
            nonsquare = candidate
            break
    if nonsquare is None:
        raise RuntimeError("finite field has no quadratic nonsquare")
    square = _multiply(field, nonsquare, nonsquare)
    cube = _multiply(field, square, nonsquare)
    twisted_a = _multiply(field, square, a)
    twisted_b = _multiply(field, cube, b)
    twisted_curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=_element(field, twisted_a),
        coefficient_b=_element(field, twisted_b),
    )
    return admitted, twisted_curve, _element(field, nonsquare)


def finite_field_quadratic_twist_relation(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldQuadraticTwistRelation:
    """Return the canonical twist model together with its nonsquare parameter.

    The twisting parameter is the first nonsquare in the field's canonical
    base-p coordinate order. For that nonsquare ``d``, the twist is
    ``y^2 = x^3 + d^2 A x + d^3 B``.
    """
    admitted, twisted_curve, parameter = _finite_field_quadratic_twist_components(
        curve, include_relation=True
    )
    return FiniteFieldQuadraticTwistRelation(
        source_curve=admitted,
        twisted_curve=twisted_curve,
        parameter=parameter,
    )


def finite_field_quadratic_twist(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldShortWeierstrassCurve:
    """Return the canonical nontrivial quadratic twist model."""
    _, twisted_curve, _ = _finite_field_quadratic_twist_components(
        curve, include_relation=False
    )
    return twisted_curve


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
    return _point_admit_with_curve(curve, point)


def _point_admit_with_curve(
    curve: FiniteFieldShortWeierstrassCurve, point: FiniteFieldEllipticPoint
) -> FiniteFieldEllipticPoint:
    """Admit a point when its source curve has already been admitted."""
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
    return _negate_point_admitted(point)


def _negate_point_admitted(
    point: FiniteFieldEllipticPoint,
) -> FiniteFieldEllipticPoint:
    """Negate a point after its curve and coordinates have been admitted."""
    curve = point.curve
    if point.at_infinity:
        return FiniteFieldEllipticPoint.infinity(curve)
    if point.x is None or point.y is None:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_coordinates",
            message="a finite elliptic-curve point requires both coordinates",
        )
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
    curve = _curve_admit(curve)
    first = _point_admit_with_curve(curve, first)
    second = _point_admit_with_curve(curve, second)
    return _add_points_admitted(first, second)


def _add_points_admitted(
    first: FiniteFieldEllipticPoint,
    second: FiniteFieldEllipticPoint,
) -> FiniteFieldEllipticPoint:
    """Add canonical curve-bound points without replaying admission checks."""
    curve = first.curve
    if first.at_infinity:
        return second
    if second.at_infinity:
        return first
    if first.x is None or first.y is None or second.x is None or second.y is None:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_coordinates",
            message="finite elliptic-curve points require both coordinates",
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
    if scalar < 0:
        point = _negate_point_admitted(point)
        scalar = -scalar
    return FiniteFieldPointResult(point=_scalar_multiply_admitted(point, scalar))


def _prime_divisors(value: int) -> tuple[int, ...]:
    """Return the increasing distinct prime divisors of a positive integer."""
    remaining = value
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def finite_field_point_order(
    curve: FiniteFieldShortWeierstrassCurve,
    point: FiniteFieldEllipticPoint,
) -> FiniteFieldPointOrderResult:
    """Compute exact point order from the bounded exact group cardinality.

    Starting with ``N = #E(F_q)``, remove each prime factor while the reduced
    multiple still annihilates the point. The final ``mP = O`` together with
    ``(m/r)P != O`` for every prime ``r | m`` proves that ``m`` is its exact
    order.
    """
    curve = _curve_admit(curve)
    point = _point_admit_with_curve(curve, point)
    q = _admit_extension_count_growth(curve, 1)
    cardinality_bound = q + 1 + isqrt(4 * q)
    scalar_bit_bound = cardinality_bound.bit_length()
    scalar_work_bound = (
        (2 * scalar_bit_bound + 1) * (scalar_bit_bound + 1) * (scalar_bit_bound + 1)
    )
    if scalar_work_bound > MAX_POINT_ORDER_SCALAR_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.point_order_work_bound",
            message="point-order scalar checks exceed the admitted exact-work envelope",
        )
    max_curve_json_bytes = len(rfc8785.dumps(curve.model_dump(mode="json")))
    max_element = _element(
        curve.field, (curve.field.characteristic - 1,) * curve.field.degree
    )
    max_point = FiniteFieldEllipticPoint.affine(curve, max_element, max_element)
    max_point_json_bytes = len(rfc8785.dumps(max_point.model_dump(mode="json")))
    output_bound = (
        max_curve_json_bytes
        + (MAX_POINT_ORDER_WITNESSES + 2) * max_point_json_bytes
        + 512
    )
    if output_bound > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("point",),
            code="elliptic_curve.finite_field.point_order_output_bound",
            message="point-order witnesses exceed the canonical output-byte envelope",
        )

    cardinality = _cardinality_from_character_sum(curve, q)
    group_primes = _prime_divisors(cardinality.cardinality)
    order = cardinality.cardinality
    for prime in group_primes:
        while order % prime == 0:
            reduced = order // prime
            if _scalar_multiply_admitted(point, reduced).at_infinity:
                order = reduced
            else:
                break
    annihilating_multiple = _scalar_multiply_admitted(point, order)
    if not annihilating_multiple.at_infinity:
        raise RuntimeError("the exact curve cardinality failed to annihilate a point")
    witnesses = tuple(
        FiniteFieldPointOrderPrimeWitness(
            prime=prime,
            reduced_scalar=order // prime,
            reduced_multiple=_scalar_multiply_admitted(point, order // prime),
        )
        for prime in _prime_divisors(order)
    )
    return FiniteFieldPointOrderResult(
        curve=curve,
        point=point,
        group_cardinality=cardinality.cardinality,
        order=order,
        annihilating_multiple=annihilating_multiple,
        prime_divisor_witnesses=witnesses,
    )


def _scalar_multiply_admitted(
    point: FiniteFieldEllipticPoint, scalar: int
) -> FiniteFieldEllipticPoint:
    """Double-and-add on one already admitted point, in O(log scalar) group steps."""
    curve = point.curve
    result = FiniteFieldEllipticPoint.infinity(curve)
    addend = point
    n = scalar
    while n:
        if n & 1:
            result = _add_points_admitted(result, addend)
        n >>= 1
        if n:
            addend = _add_points_admitted(addend, addend)
    return result


def finite_field_points(curve: FiniteFieldShortWeierstrassCurve) -> FiniteFieldPointSet:
    curve = _curve_admit(curve)
    q = _admit_point_enumeration(curve)
    return _enumerate_admitted_points(curve, q)


def finite_field_curve_base_change(
    curve: FiniteFieldShortWeierstrassCurve,
    embedding: FieldEmbedding,
    point: FiniteFieldEllipticPoint | None = None,
) -> FiniteFieldCurveBaseChangeResult:
    """Transport a curve and optional point along an explicit field embedding."""
    curve = _curve_admit(curve)
    try:
        embedding = FieldEmbedding.model_validate(embedding.model_dump())
    except (AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("embedding",),
            code="elliptic_curve.finite_field.base_change_embedding_invalid",
            message="embedding must be a canonical finite-field embedding value",
        ) from exc
    require_field(embedding.source)
    require_field(embedding.target)
    if curve.field != embedding.source:
        raise OperationDomainValidationError(
            location=("curve", "embedding"),
            code="elliptic_curve.finite_field.base_change_source_mismatch",
            message="curve field must equal the embedding source presentation",
        )
    if point is not None:
        point = _point_admit_with_curve(curve, point)

    target_field = embedding.target
    maximum = _element(
        target_field, (target_field.characteristic - 1,) * target_field.degree
    )
    sample_curve = FiniteFieldShortWeierstrassCurve.model_construct(
        field=target_field, coefficient_a=maximum, coefficient_b=maximum
    )
    sample_point = (
        FiniteFieldEllipticPoint.model_construct(
            curve=sample_curve, at_infinity=False, x=maximum, y=maximum
        )
        if point is not None and not point.at_infinity
        else FiniteFieldEllipticPoint.infinity(sample_curve)
        if point is not None
        else None
    )
    sample = FiniteFieldCurveBaseChangeResult.model_construct(
        curve=sample_curve, point=sample_point
    )
    if (
        len(rfc8785.dumps(sample.model_dump(mode="json")))
        > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("embedding", "target"),
            code="elliptic_curve.finite_field.base_change_output_bound",
            message="transported curve and point exceed the canonical output-byte envelope",
        )

    target_curve = FiniteFieldShortWeierstrassCurve.model_construct(
        field=target_field,
        coefficient_a=embed_field_element(curve.coefficient_a, embedding),
        coefficient_b=embed_field_element(curve.coefficient_b, embedding),
        model=curve.model,
    )
    target_curve = _curve_admit(target_curve)
    target_point = None
    if point is not None:
        if point.at_infinity:
            target_point = FiniteFieldEllipticPoint.infinity(target_curve)
        else:
            assert point.x is not None and point.y is not None
            target_point = FiniteFieldEllipticPoint.affine(
                target_curve,
                embed_field_element(point.x, embedding),
                embed_field_element(point.y, embedding),
            )
            target_point = _point_admit_with_curve(target_curve, target_point)
    return FiniteFieldCurveBaseChangeResult(curve=target_curve, point=target_point)


def finite_field_cardinality(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldCardinalityResult:
    points = finite_field_points(curve)
    curve = points.curve
    q = curve.field.characteristic**curve.field.degree
    return _cardinality_from_points(curve, points, q)


def finite_field_frobenius(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldFrobeniusResult:
    """Compute the exact Frobenius polynomial and p-rank class over F_q.

    For elliptic curves over a finite field of characteristic p, the curve is
    supersingular exactly when p divides the Frobenius trace; otherwise it is
    ordinary. The trace is obtained by the admitted exact quadratic-character
    sum used by the count-only operation.
    """
    curve = _curve_admit(curve)
    q = int(curve.field.characteristic**curve.field.degree)
    character_sum_work = q * curve.field.degree**2 * (8 + 2 * q.bit_length())
    if character_sum_work > MAX_FROBENIUS_CHARACTER_SUM_WORK:
        raise OperationResourceAdmissionError(
            location=("curve", "field"),
            code="elliptic_curve.finite_field.frobenius_work_bound",
            message=(
                "Frobenius trace exceeds the admitted quadratic-character "
                "sum work envelope"
            ),
        )
    trace_bound = 2 * (isqrt(q) + 1)
    if (
        128 + 8 * (q.bit_length() + trace_bound.bit_length())
        > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("curve",),
            code="elliptic_curve.finite_field.frobenius_output_bound",
            message="Frobenius data exceed the canonical output-byte envelope",
        )
    count = _cardinality_from_character_sum(curve, q)
    trace = count.trace
    return FiniteFieldFrobeniusResult(
        curve=count.curve,
        cardinality=count.cardinality,
        trace=trace,
        determinant=q,
        characteristic_polynomial=IntegerPolynomial(
            coefficients=(q, -trace, 1),
        ),
        discriminant=trace * trace - 4 * q,
        classification=(
            "SUPERSINGULAR"
            if trace % count.curve.field.characteristic == 0
            else "ORDINARY"
        ),
    )


def finite_field_zeta_polynomial(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldZetaPolynomialResult:
    """Return ``1 - a*T + q*T^2`` from one admitted exact base-field count."""
    curve = _curve_admit(curve)
    q = _admit_extension_count_growth(curve, 1)
    count = _cardinality_from_character_sum(curve, q)
    return FiniteFieldZetaPolynomialResult(
        curve=count.curve,
        cardinality=count.cardinality,
        trace=count.trace,
        numerator=IntegerPolynomial(
            coefficients=(q, -count.trace, 1),
        ),
    )


def finite_field_zeta_function(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldZetaFunctionResult:
    """Return the exact rational zeta function from one admitted base count."""
    count = finite_field_zeta_polynomial(curve)
    q = int(count.curve.field.characteristic**count.curve.field.degree)
    return FiniteFieldZetaFunctionResult(
        curve=count.curve,
        cardinality=count.cardinality,
        trace=count.trace,
        zeta_function=_zeta_rational_function(q, count.trace),
    )


def finite_field_group_structure(
    curve: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldGroupStructureResult:
    """Return invariant factors and generators of ``E(F_q)``.

    The complete point set gives the group order once. Reducing that Lagrange
    bound against every enumerated point determines the exponent, and hence
    the two invariant factors of an elliptic-curve point group. A maximal-order
    point generates the large cyclic factor; the second generator is selected
    only when its prime-order subgroups have trivial intersection with that
    cyclic factor.
    """
    curve = _curve_admit(curve)
    q = _admit_group_structure_work(curve)
    point_set = _enumerate_admitted_points(curve, q)
    cardinality = len(point_set.points)
    factors = _finite_group_factorization(cardinality)
    point_orders = tuple(
        _point_order_from_admitted_cardinality(point, cardinality, factors)
        for point in point_set.points
    )
    exponent = max(point_orders)
    small_factor = cardinality // exponent
    if small_factor > exponent or exponent % small_factor:
        raise RuntimeError(
            "finite elliptic point group invariant factors are inconsistent"
        )
    generator_two = next(
        point
        for point, order in zip(point_set.points, point_orders, strict=True)
        if order == exponent
    )
    if small_factor == 1:
        return FiniteFieldGroupStructureResult(
            curve=curve,
            group=AbelianPresentation(invariant_factors=(exponent,)),
            generators=(generator_two,),
        )

    cyclic_subgroup: set[tuple[object, ...]] = set()
    multiple = FiniteFieldEllipticPoint.infinity(curve)
    for _ in range(exponent):
        cyclic_subgroup.add(_finite_point_key(multiple))
        multiple = _add_points_admitted(multiple, generator_two)
    independent_generator: FiniteFieldEllipticPoint | None = None
    small_prime_divisors = tuple(
        prime for prime, _ in _finite_group_factorization(small_factor)
    )
    for point, order in zip(point_set.points, point_orders, strict=True):
        if order != small_factor:
            continue
        if all(
            _finite_point_key(_scalar_multiply_admitted(point, small_factor // prime))
            not in cyclic_subgroup
            for prime in small_prime_divisors
        ):
            independent_generator = point
            break
    if independent_generator is None:
        raise RuntimeError("finite elliptic point group generators were not found")
    return FiniteFieldGroupStructureResult(
        curve=curve,
        group=AbelianPresentation(invariant_factors=(small_factor, exponent)),
        generators=(independent_generator, generator_two),
    )


__all__ = [
    "FiniteFieldCardinalityResult",
    "FiniteFieldCurveBaseChangeRequest",
    "FiniteFieldCurveBaseChangeResult",
    "FiniteFieldCurveRequest",
    "FiniteFieldDiscriminantRequest",
    "FiniteFieldDiscriminantResult",
    "FiniteFieldEllipticPoint",
    "FiniteFieldExtensionCount",
    "FiniteFieldExtensionCountsRequest",
    "FiniteFieldExtensionCountsResult",
    "FiniteFieldFrobeniusResult",
    "FiniteFieldGroupStructureResult",
    "FiniteFieldIsogenyClassRequest",
    "FiniteFieldIsogenyClassResult",
    "FiniteFieldIsomorphismRequest",
    "FiniteFieldIsomorphismResult",
    "FiniteFieldPointAdditionRequest",
    "FiniteFieldPointCheckResult",
    "FiniteFieldPointOrderPrimeWitness",
    "FiniteFieldPointOrderRequest",
    "FiniteFieldPointOrderResult",
    "FiniteFieldPointRequest",
    "FiniteFieldPointResult",
    "FiniteFieldPointSet",
    "FiniteFieldQuadraticTwistRelation",
    "FiniteFieldScalarRequest",
    "FiniteFieldShortWeierstrassCurve",
    "FiniteFieldZetaFunctionResult",
    "FiniteFieldZetaPolynomialResult",
    "finite_field_cardinality",
    "finite_field_curve_base_change",
    "finite_field_discriminant",
    "finite_field_extension_counts",
    "finite_field_frobenius",
    "finite_field_group_structure",
    "finite_field_isogeny_class",
    "finite_field_isomorphism",
    "finite_field_point_add",
    "finite_field_point_check",
    "finite_field_point_negate",
    "finite_field_point_order",
    "finite_field_point_scalar",
    "finite_field_points",
    "finite_field_quadratic_twist",
    "finite_field_quadratic_twist_relation",
    "finite_field_zeta_function",
    "finite_field_zeta_polynomial",
    "require_discriminant_admission",
]
