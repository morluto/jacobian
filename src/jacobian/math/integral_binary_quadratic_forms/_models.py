"""Typed wire contracts for integral binary quadratic form operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_COEFFICIENT = 10**6
MAX_REPRESENTATION_TARGET = 10**12
# The representation kernel examines one quadratic discriminant per admissible
# y-value. The odd bound makes the inclusive interval ``[-y_bound, y_bound]``
# fit exactly; each y-value has at most two integral x-roots.
MAX_REPRESENTATION_Y_CANDIDATES = 50_001
MAX_REPRESENTATION_OUTPUT_ROWS = 2 * MAX_REPRESENTATION_Y_CANDIDATES
# With ``|y| <= 25_000``, ``|b| <= 10^6``, ``a <= 10^6``, and
# ``target <= 10^12``, the quadratic formula gives
# ``|x| <= (25_000*10^6 + sqrt(4*10^6*10^12)) / 2``.
MAX_REPRESENTATION_COORDINATE = 13_500_000_000
# A reduced-form search examines ``sum_{a=1}^A (2a+1) = A(A+2)`` candidates,
# where ``A = floor(sqrt(|D| / 3)) + 1``.  The bound keeps both the complete
# canonical class tuple and its independent result replay in one request.
MAX_REDUCED_CLASS_SEARCH_STATES = 10_000
MAX_REDUCED_CLASS_OUTPUT_ROWS = MAX_REDUCED_CLASS_SEARCH_STATES
# Canonical JSON carries integers only up to the interoperable IEEE 754 double
# range; every accepted evaluation must stay inside it so ``math.run`` returns
# a typed result instead of failing transport canonicalization.
MAX_EVALUATED_VALUE = (1 << 53) - 1


def _require_positive_primitive_form(form: tuple[int, int, int]) -> None:
    a, b, c = form
    if any(abs(value) > MAX_COEFFICIENT for value in form):
        raise ValueError("form coefficients exceed the supported bound")
    if a <= 0 or b * b - 4 * a * c >= 0:
        raise ValueError("form must be positive definite")
    from math import gcd

    if gcd(gcd(abs(a), abs(b)), abs(c)) != 1:
        raise ValueError("form must be primitive")


class PrimitivePositiveDefiniteBinaryQuadraticForm(StrictModel):
    """A canonical primitive positive-definite integral binary quadratic form.

    The fixed coefficient convention is ``Q(x, y) = a*x^2 + b*x*y + c*y^2``.
    """

    a: int = Field(gt=0, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(gt=0, le=MAX_COEFFICIENT)

    @model_validator(mode="after")
    def require_primitive_positive_definite(self) -> Self:
        _require_positive_primitive_form((self.a, self.b, self.c))
        return self

    @property
    def discriminant(self) -> int:
        """Return the derived discriminant ``b^2 - 4*a*c``."""
        return self.b * self.b - 4 * self.a * self.c


class BinaryQuadraticFormCheckRequest(StrictModel):
    """Request to check integer coefficients as a primitive positive-definite form."""

    a: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)


class BinaryQuadraticFormEvaluateRequest(StrictModel):
    """Request to evaluate a canonical form at an integer pair ``(x, y)``.

    Admitted exactly when the exact value ``Q(x,y) = a*x^2 + b*x*y + c*y^2``
    lies in ``[0, MAX_EVALUATED_VALUE]`` with
    ``MAX_EVALUATED_VALUE = 9007199254740991``, the interoperable canonical
    integer range.
    """

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    x: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    y: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)

    @model_validator(mode="after")
    def require_transportable_value(self) -> Self:
        _require_evaluated_value_bound(self.form, self.x, self.y)
        return self


class BinaryQuadraticFormReduceRequest(StrictModel):
    """Request Gauss reduction of a primitive positive-definite form."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm


class BinaryQuadraticFormProperEquivRequest(StrictModel):
    """Request to decide proper (SL_2(Z)) equivalence of two forms."""

    first: PrimitivePositiveDefiniteBinaryQuadraticForm
    second: PrimitivePositiveDefiniteBinaryQuadraticForm


class BinaryQuadraticFormReducedClassesRequest(StrictModel):
    """Request all reduced primitive positive-definite classes of a discriminant."""

    discriminant: int = Field(
        le=-3,
        description=(
            "Negative quadratic discriminant D <= -3 with D mod 4 in {0, 1}. "
            "The complete reduced-form scan uses A=floor_sqrt((-D)//3)+1 "
            "leading coefficients, so this request is admitted exactly when "
            f"A*(A+2) is at most {MAX_REDUCED_CLASS_SEARCH_STATES}."
        ),
    )

    @model_validator(mode="after")
    def require_complete_search_budget(self) -> Self:
        _require_reduced_class_search_budget(self.discriminant)
        return self


def _reduced_class_search_state_count(discriminant: int) -> int:
    """Return the exact number of ``(a,b)`` candidates in the reduced scan."""
    from math import isqrt

    a_bound = isqrt(abs(discriminant) // 3) + 1
    return a_bound * (a_bound + 2)


def _require_reduced_class_search_budget(discriminant: int) -> int:
    """Validate a negative quadratic discriminant and its complete scan."""
    if discriminant > -3:
        raise ValueError("discriminant must be at most -3")
    if discriminant % 4 not in (0, 1):
        raise ValueError("discriminant must be congruent to 0 or 1 modulo 4")
    state_count = _reduced_class_search_state_count(discriminant)
    if state_count > MAX_REDUCED_CLASS_SEARCH_STATES:
        raise ValueError(
            "reduced-class enumeration exceeds the supported candidate budget"
        )
    return state_count


def _representation_y_bound(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> int:
    """Return the exact complete y-coordinate bound for ``Q(x, y) = target``."""
    from math import isqrt

    # Completing the square gives
    #   Q(x,y) = a*(x + b*y/(2*a))^2 + (-D)*y^2/(4*a).
    # Hence every representation satisfies ``(-D)*y^2 <= 4*a*target``.
    return isqrt((4 * form.a * target) // (-form.discriminant))


def _require_representation_budget(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> int:
    """Return the complete y-bound after checking the exact scan envelope."""
    y_bound = _representation_y_bound(form, target)
    if 2 * y_bound + 1 > MAX_REPRESENTATION_Y_CANDIDATES:
        raise ValueError(
            "representation search exceeds the supported y-coordinate candidate budget"
        )
    return y_bound


def _require_evaluated_value_bound(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, x: int, y: int
) -> int:
    """Return the exact evaluated value after checking the transport range.

    The form is positive definite, so the value is nonnegative and checking
    the upper endpoint admits exactly the transportable outputs.
    """
    value = form.a * x * x + form.b * x * y + form.c * y * y
    if value > MAX_EVALUATED_VALUE:
        raise ValueError("evaluated value exceeds the interoperable integer range")
    return value


class BinaryQuadraticFormRepresentationsRequest(StrictModel):
    """Request the complete ordered signed representation set of one target."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    target: int = Field(
        ge=0,
        le=MAX_REPRESENTATION_TARGET,
        description=(
            "Nonnegative target n. With D=b^2-4ac for form, this request is "
            "admitted exactly when 2*floor_sqrt(4*a*n/(-D))+1 is at most "
            f"{MAX_REPRESENTATION_Y_CANDIDATES}; that is the complete "
            "y-coordinate scan size."
        ),
    )

    @model_validator(mode="after")
    def require_complete_search_budget(self) -> Self:
        _require_representation_budget(self.form, self.target)
        return self


class BinaryQuadraticFormRepresentation(StrictModel):
    """One ordered signed integer representation with its derived primitiveness."""

    x: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    y: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    primitive: bool

    @model_validator(mode="after")
    def bind_primitiveness(self) -> Self:
        from math import gcd

        if self.primitive != (gcd(self.x, self.y) == 1):
            raise ValueError("primitive must equal gcd(x, y) == 1")
        return self


class BinaryQuadraticFormRepresentationsResult(StrictModel):
    """The complete canonical representation set for one positive-definite form."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    target: int = Field(ge=0, le=MAX_REPRESENTATION_TARGET)
    representations: tuple[BinaryQuadraticFormRepresentation, ...] = Field(
        max_length=MAX_REPRESENTATION_OUTPUT_ROWS
    )
    count: int = Field(ge=0)
    primitive_count: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_complete_representation_set(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _enumerate_representations,
        )

        _require_representation_budget(self.form, self.target)
        expected = _enumerate_representations(self.form, self.target)
        if self.representations != expected:
            raise ValueError(
                "representations must be the complete lexicographically ordered set"
            )
        if self.count != len(self.representations):
            raise ValueError("count must equal the number of representations")
        if self.primitive_count != sum(row.primitive for row in self.representations):
            raise ValueError("primitive_count must equal the derived primitive total")
        return self


class BinaryQuadraticFormCheckResult(StrictModel):
    """Result of checking a binary quadratic form."""

    a: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    status: Literal["PRIMITIVE_POSITIVE_DEFINITE", "NOT_IN_INITIAL_DOMAIN"]
    obstruction: str | None = None
    form: PrimitivePositiveDefiniteBinaryQuadraticForm | None = None

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        obstruction: str | None = None
        discriminant = self.b**2 - 4 * self.a * self.c
        if self.a <= 0:
            obstruction = "a<=0: form is not positive definite (a must be positive)"
        elif discriminant >= 0:
            obstruction = f"discriminant D={discriminant}>=0: only negative discriminants are supported"
        else:
            from math import gcd

            divisor = gcd(gcd(abs(self.a), abs(self.b)), abs(self.c))
            if divisor > 1:
                obstruction = f"gcd(a,b,c)={divisor}>1: form is not primitive"
        if self.status == "PRIMITIVE_POSITIVE_DEFINITE":
            if obstruction is not None:
                raise ValueError("accepted form must satisfy the initial domain")
            if self.form is None:
                raise ValueError("accepted form must carry the canonical form value")
            if (self.form.a, self.form.b, self.form.c) != (self.a, self.b, self.c):
                raise ValueError("canonical form must match the checked coefficients")
            if self.obstruction is not None:
                raise ValueError("accepted form cannot carry an obstruction")
        else:
            if obstruction is None:
                raise ValueError(
                    "valid form cannot be classified outside the initial domain"
                )
            if self.form is not None:
                raise ValueError("rejected form cannot carry a canonical form value")
            if self.obstruction != obstruction:
                raise ValueError("obstruction must match the checked coefficients")
        return self


class BinaryQuadraticFormEvaluateResult(StrictModel):
    """Result of evaluating a form at (x, y)."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    x: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    y: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    value: int
    primitive: bool

    @model_validator(mode="after")
    def bind_value(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _evaluate,
            _gcd,
        )

        _require_evaluated_value_bound(self.form, self.x, self.y)
        value = _evaluate(self.form.a, self.form.b, self.form.c, self.x, self.y)
        if self.value != value:
            raise ValueError("value must be a*x^2 + b*x*y + c*y^2")
        primitive = _gcd(self.x, self.y) == 1
        if self.primitive != primitive:
            raise ValueError("primitive must be gcd(x,y)==1")
        return self


class ReducedBinaryQuadraticFormResult(StrictModel):
    """Result of Gauss reduction."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    reduced_form: PrimitivePositiveDefiniteBinaryQuadraticForm
    matrix: tuple[tuple[int, int], tuple[int, int]]

    @model_validator(mode="after")
    def bind_reduction(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _check_reduced,
        )

        if not _check_reduced(
            self.reduced_form.a, self.reduced_form.b, self.reduced_form.c
        ):
            raise ValueError("reduced form must satisfy |b|<=a<=c with tie-breaking")
        p, q = self.matrix[0]
        r, s = self.matrix[1]
        if p * s - q * r != 1:
            raise ValueError("transformation matrix must have determinant 1")
        ra, rb, rc = _transform(self.form.a, self.form.b, self.form.c, p, q, r, s)
        if (ra, rb, rc) != (
            self.reduced_form.a,
            self.reduced_form.b,
            self.reduced_form.c,
        ):
            raise ValueError("transformation must map original to reduced form")
        return self


def _transform(
    a: int, b: int, c: int, p: int, q: int, r: int, s: int
) -> tuple[int, int, int]:
    """Apply SL_2(Z) transformation U=[[p,q],[r,s]] to form [a,b,c]."""
    na = a * p * p + b * p * r + c * r * r
    nb = 2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s
    nc = a * q * q + b * q * s + c * s * s
    return na, nb, nc


class ProperEquivalenceResult(StrictModel):
    """Result of proper equivalence decision."""

    first: PrimitivePositiveDefiniteBinaryQuadraticForm
    second: PrimitivePositiveDefiniteBinaryQuadraticForm
    status: Literal["PROPERLY_EQUIVALENT", "NOT_PROPERLY_EQUIVALENT"]
    matrix: tuple[tuple[int, int], tuple[int, int]] | None = None

    @model_validator(mode="after")
    def bind_equivalence(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import _reduce

        first_reduced = _reduce(self.first.a, self.first.b, self.first.c)[:3]
        second_reduced = _reduce(self.second.a, self.second.b, self.second.c)[:3]
        equivalent = (
            self.first.discriminant == self.second.discriminant
            and first_reduced == second_reduced
        )
        if (self.status == "PROPERLY_EQUIVALENT") != equivalent:
            raise ValueError("status must match canonical proper-equivalence decision")
        if self.status == "PROPERLY_EQUIVALENT" and self.matrix is None:
            raise ValueError("proper equivalence requires a witness matrix")
        if self.status == "PROPERLY_EQUIVALENT" and self.matrix is not None:
            p, q = self.matrix[0]
            r, s = self.matrix[1]
            if p * s - q * r != 1:
                raise ValueError("witness matrix must have determinant 1")
            ta, tb, tc = _transform(
                self.first.a, self.first.b, self.first.c, p, q, r, s
            )
            if (ta, tb, tc) != (self.second.a, self.second.b, self.second.c):
                raise ValueError("witness must map first to second form")
        if self.status == "NOT_PROPERLY_EQUIVALENT" and self.matrix is not None:
            raise ValueError("nonequivalence cannot carry a witness matrix")
        return self


class ReducedClassesResult(StrictModel):
    """Result of enumerating reduced classes of a discriminant."""

    discriminant: int
    classes: tuple[PrimitivePositiveDefiniteBinaryQuadraticForm, ...] = Field(
        max_length=MAX_REDUCED_CLASS_OUTPUT_ROWS
    )
    class_number: int

    @model_validator(mode="after")
    def bind_classes(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _check_reduced,
        )

        if self.class_number != len(self.classes):
            raise ValueError("class_number must equal the number of classes")
        _require_reduced_class_search_budget(self.discriminant)
        for form in self.classes:
            if form.discriminant != self.discriminant:
                raise ValueError("every class must have the requested discriminant")
            if not _check_reduced(form.a, form.b, form.c):
                raise ValueError("every class must be reduced")
        seen = set(self.classes)
        if len(seen) != len(self.classes):
            raise ValueError("classes must be distinct")
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _enumerate_reduced_classes,
        )

        if self.classes != _enumerate_reduced_classes(self.discriminant):
            raise ValueError("classes must be the complete reduced class set")
        return self
