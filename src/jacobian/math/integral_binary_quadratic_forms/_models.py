"""Typed wire contracts for integral binary quadratic form operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _require_positive_primitive_form(form: tuple[int, int, int]) -> None:
    a, b, c = form
    if any(abs(value) > MAX_COEFFICIENT for value in form):
        raise _validation_error(
            "integral_binary_quadratic_form.coefficient_bound",
            "form coefficients exceed the supported bound",
        )
    if a <= 0 or b * b - 4 * a * c >= 0:
        raise _validation_error(
            "integral_binary_quadratic_form.not_positive_definite",
            "form must be positive definite",
        )
    from math import gcd

    if gcd(gcd(abs(a), abs(b)), abs(c)) != 1:
        raise _validation_error(
            "integral_binary_quadratic_form.not_primitive", "form must be primitive"
        )


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
        _require_representation_coordinate(self.x)
        _require_representation_coordinate(self.y)
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
        raise _validation_error(
            "integral_binary_quadratic_form.discriminant_too_large",
            "discriminant must be at most -3",
        )
    if discriminant % 4 not in (0, 1):
        raise _validation_error(
            "integral_binary_quadratic_form.invalid_discriminant_congruence",
            "discriminant must be congruent to 0 or 1 modulo 4",
        )
    state_count = _reduced_class_search_state_count(discriminant)
    if state_count > MAX_REDUCED_CLASS_SEARCH_STATES:
        raise _validation_error(
            "integral_binary_quadratic_form.reduced_class_candidate_budget",
            "reduced-class enumeration exceeds the supported candidate budget",
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


def _has_sum_of_two_squares_mod_four_obstruction(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> bool:
    """Whether a request has the immediate ``x² + y² ≡ 3 (mod 4)`` obstruction.

    Squares modulo four are zero or one, so the sum-of-two-squares form cannot
    represent a target congruent to three.  This is a complete empty-result
    certificate, not a heuristic used to narrow the general y-coordinate scan.
    """

    return (form.a, form.b, form.c) == (1, 0, 1) and target % 4 == 3


def _require_representation_budget(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> int | None:
    """Return the complete scan bound, or ``None`` for a proved empty result."""
    if _has_sum_of_two_squares_mod_four_obstruction(form, target):
        return None
    y_bound = _representation_y_bound(form, target)
    if 2 * y_bound + 1 > MAX_REPRESENTATION_Y_CANDIDATES:
        raise _validation_error(
            "integral_binary_quadratic_form.representation_candidate_budget",
            "representation search exceeds the supported y-coordinate candidate budget",
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
        raise _validation_error(
            "integral_binary_quadratic_form.evaluated_value_range",
            "evaluated value exceeds the interoperable integer range",
        )
    return value


def _require_representation_coordinate(value: int) -> int:
    """Validate one coordinate accepted by form evaluation and enumeration."""
    if not -MAX_REPRESENTATION_COORDINATE <= value <= MAX_REPRESENTATION_COORDINATE:
        raise _validation_error(
            "integral_binary_quadratic_form.coordinate_bound",
            "coordinate exceeds the supported bound",
        )
    return value


class BinaryQuadraticFormRepresentationsRequest(StrictModel):
    """Request the complete ordered signed representation set of one target."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    target: int = Field(
        ge=0,
        le=MAX_REPRESENTATION_TARGET,
        description=(
            "Nonnegative target n. With D=b^2-4ac for form, this request is "
            "admitted exactly when either (a,b,c)=(1,0,1) and n mod 4 is 3, "
            "where squares are 0 or 1 modulo 4 so x^2+y^2=n is proved empty "
            "without any scan, or 2*floor_sqrt(4*a*n/(-D))+1 is at most "
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

    @classmethod
    def _from_kernel(cls, *, x: int, y: int, primitive: bool) -> Self:
        """Construct one representation after the enumeration kernel established it."""

        return cls.model_construct(x=x, y=y, primitive=primitive)


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
    def require_result_shape(self) -> Self:
        if self.count != len(self.representations):
            raise _validation_error(
                "integral_binary_quadratic_form.count_mismatch",
                "count must equal the number of representations",
            )
        if self.primitive_count != sum(row.primitive for row in self.representations):
            raise _validation_error(
                "integral_binary_quadratic_form.primitive_count_mismatch",
                "primitive_count must equal the derived primitive total",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: PrimitivePositiveDefiniteBinaryQuadraticForm,
        target: int,
        representations: tuple[BinaryQuadraticFormRepresentation, ...],
    ) -> Self:
        """Construct a trusted result from the owner-local enumeration kernel."""
        return cls.model_construct(
            form=form,
            target=target,
            representations=representations,
            count=len(representations),
            primitive_count=sum(row.primitive for row in representations),
        )


class BinaryQuadraticFormCheckResult(StrictModel):
    """Result of checking a binary quadratic form."""

    a: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    status: Literal["PRIMITIVE_POSITIVE_DEFINITE", "NOT_IN_INITIAL_DOMAIN"]
    obstruction: str | None = None
    form: PrimitivePositiveDefiniteBinaryQuadraticForm | None = None

    @classmethod
    def _from_kernel(
        cls,
        *,
        a: int,
        b: int,
        c: int,
        status: Literal["PRIMITIVE_POSITIVE_DEFINITE", "NOT_IN_INITIAL_DOMAIN"],
        obstruction: str | None = None,
        form: PrimitivePositiveDefiniteBinaryQuadraticForm | None = None,
    ) -> Self:
        """Construct a trusted result from the owner-local domain check."""
        return cls(a=a, b=b, c=c, status=status, obstruction=obstruction, form=form)


class BinaryQuadraticFormEvaluateResult(StrictModel):
    """Result of evaluating a form at (x, y)."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    x: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    y: int = Field(ge=-MAX_REPRESENTATION_COORDINATE, le=MAX_REPRESENTATION_COORDINATE)
    value: int = Field(ge=0, le=MAX_EVALUATED_VALUE)
    primitive: bool

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: PrimitivePositiveDefiniteBinaryQuadraticForm,
        x: int,
        y: int,
        value: int,
        primitive: bool,
    ) -> Self:
        """Construct a trusted result from the owner-local evaluation kernel."""
        return cls(form=form, x=x, y=y, value=value, primitive=primitive)


class ReducedBinaryQuadraticFormResult(StrictModel):
    """Result of Gauss reduction."""

    form: PrimitivePositiveDefiniteBinaryQuadraticForm
    reduced_form: PrimitivePositiveDefiniteBinaryQuadraticForm
    matrix: tuple[tuple[int, int], tuple[int, int]]

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: PrimitivePositiveDefiniteBinaryQuadraticForm,
        reduced_form: PrimitivePositiveDefiniteBinaryQuadraticForm,
        matrix: tuple[tuple[int, int], tuple[int, int]],
    ) -> Self:
        """Construct a trusted result from the owner-local reduction kernel."""
        return cls(form=form, reduced_form=reduced_form, matrix=matrix)


class ProperEquivalenceResult(StrictModel):
    """Result of proper equivalence decision."""

    first: PrimitivePositiveDefiniteBinaryQuadraticForm
    second: PrimitivePositiveDefiniteBinaryQuadraticForm
    status: Literal["PROPERLY_EQUIVALENT", "NOT_PROPERLY_EQUIVALENT"]
    matrix: tuple[tuple[int, int], tuple[int, int]] | None = None

    @classmethod
    def _from_kernel(
        cls,
        *,
        first: PrimitivePositiveDefiniteBinaryQuadraticForm,
        second: PrimitivePositiveDefiniteBinaryQuadraticForm,
        status: Literal["PROPERLY_EQUIVALENT", "NOT_PROPERLY_EQUIVALENT"],
        matrix: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ) -> Self:
        """Construct a trusted result from the owner-local equivalence kernel."""
        return cls(first=first, second=second, status=status, matrix=matrix)


class ReducedClassesResult(StrictModel):
    """Result of enumerating reduced classes of a discriminant."""

    discriminant: int
    classes: tuple[PrimitivePositiveDefiniteBinaryQuadraticForm, ...] = Field(
        max_length=MAX_REDUCED_CLASS_OUTPUT_ROWS
    )
    class_number: int

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        if self.class_number != len(self.classes):
            raise _validation_error(
                "integral_binary_quadratic_form.class_number_mismatch",
                "class_number must equal the number of classes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        discriminant: int,
        classes: tuple[PrimitivePositiveDefiniteBinaryQuadraticForm, ...],
    ) -> Self:
        """Construct a trusted result from the owner-local class enumerator."""
        return cls.model_construct(
            discriminant=discriminant,
            classes=classes,
            class_number=len(classes),
        )
