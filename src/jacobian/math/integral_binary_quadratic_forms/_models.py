"""Typed wire contracts for integral binary quadratic form operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_COEFFICIENT = 10**6


class BinaryQuadraticFormCheckRequest(StrictModel):
    """Request to check integer coefficients as a primitive positive-definite form."""

    a: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)


class BinaryQuadraticFormEvaluateRequest(StrictModel):
    """Request to evaluate a checked form at an integer pair (x, y)."""

    a: int
    b: int
    c: int
    x: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    y: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)


class BinaryQuadraticFormReduceRequest(StrictModel):
    """Request Gauss reduction of a primitive positive-definite form."""

    a: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    b: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)
    c: int = Field(ge=-MAX_COEFFICIENT, le=MAX_COEFFICIENT)

    @model_validator(mode="after")
    def require_valid_form(self) -> Self:
        if self.a <= 0:
            raise ValueError("a must be positive for positive-definite")
        disc = self.b * self.b - 4 * self.a * self.c
        if disc >= 0:
            raise ValueError("discriminant must be negative")
        if disc % 4 not in (0, 1):
            raise ValueError("discriminant must be 0 or 1 mod 4")
        from math import gcd

        if gcd(gcd(self.a, self.b), self.c) != 1:
            raise ValueError("form must be primitive")
        return self


class BinaryQuadraticFormProperEquivRequest(StrictModel):
    """Request to decide proper (SL_2(Z)) equivalence of two forms."""

    form1: tuple[int, int, int]
    form2: tuple[int, int, int]


class BinaryQuadraticFormReducedClassesRequest(StrictModel):
    """Request all reduced primitive positive-definite classes of a discriminant."""

    discriminant: int = Field(le=-3)


class BinaryQuadraticFormCheckResult(StrictModel):
    """Result of checking a binary quadratic form."""

    status: Literal["PRIMITIVE_POSITIVE_DEFINITE", "NOT_IN_INITIAL_DOMAIN"]
    obstruction: str | None = None
    a: int | None = None
    b: int | None = None
    c: int | None = None
    discriminant: int | None = None
    gram: tuple[tuple[int, ...], ...] | None = None

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        if self.status == "PRIMITIVE_POSITIVE_DEFINITE":
            if self.a is None or self.b is None or self.c is None:
                raise ValueError("accepted form must carry coefficients")
            if self.discriminant is None:
                raise ValueError("accepted form must carry discriminant")
            if self.discriminant != self.b**2 - 4 * self.a * self.c:
                raise ValueError("discriminant must be b^2 - 4ac")
            if self.gram is not None and self.gram != (
                (self.a, self.b),
                (self.b, self.c),
            ):
                raise ValueError("gram must be [[a,b],[b,c]]")
        return self


class BinaryQuadraticFormEvaluateResult(StrictModel):
    """Result of evaluating a form at (x, y)."""

    a: int
    b: int
    c: int
    x: int
    y: int
    value: int
    primitive: bool

    @model_validator(mode="after")
    def bind_value(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _evaluate,
            _gcd,
        )

        value = _evaluate(self.a, self.b, self.c, self.x, self.y)
        if self.value != value:
            raise ValueError("value must be a*x^2 + b*x*y + c*y^2")
        primitive = _gcd(self.x, self.y) == 1
        if self.primitive != primitive:
            raise ValueError("primitive must be gcd(x,y)==1")
        return self


class ReducedBinaryQuadraticFormResult(StrictModel):
    """Result of Gauss reduction."""

    a: int
    b: int
    c: int
    reduced_a: int
    reduced_b: int
    reduced_c: int
    matrix: tuple[tuple[int, int], tuple[int, int]]
    steps: tuple[tuple[int, int, int, int, int, int, int, int, int, int], ...]

    @model_validator(mode="after")
    def bind_reduction(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _check_reduced,
        )

        if not _check_reduced(self.reduced_a, self.reduced_b, self.reduced_c):
            raise ValueError("reduced form must satisfy |b|<=a<=c with tie-breaking")
        if (
            self.a == self.reduced_a
            and self.b == self.reduced_b
            and self.c == self.reduced_c
        ):
            return self
        p, q = self.matrix[0]
        r, s = self.matrix[1]
        if p * s - q * r != 1:
            raise ValueError("transformation matrix must have determinant 1")
        ra, rb, rc = _transform(self.a, self.b, self.c, p, q, r, s)
        if (ra, rb, rc) != (self.reduced_a, self.reduced_b, self.reduced_c):
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

    form1: tuple[int, int, int]
    form2: tuple[int, int, int]
    status: Literal["PROPERLY_EQUIVALENT", "NOT_PROPERLY_EQUIVALENT"]
    matrix: tuple[tuple[int, int], tuple[int, int]] | None = None

    @model_validator(mode="after")
    def bind_equivalence(self) -> Self:
        if self.status == "PROPERLY_EQUIVALENT" and self.matrix is not None:
            p, q = self.matrix[0]
            r, s = self.matrix[1]
            if p * s - q * r != 1:
                raise ValueError("witness matrix must have determinant 1")
            a1, b1, c1 = self.form1
            ta, tb, tc = _transform(a1, b1, c1, p, q, r, s)
            if (ta, tb, tc) != self.form2:
                raise ValueError("witness must map form1 to form2")
        return self


class ReducedClassesResult(StrictModel):
    """Result of enumerating reduced classes of a discriminant."""

    discriminant: int
    classes: tuple[tuple[int, int, int], ...]
    class_number: int

    @model_validator(mode="after")
    def bind_classes(self) -> Self:
        from jacobian.math.integral_binary_quadratic_forms._operations import (
            _check_reduced,
        )

        if self.class_number != len(self.classes):
            raise ValueError("class_number must equal the number of classes")
        for a, b, c in self.classes:
            if b * b - 4 * a * c != self.discriminant:
                raise ValueError("every class must have the requested discriminant")
            if not _check_reduced(a, b, c):
                raise ValueError("every class must be reduced")
        seen = set(self.classes)
        if len(seen) != len(self.classes):
            raise ValueError("classes must be distinct")
        return self
