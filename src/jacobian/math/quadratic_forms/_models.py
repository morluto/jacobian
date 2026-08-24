"""Typed wire contracts for quadratic form operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

MAX_DIM = 50
MAX_ENTRY_DIGITS = 100  # limit entry magnitude to ~10^100 for bounded eigenvalue work
MAX_VECTOR_DIGITS = 100
MAX_EVALUATION_DIGITS = MAX_ENTRY_DIGITS + 2 * MAX_VECTOR_DIGITS + 3
MAX_DISCRIMINANT_DIGITS = MAX_DIM * (MAX_ENTRY_DIGITS + 1)
# Representation-number enumeration remains bounded separately; see
# MAX_REPRESENTATION_DIM below. Larger dimensions are served by O(n^3)
# linear-algebra ops (discriminant, signature); enumeration ops will be
# split to dedicated types in a follow-up.
MAX_REPRESENTATION_DIM = 10


def _require_integer_digits(value: CanonicalInteger, maximum: int, label: str) -> None:
    if len(value.lstrip("-")) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} digits")


class SymmetricMatrix(StrictModel):
    """A symmetric integer matrix representing a quadratic form."""

    matrix: tuple[tuple[CanonicalInteger, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_symmetric_square(self) -> Self:
        n = len(self.matrix)
        if n > MAX_DIM:
            raise ValueError(f"dimension must not exceed {MAX_DIM}")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("matrix must be square")
            for entry in row:
                _require_integer_digits(entry, MAX_ENTRY_DIGITS, "matrix entries")
        for i in range(n):
            for j in range(n):
                if self.matrix[i][j] != self.matrix[j][i]:
                    raise ValueError("matrix must be symmetric")
        return self


class EvaluationRequest(StrictModel):
    """Evaluate q(x) = x^T A x for an integer vector x."""

    form: SymmetricMatrix
    vector: tuple[CanonicalInteger, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        if len(self.vector) != len(self.form.matrix):
            raise ValueError("vector dimension must match form dimension")
        for entry in self.vector:
            _require_integer_digits(entry, MAX_VECTOR_DIGITS, "vector entries")
        return self


class DiscriminantRequest(StrictModel):
    """Compute the discriminant of a quadratic form."""

    form: SymmetricMatrix


class SignatureRequest(StrictModel):
    """Compute the signature (n_pos, n_neg, n_zero) of a quadratic form."""

    form: SymmetricMatrix


class EvaluationResult(StrictModel):
    """The value q(x) = x^T A x."""

    value: CanonicalInteger
    dimension: int

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        _require_integer_digits(self.value, MAX_EVALUATION_DIGITS, "evaluation value")
        return self


class DiscriminantResult(StrictModel):
    """The discriminant det(A) of a quadratic form."""

    discriminant: CanonicalInteger
    dimension: int

    @model_validator(mode="after")
    def require_bounded_discriminant(self) -> Self:
        _require_integer_digits(
            self.discriminant, MAX_DISCRIMINANT_DIGITS, "discriminant"
        )
        return self


class SignatureResult(StrictModel):
    """The inertia (positive, negative, zero eigenvalue counts) of a form."""

    n_positive: int
    n_negative: int
    n_zero: int
    is_positive_definite: bool
    is_negative_definite: bool
    is_indefinite: bool

    @model_validator(mode="after")
    def require_consistent_inertia(self) -> Self:
        counts = (self.n_positive, self.n_negative, self.n_zero)
        if any(count < 0 for count in counts):
            raise ValueError("inertia counts must be nonnegative")
        if self.is_positive_definite != (
            self.n_positive > 0 and self.n_negative == 0 and self.n_zero == 0
        ):
            raise ValueError("positive-definite flag must agree with inertia")
        if self.is_negative_definite != (
            self.n_negative > 0 and self.n_positive == 0 and self.n_zero == 0
        ):
            raise ValueError("negative-definite flag must agree with inertia")
        if self.is_indefinite != (self.n_positive > 0 and self.n_negative > 0):
            raise ValueError("indefinite flag must agree with inertia")
        return self


class RepresentationNumbersRequest(StrictModel):
    """Request the representation numbers r(n) for n = 0, 1, ..., bound."""

    form: SymmetricMatrix
    bound: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def require_valid_bound(self) -> Self:
        if self.bound > 1000:
            raise ValueError("bound must not exceed 1000")
        if len(self.form.matrix) > MAX_REPRESENTATION_DIM:
            raise ValueError(
                f"representation_numbers dimension must not exceed {MAX_REPRESENTATION_DIM} "
                "(enumeration bound; larger dimensions need a dedicated operation)"
            )
        return self


class RepresentationNumbersResult(StrictModel):
    """The representation numbers r(0), r(1), ..., r(bound)."""

    form: SymmetricMatrix
    bound: int = Field(ge=0, le=1000)
    counts: tuple[int, ...]

    @model_validator(mode="after")
    def bind_counts(self) -> Self:
        from jacobian.math.quadratic_forms._operations import _representation_numbers

        counts = _representation_numbers(self.form.matrix, self.bound)
        if self.counts != counts:
            raise ValueError("counts must be the exact representation numbers")
        return self


class ThetaSeriesPrefixRequest(StrictModel):
    """Request the theta series prefix q^0 through q^bound."""

    form: SymmetricMatrix
    bound: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def require_valid_bound(self) -> Self:
        if self.bound > 1000:
            raise ValueError("bound must not exceed 1000")
        if len(self.form.matrix) > MAX_REPRESENTATION_DIM:
            raise ValueError(
                f"theta_series dimension must not exceed {MAX_REPRESENTATION_DIM} "
                "(enumeration bound; larger dimensions need a dedicated operation)"
            )
        return self


class ThetaSeriesPrefixResult(StrictModel):
    """The theta series prefix coefficients r(0), ..., r(bound)."""

    form: SymmetricMatrix
    bound: int = Field(ge=0, le=1000)
    coefficients: tuple[int, ...]

    @model_validator(mode="after")
    def bind_coefficients(self) -> Self:
        from jacobian.math.quadratic_forms._operations import _representation_numbers

        coeffs = _representation_numbers(self.form.matrix, self.bound)
        if self.coefficients != coeffs:
            raise ValueError("coefficients must be the exact theta series prefix")
        return self


class ScalingRequest(StrictModel):
    """Request scaling of a quadratic form by an integer factor."""

    form: SymmetricMatrix
    factor: int = Field(ge=-1000, le=1000)


class ScalingResult(StrictModel):
    """The scaled quadratic form factor * A."""

    form: SymmetricMatrix
    factor: int = Field(ge=-1000, le=1000)
    scaled_form: SymmetricMatrix

    @model_validator(mode="after")
    def bind_scaled(self) -> Self:
        from jacobian.math.quadratic_forms._operations import _scale_form

        expected = _scale_form(self.form.matrix, self.factor)
        if self.scaled_form.matrix != expected:
            raise ValueError("scaled_form must be factor * A")
        return self


class DirectSumRequest(StrictModel):
    """Request the direct sum of two quadratic forms."""

    form1: SymmetricMatrix
    form2: SymmetricMatrix

    @model_validator(mode="after")
    def require_bounded_sum(self) -> Self:
        if len(self.form1.matrix) + len(self.form2.matrix) > MAX_DIM:
            raise ValueError(f"combined dimension must not exceed {MAX_DIM}")
        return self


class DirectSumResult(StrictModel):
    """The block diagonal direct sum A ⊕ B."""

    form1: SymmetricMatrix
    form2: SymmetricMatrix
    direct_sum: SymmetricMatrix

    @model_validator(mode="after")
    def bind_direct_sum(self) -> Self:
        from jacobian.math.quadratic_forms._operations import _direct_sum

        expected = _direct_sum(self.form1.matrix, self.form2.matrix)
        if self.direct_sum.matrix != expected:
            raise ValueError("direct_sum must be the block diagonal A ⊕ B")
        return self
