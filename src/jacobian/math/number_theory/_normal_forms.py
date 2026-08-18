"""Pydantic wire contracts for integer multiplicative normal-form operations.

These models cover the maximal perfect-power profile, k-free decomposition,
squarefree decomposition, squarefree part, and normalized positive quadratic
radical operations proposed in issue #1893.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

# ---------------------------------------------------------------------------
# Shared bounds
# ---------------------------------------------------------------------------

_MAX_INTEGER_LENGTH = 256
_MAX_EXPONENT = 1_000_000


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class PerfectPowerProfileRequest(StrictModel):
    """One canonical integer for the maximal perfect-power profile."""

    value: str = Field(pattern=r"^-?(?:0|[1-9][0-9]*)$", max_length=_MAX_INTEGER_LENGTH)


class KFreeDecompositionRequest(StrictModel):
    """One integer and exponent k >= 2 for k-free decomposition."""

    value: str = Field(pattern=r"^-?(?:0|[1-9][0-9]*)$", max_length=_MAX_INTEGER_LENGTH)
    k: int = Field(ge=2, le=10_000)


class SquarefreeDecompositionRequest(StrictModel):
    """One integer for squarefree decomposition."""

    value: str = Field(pattern=r"^-?(?:0|[1-9][0-9]*)$", max_length=_MAX_INTEGER_LENGTH)


class QuadraticRadicalNormalizeRequest(StrictModel):
    """One nonnegative integer for positive quadratic-radical normalization."""

    value: str = Field(pattern=r"^(?:0|[1-9][0-9]*)$", max_length=_MAX_INTEGER_LENGTH)


# ---------------------------------------------------------------------------
# Result models — perfect power profile
# ---------------------------------------------------------------------------


class PrimeExponentRow(StrictModel):
    """One prime-exponent row from a complete factorization."""

    prime: str = Field(pattern=r"^[1-9][0-9]*$")
    exponent: int = Field(ge=1, le=_MAX_EXPONENT)


class MaximalPerfectPowerResult(StrictModel):
    """The maximal perfect-power profile of one integer.

    Zero, positive unit (1), and negative unit (-1) have no finite maximal
    exponent and use closed structural variants.  Nonunit values carry a
    canonical base, maximal exponent, exact reconstruction, and the complete
    prime-exponent derivation.
    """

    source: str
    classification: Literal["ZERO", "POSITIVE_UNIT", "NEGATIVE_UNIT", "NONUNIT"]

    base: str | None = Field(default=None, pattern=r"^-?[1-9][0-9]*$")
    exponent: int | None = Field(default=None, ge=1, le=_MAX_EXPONENT)
    is_nontrivial_perfect_power: bool | None = None
    factors: tuple[PrimeExponentRow, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def bind_nonunit_fields(self) -> Self:
        if self.classification == "NONUNIT":
            if self.base is None or self.exponent is None:
                raise ValueError(
                    "nonunit perfect-power profile requires base and exponent"
                )
            if self.is_nontrivial_perfect_power is None:
                raise ValueError("nonunit profile requires is_nontrivial_perfect_power")
            if int(self.base) ** self.exponent != int(self.source):
                raise ValueError("base^exponent does not reconstruct the source")
            if not self.factors:
                raise ValueError("nonunit profile requires prime-exponent factors")
        else:
            if self.base is not None or self.exponent is not None:
                raise ValueError("unit/zero profiles must not carry base or exponent")
        return self


# ---------------------------------------------------------------------------
# Result models — k-free decomposition
# ---------------------------------------------------------------------------


class PrimeQuotientRemainderRow(StrictModel):
    """One prime with quotient and remainder when its exponent is divided by k."""

    prime: str = Field(pattern=r"^[1-9][0-9]*$")
    quotient: int = Field(ge=0, le=_MAX_EXPONENT)
    remainder: int = Field(ge=0, le=_MAX_EXPONENT)


class KFreeDecompositionResult(StrictModel):
    """The canonical k-free decomposition of one integer: n = a^k * c.

    For nonzero n: a >= 1, c has the same sign as n, and no prime to the k-th
    power divides |c|.

    For n = 0: classification is ZERO and a, c are not applicable.
    """

    source: str
    k: int = Field(ge=2, le=10_000)
    classification: Literal["ZERO", "NONZERO"]

    extracted_base: str | None = Field(default=None, pattern=r"^[1-9][0-9]*$|^0$")
    k_free_cofactor: str | None = Field(default=None)
    factor_rows: tuple[PrimeQuotientRemainderRow, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def bind_nonzero_fields(self) -> Self:
        if self.classification == "NONZERO":
            if self.extracted_base is None or self.k_free_cofactor is None:
                raise ValueError("nonzero decomposition requires base and cofactor")
            a = int(self.extracted_base)
            c = int(self.k_free_cofactor)
            if a**self.k * c != int(self.source):
                raise ValueError("a^k * c does not reconstruct the source")
        else:
            if self.extracted_base is not None or self.k_free_cofactor is not None:
                raise ValueError("zero decomposition must not carry base or cofactor")
        return self


# ---------------------------------------------------------------------------
# Result models — squarefree decomposition
# ---------------------------------------------------------------------------


class PrimeExponentParityRow(StrictModel):
    """One prime with its exponent and parity (exponent mod 2)."""

    prime: str = Field(pattern=r"^[1-9][0-9]*$")
    exponent: int = Field(ge=1, le=_MAX_EXPONENT)
    parity: int = Field(ge=0, le=1)


class SquarefreeDecompositionResult(StrictModel):
    """The squarefree decomposition of one integer: n = s^2 * d.

    For nonzero n: s >= 1, d has the same sign as n, |d| is squarefree.
    For n = 0: classification is ZERO and s, d are not applicable.
    """

    source: str
    classification: Literal["ZERO", "NONZERO"]

    square_factor: str | None = Field(default=None, pattern=r"^[1-9][0-9]*$|^0$")
    signed_squarefree_part: str | None = Field(default=None)
    parity_rows: tuple[PrimeExponentParityRow, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def bind_nonzero_fields(self) -> Self:
        if self.classification == "NONZERO":
            if self.square_factor is None or self.signed_squarefree_part is None:
                raise ValueError(
                    "nonzero decomposition requires square factor and part"
                )
            s = int(self.square_factor)
            d = int(self.signed_squarefree_part)
            if s * s * d != int(self.source):
                raise ValueError("s^2 * d does not reconstruct the source")
        else:
            if (
                self.square_factor is not None
                or self.signed_squarefree_part is not None
            ):
                raise ValueError(
                    "zero decomposition must not carry square factor or part"
                )
        return self


# ---------------------------------------------------------------------------
# Result models — normalized positive quadratic radical
# ---------------------------------------------------------------------------


class NormalizedQuadraticRadicalResult(StrictModel):
    """The canonical positive square root: sqrt(n) = s * sqrt(d).

    For n = 0: s = 0, d = 1, classification is ZERO.
    For n > 0: n = s^2 * d from squarefree decomposition, d >= 1 squarefree.
    """

    source: str
    coefficient: str = Field(pattern=r"^(?:0|[1-9][0-9]*)$")
    radicand: str = Field(pattern=r"^[1-9][0-9]*$")
    classification: Literal["ZERO", "RATIONAL_INTEGER", "IRRATIONAL_QUADRATIC"]

    @model_validator(mode="after")
    def verify_reconstruction(self) -> Self:
        s = int(self.coefficient)
        d = int(self.radicand)
        source = int(self.source)
        if s * s * d != source:
            raise ValueError("s^2 * d does not reconstruct the source")
        if self.classification == "ZERO" and (s != 0 or d != 1):
            raise ValueError("zero classification requires s=0, d=1")
        if self.classification == "RATIONAL_INTEGER" and (d != 1 or s <= 0):
            raise ValueError("rational classification requires d=1 and s>0")
        if self.classification == "IRRATIONAL_QUADRATIC" and (d <= 1 or s <= 0):
            raise ValueError("irrational classification requires d>1 and s>0")
        return self
