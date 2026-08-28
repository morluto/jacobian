"""Typed wire contracts for recurrence solving."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 256
MAX_RATIONAL_SEQUENCE_LENGTH = 256
MAX_CLOSED_FORM_ORDER = 16


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"recurrence_solving.{reason}", message)


class RecurrenceFindRequest(StrictModel):
    """Find the minimal linear recurrence of a sequence over QQ."""

    sequence: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=MAX_RATIONAL_SEQUENCE_LENGTH
    )


class RecurrenceFindResult(StrictModel):
    """A fitted recurrence or an explicit finite-prefix missing outcome."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_RATIONAL_SEQUENCE_LENGTH - 1
    )
    order: int = Field(ge=0, le=MAX_RATIONAL_SEQUENCE_LENGTH - 1)
    status: Literal["FOUND", "NO_FITTING_RECURRENCE"]
    method: Literal["RATIONAL_INTERPOLATION"] = "RATIONAL_INTERPOLATION"

    @model_validator(mode="after")
    def require_status_consistent_coefficients(self) -> Self:
        if self.status == "FOUND":
            if self.order == 0 or len(self.coefficients) != self.order:
                raise _validation_error(
                    "found_coefficients_mismatch",
                    "a found recurrence must have one coefficient per order",
                )
        elif self.order != 0 or self.coefficients:
            raise _validation_error(
                "missing_coefficients_mismatch",
                "a missing recurrence must have zero order and no coefficients",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        coefficients: tuple[CanonicalRational, ...],
        order: int,
        status: Literal["FOUND", "NO_FITTING_RECURRENCE"],
    ) -> Self:
        """Construct a rational recurrence result from the trusted kernel."""

        return cls.model_construct(
            coefficients=coefficients,
            order=order,
            status=status,
            method="RATIONAL_INTERPOLATION",
        )


class ClosedFormRequest(StrictModel):
    """Compute a bounded SymPy-expression closed form for a recurrence."""

    characteristic_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CLOSED_FORM_ORDER + 1,
        description=(
            "Characteristic polynomial coefficients in descending order, "
            f"with degree at most {MAX_CLOSED_FORM_ORDER}."
        ),
    )
    initial_values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_CLOSED_FORM_ORDER
    )


class ClosedFormResult(StrictModel):
    """The closed-form solution as a SymPy expression string."""

    expression: str
    method: Literal["SYMPY_RSOLVE"] = "SYMPY_RSOLVE"

    @classmethod
    def _from_kernel(cls, *, expression: str) -> Self:
        """Construct a closed-form result from the trusted SymPy adapter."""

        return cls.model_construct(expression=expression, method="SYMPY_RSOLVE")


# ---------------------------------------------------------------------------
# Berlekamp-Massey over an explicit prime field
# ---------------------------------------------------------------------------

_MAX_FIELD_SEQUENCE_LENGTH = 256
_MAX_FIELD_PRIME = 10_000


def _require_canonical_residues(
    values: tuple[int, ...], prime: int, label: str
) -> None:
    for value in values:
        if type(value) is not int or not 0 <= value < prime:
            raise _validation_error(
                "noncanonical_residue",
                f"{label} must be canonical residues modulo the prime",
            )


class PrimeFieldRecurrence(StrictModel):
    """Minimal linear recurrence over an explicit prime field ``GF(p)``.

    The domain-owned canonical recurrence value: native producers return it
    and MCP results embed it unchanged, so native and wire consumers share
    one type.  The recurrence is established only on the observed prefix
    ``L <= n < len(sequence)``; it carries no claim about unobserved terms.
    Every admitted finite sequence admits a recurrence of order at most its
    own length (order ``len(sequence)`` fits vacuously), so the outcome is
    always a fitted recurrence and no missing-recurrence state exists.
    """

    prime: StrictInt = Field(ge=2, le=_MAX_FIELD_PRIME)
    coefficients: tuple[StrictInt, ...] = Field(
        max_length=_MAX_FIELD_SEQUENCE_LENGTH,
        description=(
            "Recurrence coefficients (c_1, ..., c_L) as canonical residues "
            "modulo the prime: each value satisfies 0 <= value < prime."
        ),
    )
    order: StrictInt = Field(ge=0, le=_MAX_FIELD_SEQUENCE_LENGTH)
    status: Literal["FOUND"] = Field(
        description="Always FOUND: every admitted finite sequence fits a recurrence."
    )

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _require_canonical_residues(self.coefficients, self.prime, "coefficients")
        if self.order != len(self.coefficients):
            raise _validation_error(
                "order_coefficients_mismatch",
                "order must equal the number of coefficients",
            )
        return self


class PrimeFieldRecurrenceFindRequest(StrictModel):
    """Find the minimal linear recurrence of a sequence over ``GF(p)``."""

    prime: StrictInt = Field(
        ge=2,
        le=_MAX_FIELD_PRIME,
        description=f"Prime modulus p of the field GF(p), between 2 and {_MAX_FIELD_PRIME}.",
    )
    sequence: tuple[StrictInt, ...] = Field(
        min_length=2,
        max_length=_MAX_FIELD_SEQUENCE_LENGTH,
        description=(
            "Finite sequence of canonical residues modulo the supplied prime: "
            "each value must be an integer with 0 <= value < prime; negative "
            "or oversized representatives are rejected."
        ),
    )


class PrimeFieldRecurrenceFindResult(StrictModel):
    """The minimal LFSR over ``GF(p)`` found by Berlekamp-Massey.

    Embeds the canonical :class:`PrimeFieldRecurrence` value rather than
    re-flattening its fields, so native and MCP producers expose one
    compatible public type.
    """

    sequence: tuple[StrictInt, ...] = Field(
        min_length=2,
        max_length=_MAX_FIELD_SEQUENCE_LENGTH,
        description=(
            "The supplied sequence of canonical residues modulo the prime: "
            "each value satisfies 0 <= value < prime."
        ),
    )
    recurrence: PrimeFieldRecurrence
    method: Literal["BERLEKAMP_MASSEY"] = "BERLEKAMP_MASSEY"

    @model_validator(mode="after")
    def require_status_consistent_coefficients(self) -> Self:
        _require_canonical_residues(
            self.sequence, self.recurrence.prime, "sequence values"
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        sequence: tuple[int, ...],
        recurrence: PrimeFieldRecurrence,
    ) -> Self:
        """Construct a result from the trusted Berlekamp-Massey kernel.

        Normal model parsing deliberately checks only the bounded wire
        structure.
        """

        return cls.model_construct(
            sequence=sequence,
            recurrence=recurrence,
            method="BERLEKAMP_MASSEY",
        )


__all__ = [
    "ClosedFormRequest",
    "ClosedFormResult",
    "PrimeFieldRecurrence",
    "PrimeFieldRecurrenceFindRequest",
    "PrimeFieldRecurrenceFindResult",
    "RecurrenceFindRequest",
    "RecurrenceFindResult",
]
