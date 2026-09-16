"""Exact tropical-semiring scalar values."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_TROPICAL_SCALAR_DIGITS = 8_192


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tropical.{reason}", message)


class TropicalSemiring(StrictModel):
    """One explicit tropical semiring over a bounded exact ordered group.

    ``MIN_PLUS`` is QQ (or ZZ) union {+infinity} with tropical addition
    ``min``; ``MAX_PLUS`` uses {-infinity} with ``max``. In both cases
    tropical multiplication is ordinary addition, the additive identity
    (``zero_T``) is the licensed infinity, and the multiplicative identity
    (``one_T``) is zero.
    """

    convention: Literal["MIN_PLUS", "MAX_PLUS"] = Field(
        description="Idempotent addition convention: min-plus or max-plus."
    )
    base: Literal["ZZ", "QQ"] = Field(
        description="Exact totally ordered additive group of finite scalars."
    )


class TropicalScalar(StrictModel):
    """One exact scalar bound to its tropical semiring identity.

    Only the infinity licensed by the semiring convention is an ordinary
    element: ``POSITIVE_INFINITY`` belongs to ``MIN_PLUS`` and
    ``NEGATIVE_INFINITY`` belongs to ``MAX_PLUS``. Finite scalars over
    ``ZZ`` must be integral.
    """

    semiring: TropicalSemiring
    kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"] = Field(
        description="Closed variant: one finite rational or the licensed infinity."
    )
    value: CanonicalRational | None = Field(
        default=None,
        description="Finite value; present exactly when kind is FINITE.",
    )

    @model_validator(mode="after")
    def require_licensed_scalar(self) -> Self:
        if self.kind == "FINITE":
            if self.value is None:
                raise _validation_error(
                    "finite_missing_value", "a finite scalar must carry its value"
                )
            if self.semiring.base == "ZZ" and self.value.den != 1:
                raise _validation_error(
                    "nonintegral_integer_scalar",
                    "a ZZ tropical scalar must be integral",
                )
        else:
            if self.value is not None:
                raise _validation_error(
                    "infinite_carries_value", "an infinite scalar carries no value"
                )
            if self.kind == "POSITIVE_INFINITY" and self.semiring.convention != (
                "MIN_PLUS"
            ):
                raise _validation_error(
                    "unlicensed_infinity",
                    "positive infinity is licensed only by MIN_PLUS",
                )
            if self.kind == "NEGATIVE_INFINITY" and self.semiring.convention != (
                "MAX_PLUS"
            ):
                raise _validation_error(
                    "unlicensed_infinity",
                    "negative infinity is licensed only by MAX_PLUS",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        semiring: TropicalSemiring,
        kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"],
        value: CanonicalRational | None,
    ) -> Self:
        """Construct a scalar after the kernel established its branch."""

        return cls.model_construct(semiring=semiring, kind=kind, value=value)


def require_scalar_budget(scalar: TropicalScalar) -> None:
    """Preflight finite-scalar digits before tropical arithmetic."""

    if scalar.value is not None:
        try:
            require_bounded_rational(
                scalar.value,
                max_digits=MAX_TROPICAL_SCALAR_DIGITS,
                label="tropical scalar",
            )
        except ValueError as error:
            raise _validation_error("scalar_budget", str(error)) from error


__all__ = [
    "MAX_TROPICAL_SCALAR_DIGITS",
    "TropicalScalar",
    "TropicalSemiring",
    "require_scalar_budget",
]
