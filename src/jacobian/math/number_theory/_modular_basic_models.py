"""Contracts owned by basic modular-arithmetic kernels.

Polynomial residue-image contracts remain in ``_modular_models``; this module
owns the distinct unit, CRT, Jacobi, and quadratic-residue envelopes.
"""

from __future__ import annotations

from math import gcd
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

MAX_MODULUS = 1_000_000
MAX_CRT_SIZE = 64
# The CRT result carries its combined modulus as a canonical integer whose
# width is bounded by the neutral integer grammar.
MAX_CRT_COMBINED_MODULUS = 10**256


class ModularValueRequest(StrictModel):
    """One canonical integer and a bounded modulus."""

    value: BoundedInteger
    modulus: StrictInt = Field(ge=2, le=MAX_MODULUS)


class ModularUnitRequest(ModularValueRequest):
    """One canonical integer that is a unit modulo the supplied modulus."""

    @model_validator(mode="after")
    def require_coprime(self) -> Self:
        if gcd(int(self.value), self.modulus) != 1:
            raise _validation_error(
                "value_must_be_coprime_to_the_modulus",
                "value must be coprime to the modulus",
            )
        return self


class ModulusRequest(StrictModel):
    """A single bounded modulus."""

    modulus: StrictInt = Field(ge=2, le=MAX_MODULUS)


class ChineseRemainderRequest(StrictModel):
    """A finite, compatible system of canonical integer congruences."""

    residues: tuple[int, ...] = Field(min_length=1, max_length=MAX_CRT_SIZE)
    moduli: tuple[int, ...] = Field(min_length=1, max_length=MAX_CRT_SIZE)

    @model_validator(mode="after")
    def require_parallel_positive_moduli(self) -> Self:
        if len(self.residues) != len(self.moduli):
            raise _validation_error(
                "residues_and_moduli_must_have_equal_length",
                "residues and moduli must have equal length",
            )
        if any(modulus < 2 or modulus > MAX_MODULUS for modulus in self.moduli):
            raise _validation_error(
                "every_modulus_must_be_between_2_and_1_000_000",
                "every modulus must be between 2 and 1,000,000",
            )
        combined = 1
        for modulus in self.moduli:
            combined = combined // gcd(combined, modulus) * modulus
            if combined > MAX_CRT_COMBINED_MODULUS:
                raise _validation_error(
                    "the_system_s_combined_modulus_must_have_at",
                    "the system's combined modulus must have at most 256 digits; "
                    "split the congruence system into narrower subsystems",
                )
        if any(
            residue < 0 or residue >= modulus
            for residue, modulus in zip(self.residues, self.moduli, strict=True)
        ):
            raise _validation_error(
                "every_residue_must_be_canonical_for_its_modulus",
                "every residue must be canonical for its modulus",
            )
        for index, modulus in enumerate(self.moduli):
            for other_index in range(index + 1, len(self.moduli)):
                common_divisor = gcd(modulus, self.moduli[other_index])
                if (self.residues[index] - self.residues[other_index]) % common_divisor:
                    raise _validation_error(
                        "congruence_system_is_inconsistent",
                        "congruence system is inconsistent",
                    )
        return self


class JacobiSymbolRequest(StrictModel):
    """Arguments for the Jacobi symbol ``(a / n)`` with odd positive ``n``."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=MAX_MODULUS)

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise _validation_error(
                "jacobi_symbol_denominator_must_be_odd",
                "Jacobi symbol denominator must be odd",
            )
        return self


class QuadraticResiduesResult(StrictModel):
    """All quadratic residues modulo one admitted modulus."""

    residues: tuple[BoundedInteger, ...]


class ChineseRemainderResult(StrictModel):
    """The least non-negative solution and modulus of a CRT system."""

    residue: BoundedInteger
    modulus: BoundedInteger


class JacobiSymbolResult(StrictModel):
    """The exact Jacobi symbol, bound to its normalized arguments."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=MAX_MODULUS)
    jacobi: Literal[-1, 0, 1]

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise _validation_error(
                "jacobi_symbol_denominator_must_be_odd",
                "Jacobi symbol denominator must be odd",
            )
        return self


__all__ = [
    "MAX_CRT_COMBINED_MODULUS",
    "MAX_CRT_SIZE",
    "MAX_MODULUS",
    "ChineseRemainderRequest",
    "ChineseRemainderResult",
    "JacobiSymbolRequest",
    "JacobiSymbolResult",
    "ModularUnitRequest",
    "ModularValueRequest",
    "ModulusRequest",
    "QuadraticResiduesResult",
]
