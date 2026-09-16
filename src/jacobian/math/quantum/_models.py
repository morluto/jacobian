"""Typed contracts for exact binary stabilizer check spaces."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"stabilizer.{reason}", message)


MAX_QUBITS = 32
"""Maximum labelled qubits in one admitted check-space request."""

MAX_CHECK_ROWS = 64
"""Maximum phase-free generator rows in one admitted request."""

MAX_QUBIT_LABEL_LENGTH = 64
"""Maximum length of a qubit or generator identifier."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


QubitId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH, strict=True),
    AfterValidator(_require_scalar_label),
]
"""One labelled qubit in an ordered register."""


class BinaryPauliRow(StrictModel):
    """One phase-free Pauli generator ``(x | z)`` on the register axis.

    Per-qubit pairs read ``(0,0)=I``, ``(1,0)=X``, ``(0,1)=Z``,
    ``(1,1)=Y`` modulo phase. The row is a coset of the Pauli group modulo
    scalar phases; commutation is decided by the binary symplectic pairing
    ``<v, w> = x.z' + z.x' mod 2``.
    """

    row_id: Annotated[str, AfterValidator(_require_scalar_label)] = Field(
        min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH
    )
    x_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    z_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)

    @model_validator(mode="after")
    def require_binary_bits(self) -> Self:
        if any(bit not in (0, 1) for bit in (*self.x_bits, *self.z_bits)):
            raise _validation_error(
                "pauli_bits_binary", "phase-free Pauli bits must be 0 or 1"
            )
        if len(self.x_bits) != len(self.z_bits):
            raise _validation_error(
                "pauli_row_shape",
                "x and z bit rows must share one register length",
            )
        return self


class CheckSpaceCanonicalizeRequest(StrictModel):
    """Canonicalize a binary check matrix over a labelled qubit register.

    The request carries an ordered register of 1 to ``MAX_QUBITS`` unique
    qubits and 1 to ``MAX_CHECK_ROWS`` labelled generator rows whose bit
    rows match the register length. Row order is presentation only; the
    canonical basis is independent of it.
    """

    qubit_ids: tuple[QubitId, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    generators: tuple[BinaryPauliRow, ...] = Field(
        min_length=1, max_length=MAX_CHECK_ROWS
    )

    @model_validator(mode="after")
    def require_register_binding(self) -> Self:
        if len(set(self.qubit_ids)) != len(self.qubit_ids):
            raise _validation_error("qubit_ids_unique", "qubit IDs must be unique")
        row_ids = tuple(row.row_id for row in self.generators)
        if tuple(sorted(row_ids)) != row_ids or len(set(row_ids)) != len(row_ids):
            raise _validation_error(
                "generator_row_ids",
                "generator row IDs must be unique and strictly ordered",
            )
        width = len(self.qubit_ids)
        if any(len(row.x_bits) != width for row in self.generators):
            raise _validation_error(
                "register_binding",
                "every generator bit row must match the register length",
            )
        return self


CheckSpaceStatus = Literal["ISOTROPIC_CHECK_SPACE", "NOT_ISOTROPIC"]


class CanonicalCheckRow(StrictModel):
    """One RREF basis row of the canonical isotropic check space."""

    pivot: StrictInt = Field(ge=0)
    x_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    z_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)


class NonCommutingWitness(StrictModel):
    """Two generator rows with nonzero symplectic pairing."""

    first_row_id: str = Field(min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH)
    second_row_id: str = Field(min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH)
    pairing: Literal[1] = 1


class CheckSpaceCanonicalizeResult(StrictModel):
    """Canonical RREF plus rank for isotropic families, else a witness pair."""

    status: CheckSpaceStatus
    qubit_ids: tuple[QubitId, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    rank: StrictInt = Field(ge=0)
    basis: tuple[CanonicalCheckRow, ...] = Field(default=(), max_length=MAX_CHECK_ROWS)
    witness: NonCommutingWitness | None = None

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        if (self.status == "ISOTROPIC_CHECK_SPACE") == (self.witness is not None):
            raise _validation_error(
                "check_space_branch",
                "isotropic results carry a basis without a witness; "
                "non-isotropic results carry a witness without a basis",
            )
        if self.status == "ISOTROPIC_CHECK_SPACE":
            if self.rank != len(self.basis):
                raise _validation_error(
                    "check_space_rank",
                    "isotropic rank must equal the canonical basis length",
                )
            pivots = tuple(row.pivot for row in self.basis)
            if pivots != tuple(sorted(pivots)) or len(set(pivots)) != len(pivots):
                raise _validation_error(
                    "check_space_pivots",
                    "canonical basis pivots must be unique and ordered",
                )
        elif self.rank != 0 or self.basis:
            raise _validation_error(
                "check_space_witness_rank",
                "non-isotropic results carry no rank or basis rows",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: CheckSpaceStatus,
        qubit_ids: tuple[QubitId, ...],
        rank: int,
        basis: tuple[CanonicalCheckRow, ...],
        witness: NonCommutingWitness | None,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its elimination."""

        return cls.model_construct(
            status=status,
            qubit_ids=qubit_ids,
            rank=rank,
            basis=basis,
            witness=witness,
        )


__all__ = [
    "MAX_CHECK_ROWS",
    "MAX_QUBITS",
    "MAX_QUBIT_LABEL_LENGTH",
    "BinaryPauliRow",
    "CanonicalCheckRow",
    "CheckSpaceCanonicalizeRequest",
    "CheckSpaceCanonicalizeResult",
    "CheckSpaceStatus",
    "NonCommutingWitness",
    "QubitId",
]
