"""Typed contracts for exact binary stabilizer check spaces."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    ConfigDict,
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

    row_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH, strict=True),
        AfterValidator(_require_scalar_label),
    ]
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
    rows match the register length. Generator IDs are unique and strictly
    ordered so the serialized request has one canonical row presentation.
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


class QubitRegister(StrictModel):
    """An ordered register that is the parent of every compact Pauli value."""

    qubit_ids: tuple[QubitId, ...] = Field(min_length=1, max_length=MAX_QUBITS)

    @model_validator(mode="after")
    def require_unique_ids(self) -> Self:
        if len(set(self.qubit_ids)) != len(self.qubit_ids):
            raise _validation_error(
                "register_ids_unique", "qubit register IDs must be unique"
            )
        return self


class PhaseFreeQubitPauli(StrictModel):
    model_config = ConfigDict(populate_by_name=True)
    """A phase-free Pauli vector on one explicit qubit register."""

    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    x_bits: tuple[StrictInt, ...]
    z_bits: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_register_shape(self) -> Self:
        width = len(self.qubit_register.qubit_ids)
        if len(self.x_bits) != width or len(self.z_bits) != width:
            raise _validation_error(
                "pauli_register_binding", "Pauli coordinates must match their register"
            )
        if any(bit not in (0, 1) for bit in (*self.x_bits, *self.z_bits)):
            raise _validation_error(
                "pauli_bits_binary", "phase-free Pauli bits must be binary"
            )
        return self

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(
            q
            for q, x, z in zip(
                self.qubit_register.qubit_ids, self.x_bits, self.z_bits, strict=True
            )
            if x or z
        )

    @property
    def weight(self) -> int:
        return len(self.support)


class ExactQubitPauli(StrictModel):
    """The exact Pauli ``i^phase X^x Z^z`` under Jacobian's fixed convention."""

    phase_free: PhaseFreeQubitPauli
    phase: StrictInt = Field(ge=0, le=3)

    @property
    def register(self) -> QubitRegister:
        return self.phase_free.qubit_register


class PauliProductRequest(StrictModel):
    left: ExactQubitPauli
    right: ExactQubitPauli


class PauliInverseRequest(StrictModel):
    pauli: ExactQubitPauli


class PauliProductResult(StrictModel):
    left: ExactQubitPauli
    right: ExactQubitPauli
    product: ExactQubitPauli


class PauliInverseResult(StrictModel):
    source: ExactQubitPauli
    inverse: ExactQubitPauli


class PauliPairingRequest(StrictModel):
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli


class PauliPairingResult(StrictModel):
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli
    pairing: StrictInt = Field(ge=0, le=1)
    commute: bool


class CheckSpaceValue(StrictModel):
    model_config = ConfigDict(populate_by_name=True)
    """An isotropic check space with an explicit register parent."""

    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_CHECK_ROWS)

    @model_validator(mode="after")
    def require_basis_parent(self) -> Self:
        if any(row.qubit_register != self.qubit_register for row in self.basis):
            raise _validation_error(
                "check_space_parent", "all check rows must use the same register"
            )
        return self


class NormalizerResult(StrictModel):
    """The exact symplectic orthogonal space S-perp of an isotropic check space."""

    check_space: CheckSpaceValue
    orthogonal_basis: tuple[PhaseFreeQubitPauli, ...]
    rank: int = Field(ge=0)
    orthogonal_rank: int = Field(ge=0)
    logical_dimension: int = Field(ge=0)

    @model_validator(mode="after")
    def require_dimensions(self) -> Self:
        if not isinstance(self.check_space, CheckSpaceValue) or not isinstance(
            self.check_space.qubit_register, QubitRegister
        ):
            raise _validation_error(
                "normalizer_parent", "normalizer check-space parent is malformed"
            )
        if (
            not isinstance(self.check_space.basis, tuple)
            or any(not isinstance(row, PhaseFreeQubitPauli) for row in self.orthogonal_basis)
        ):
            raise _validation_error(
                "normalizer_parent", "normalizer rows and check basis must be typed values"
            )
        if self.rank != len(self.check_space.basis) or self.orthogonal_rank != len(
            self.orthogonal_basis
        ):
            raise _validation_error(
                "normalizer_dimensions", "normalizer ranks must match retained bases"
            )
        if any(
            row.qubit_register != self.check_space.qubit_register
            for row in self.orthogonal_basis
        ):
            raise _validation_error(
                "normalizer_parent", "every normalizer row must use the check-space register"
            )
        if self.logical_dimension != self.orthogonal_rank - self.rank:
            raise _validation_error(
                "logical_dimension", "logical dimension must be dim(S-perp)-dim(S)"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        check_space: CheckSpaceValue,
        orthogonal_basis: tuple[PhaseFreeQubitPauli, ...],
    ) -> Self:
        return cls.model_construct(
            check_space=check_space,
            orthogonal_basis=orthogonal_basis,
            rank=len(check_space.basis),
            orthogonal_rank=len(orthogonal_basis),
            logical_dimension=len(orthogonal_basis) - len(check_space.basis),
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
    "CheckSpaceValue",
    "ExactQubitPauli",
    "NonCommutingWitness",
    "NormalizerResult",
    "PauliInverseRequest",
    "PauliInverseResult",
    "PauliPairingRequest",
    "PauliPairingResult",
    "PauliProductRequest",
    "PauliProductResult",
    "PhaseFreeQubitPauli",
    "QubitId",
    "QubitRegister",
]
