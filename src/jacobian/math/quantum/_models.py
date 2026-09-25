"""Typed contracts for exact binary stabilizer check spaces."""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictBool,
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


class PauliFromLabelsRequest(StrictModel):
    """A complete ordered I/X/Y/Z row and its scalar phase."""

    model_config = ConfigDict(populate_by_name=True)

    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    labels: tuple[Literal["I", "X", "Y", "Z"], ...] = Field(
        min_length=1, max_length=MAX_QUBITS
    )
    phase: StrictInt = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def require_complete_row(self) -> Self:
        if len(self.labels) != len(self.qubit_register.qubit_ids):
            raise _validation_error(
                "label_register_shape",
                "Pauli labels must cover the full ordered register",
            )
        return self


class PauliFromLabelsResult(StrictModel):
    source: PauliFromLabelsRequest
    pauli: ExactQubitPauli

    @model_validator(mode="after")
    def require_register_binding(self) -> Self:
        if self.pauli.register != self.source.qubit_register:
            raise _validation_error(
                "label_result_register",
                "converted Pauli must retain the source register",
            )
        return self


class PauliToLabelsRequest(StrictModel):
    pauli: ExactQubitPauli


class PauliToLabelsResult(StrictModel):
    source: ExactQubitPauli
    labels: tuple[Literal["I", "X", "Y", "Z"], ...] = Field(
        min_length=1, max_length=MAX_QUBITS
    )
    phase: StrictInt = Field(ge=0, le=3)

    @model_validator(mode="after")
    def require_complete_row(self) -> Self:
        if len(self.labels) != len(self.source.register.qubit_ids):
            raise _validation_error(
                "label_result_shape", "labels must cover the complete Pauli register"
            )
        return self


def _valid_register(value: object) -> bool:
    if not isinstance(value, QubitRegister):
        return False
    ids = getattr(value, "qubit_ids", None)
    return (
        type(ids) is tuple
        and 1 <= len(ids) <= MAX_QUBITS
        and all(
            type(identifier) is str
            and 1 <= len(identifier) <= MAX_QUBIT_LABEL_LENGTH
            and not any(0xD800 <= ord(character) <= 0xDFFF for character in identifier)
            for identifier in ids
        )
        and len(set(ids)) == len(ids)
    )


def _valid_phase_free(value: object) -> bool:
    if not isinstance(value, PhaseFreeQubitPauli):
        return False
    register = getattr(value, "qubit_register", None)
    x_bits = getattr(value, "x_bits", None)
    z_bits = getattr(value, "z_bits", None)
    if not _valid_register(register):
        return False
    register = cast(QubitRegister, register)
    width = len(register.qubit_ids)
    return (
        type(x_bits) is tuple
        and type(z_bits) is tuple
        and len(x_bits) == width
        and len(z_bits) == width
        and all(type(bit) is int and bit in (0, 1) for bit in (*x_bits, *z_bits))
    )


def _valid_exact(value: object) -> bool:
    if not isinstance(value, ExactQubitPauli):
        return False
    phase = getattr(value, "phase", None)
    return (
        _valid_phase_free(getattr(value, "phase_free", None))
        and type(phase) is int
        and 0 <= phase <= 3
    )


def _require_exact_registers(values: tuple[object, ...], reason: str) -> None:
    if not all(_valid_exact(value) for value in values):
        raise _validation_error(
            reason, "result Paulis must be valid exact register-bound values"
        )
    exact_values = tuple(cast(ExactQubitPauli, value) for value in values)
    registers = [value.phase_free.qubit_register for value in exact_values]
    if any(register.qubit_ids != registers[0].qubit_ids for register in registers[1:]):
        raise _validation_error(
            "register_binding",
            "all result Paulis must use one identical ordered qubit register",
        )


def _require_phase_free_registers(values: tuple[object, ...]) -> None:
    if not all(_valid_phase_free(value) for value in values):
        raise _validation_error(
            "register_binding", "result Paulis must be valid register-bound values"
        )
    phase_free_values = tuple(cast(PhaseFreeQubitPauli, value) for value in values)
    registers = [value.qubit_register for value in phase_free_values]
    if any(register.qubit_ids != registers[0].qubit_ids for register in registers[1:]):
        raise _validation_error(
            "register_binding",
            "all result Paulis must use one identical ordered qubit register",
        )


class PauliProductResult(StrictModel):
    left: ExactQubitPauli
    right: ExactQubitPauli
    product: ExactQubitPauli

    @model_validator(mode="after")
    def require_register_binding(self) -> Self:
        _require_exact_registers(
            (self.left, self.right, self.product), "product_parent"
        )
        return self


class PauliInverseResult(StrictModel):
    source: ExactQubitPauli
    inverse: ExactQubitPauli

    @model_validator(mode="after")
    def require_register_binding(self) -> Self:
        _require_exact_registers((self.source, self.inverse), "inverse_parent")
        return self


class PauliPairingRequest(StrictModel):
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli


class PauliPairingResult(StrictModel):
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli
    pairing: StrictInt = Field(ge=0, le=1)
    commute: bool

    @model_validator(mode="after")
    def require_pairing_binding(self) -> Self:
        _require_phase_free_registers((self.left, self.right))
        if type(self.pairing) is not int or self.pairing not in (0, 1):
            raise _validation_error(
                "pairing_value", "Pauli pairing must be zero or one"
            )
        if type(self.commute) is not bool or self.commute != (self.pairing == 0):
            raise _validation_error(
                "pairing_commute",
                "commute must be equivalent to a zero symplectic pairing",
            )
        return self


class PauliFamilyEntry(StrictModel):
    """One named phase-free Pauli row in an ordered family."""

    pauli_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=MAX_QUBIT_LABEL_LENGTH, strict=True),
        AfterValidator(_require_scalar_label),
    ]
    pauli: PhaseFreeQubitPauli


class PauliFamilyCommutationRequest(StrictModel):
    """An ordered named family of Paulis on one common register."""

    family: tuple[PauliFamilyEntry, ...] = Field(
        min_length=1, max_length=MAX_CHECK_ROWS
    )

    @model_validator(mode="after")
    def require_family_axis(self) -> Self:
        identifiers = tuple(entry.pauli_id for entry in self.family)
        if len(set(identifiers)) != len(identifiers):
            raise _validation_error("family_ids_unique", "Pauli IDs must be unique")
        register = self.family[0].pauli.qubit_register
        if any(entry.pauli.qubit_register != register for entry in self.family):
            raise _validation_error(
                "family_register",
                "every Pauli in the family must use one identical ordered register",
            )
        return self


class PauliFamilyCommutationResult(StrictModel):
    """The exact alternating commutation matrix, retaining its named row axis."""

    source: PauliFamilyCommutationRequest
    commutation_matrix: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_CHECK_ROWS
    )

    @model_validator(mode="after")
    def require_axis_and_form(self) -> Self:
        count = len(self.source.family)
        matrix = self.commutation_matrix
        if len(matrix) != count or any(len(row) != count for row in matrix):
            raise _validation_error(
                "commutation_matrix_shape",
                "matrix must be square on the retained family axis",
            )
        if any(bit not in (0, 1) for row in matrix for bit in row):
            raise _validation_error(
                "commutation_matrix_bits", "commutation matrix entries must be binary"
            )
        if any(matrix[i][i] != 0 for i in range(count)) or any(
            matrix[i][j] != matrix[j][i]
            for i in range(count)
            for j in range(i + 1, count)
        ):
            raise _validation_error(
                "commutation_matrix_form",
                "commutation matrix must be alternating and symmetric over GF(2)",
            )
        return self


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
    orthogonal_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=2 * MAX_QUBITS)
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
            or any(
                not isinstance(row, PhaseFreeQubitPauli)
                for row in self.check_space.basis
            )
            or not isinstance(self.orthogonal_basis, tuple)
            or len(self.orthogonal_basis) > 2 * MAX_QUBITS
            or any(
                not isinstance(row, PhaseFreeQubitPauli)
                for row in self.orthogonal_basis
            )
        ):
            raise _validation_error(
                "normalizer_parent",
                "normalizer rows and check basis must be typed values",
            )
        register = self.check_space.qubit_register
        width = len(register.qubit_ids)
        for row in (*self.check_space.basis, *self.orthogonal_basis):
            if (
                row.qubit_register != register
                or not isinstance(row.x_bits, tuple)
                or not isinstance(row.z_bits, tuple)
                or len(row.x_bits) != width
                or len(row.z_bits) != width
                or any(
                    type(bit) is not int or bit not in (0, 1)
                    for bit in (*row.x_bits, *row.z_bits)
                )
            ):
                raise _validation_error(
                    "normalizer_parent",
                    "normalizer rows must be valid binary values on the check register",
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
                "normalizer_parent",
                "every normalizer row must use the check-space register",
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


class StabilizerSyndromeRequest(StrictModel):
    """Measure one phase-free Pauli against a register-bound check space."""

    check_space: CheckSpaceValue
    error: PhaseFreeQubitPauli


class StabilizerSyndromeResult(StrictModel):
    """Syndrome bits indexed by the canonical RREF basis of the check space."""

    check_space: CheckSpaceValue
    error: PhaseFreeQubitPauli
    syndrome: tuple[StrictInt, ...] = Field(max_length=MAX_CHECK_ROWS)
    zero_syndrome: bool

    @model_validator(mode="after")
    def require_syndrome_binding(self) -> Self:
        register = self.check_space.qubit_register
        if self.error.qubit_register != register:
            raise _validation_error(
                "syndrome_register", "error and check space must share a register"
            )
        if len(self.syndrome) != len(self.check_space.basis):
            raise _validation_error(
                "syndrome_axis", "one syndrome bit is required per canonical check"
            )
        if any(bit not in (0, 1) for bit in self.syndrome):
            raise _validation_error("syndrome_bits", "syndrome entries must be bits")
        if type(self.zero_syndrome) is not bool:
            raise _validation_error("syndrome_zero", "zero_syndrome must be boolean")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        check_space: CheckSpaceValue,
        error: PhaseFreeQubitPauli,
        syndrome: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            check_space=check_space,
            error=error,
            syndrome=syndrome,
            zero_syndrome=not any(syndrome),
        )


class StabilizerErrorEquivalenceRequest(StrictModel):
    """Compare two phase-free errors modulo one isotropic check space."""

    check_space: CheckSpaceValue
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli


class StabilizerErrorEquivalenceResult(StrictModel):
    """Whether two Pauli errors differ by an element of the check space."""

    check_space: CheckSpaceValue
    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli
    difference: PhaseFreeQubitPauli
    equivalent_mod_stabilizers: bool

    @model_validator(mode="after")
    def require_equivalence_binding(self) -> Self:
        register = self.check_space.qubit_register
        if any(
            row.qubit_register != register
            for row in (self.left, self.right, self.difference)
        ):
            raise _validation_error(
                "equivalence_register", "errors and check space must share a register"
            )
        if type(self.equivalent_mod_stabilizers) is not bool:
            raise _validation_error(
                "equivalence_decision", "equivalence decision must be boolean"
            )
        return self


class CSSCheckSpaceRequest(StrictModel):
    """Binary X- and Z-check rows on one explicitly ordered register."""

    model_config = ConfigDict(populate_by_name=True)
    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    x_checks: tuple[
        Annotated[tuple[StrictInt, ...], Field(min_length=1, max_length=MAX_QUBITS)],
        ...,
    ] = Field(max_length=MAX_CHECK_ROWS)
    z_checks: tuple[
        Annotated[tuple[StrictInt, ...], Field(min_length=1, max_length=MAX_QUBITS)],
        ...,
    ] = Field(max_length=MAX_CHECK_ROWS)

    @model_validator(mode="after")
    def require_check_shapes(self) -> Self:
        width = len(self.qubit_register.qubit_ids)
        rows = (*self.x_checks, *self.z_checks)
        if any(len(row) != width for row in rows):
            raise _validation_error(
                "css_row_shape", "CSS rows must match register width"
            )
        if any(bit not in (0, 1) for row in rows for bit in row):
            raise _validation_error("css_bits", "CSS check entries must be bits")
        return self


class CSSNonOrthogonalWitness(StrictModel):
    """Register-bound input row indices the kernel found with inner product one.

    The kernel computes the GF(2) pairing once when it selects the obstruction;
    validation and transport stay structural. A consumer that relies on the
    nonorthogonality recomputes it against these retained rows on this
    retained register.
    """

    model_config = ConfigDict(populate_by_name=True)
    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    x_row: StrictInt = Field(ge=0, lt=MAX_CHECK_ROWS)
    z_row: StrictInt = Field(ge=0, lt=MAX_CHECK_ROWS)
    x_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    z_bits: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_QUBITS)
    dot_product: Literal[1] = 1

    @model_validator(mode="after")
    def require_register_bound_shape(self) -> Self:
        if not isinstance(self.qubit_register, QubitRegister):
            raise _validation_error(
                "css_witness_register", "CSS witness requires its source register"
            )
        width = len(self.qubit_register.qubit_ids)
        if len(self.x_bits) != width or len(self.z_bits) != width:
            raise _validation_error(
                "css_witness_shape", "CSS witness rows must span the retained register"
            )
        if any(bit not in (0, 1) for bit in (*self.x_bits, *self.z_bits)):
            raise _validation_error(
                "css_witness_bits", "CSS witness rows must be binary vectors"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        register: QubitRegister,
        x_row: int,
        z_row: int,
        x_bits: tuple[int, ...],
        z_bits: tuple[int, ...],
    ) -> Self:
        """Build a trusted obstruction without replaying its kernel pairing."""

        return cls.model_construct(
            qubit_register=register,
            x_row=x_row,
            z_row=z_row,
            x_bits=x_bits,
            z_bits=z_bits,
            dot_product=1,
        )


class CSSCheckSpaceValue(StrictModel):
    """A successful CSS decomposition of one binary stabilizer check space."""

    model_config = ConfigDict(populate_by_name=True)
    qubit_register: QubitRegister = Field(
        alias="register", serialization_alias="register"
    )
    x_check_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_CHECK_ROWS)
    z_check_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_CHECK_ROWS)
    check_space: CheckSpaceValue

    @model_validator(mode="after")
    def require_register_and_roles(self) -> Self:
        if any(
            row.qubit_register != self.qubit_register
            for row in (*self.x_check_basis, *self.z_check_basis)
        ):
            raise _validation_error(
                "css_register", "CSS rows must use the result register"
            )
        if any(any(row.z_bits) for row in self.x_check_basis) or any(
            any(row.x_bits) for row in self.z_check_basis
        ):
            raise _validation_error(
                "css_roles", "X and Z check bases must retain their roles"
            )
        if self.check_space.qubit_register != self.qubit_register:
            raise _validation_error(
                "css_check_space_register",
                "combined checks must use the result register",
            )
        return self


class CSSCheckSpaceResult(StrictModel):
    """A successful CSS check-space value or a nonorthogonality witness."""

    css_check_space: CSSCheckSpaceValue | None = None
    witness: CSSNonOrthogonalWitness | None = None

    @model_validator(mode="after")
    def require_branch(self) -> Self:
        if (self.css_check_space is None) == (self.witness is None):
            raise _validation_error(
                "css_branch", "CSS result requires exactly one outcome"
            )
        return self


class LogicalPauliFrame(StrictModel):
    """A paired phase-free symplectic basis of ``S-perp/S``."""

    check_space: CheckSpaceValue
    x_logical_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_QUBITS)
    z_logical_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_QUBITS)
    logical_qubits: int = Field(ge=0, le=MAX_QUBITS)

    @model_validator(mode="after")
    def require_quotient_frame(self) -> Self:
        # Structural deserialization only: the producing kernel establishes the
        # isotropic rank, S-perp membership, and canonical symplectic pairings
        # once via _from_kernel. Replaying GF(2) elimination here would redo
        # that mathematical work on every model_validate round trip.
        if not isinstance(self.check_space, CheckSpaceValue):
            raise _validation_error(
                "logical_frame_parent", "frame requires a check space"
            )
        if (
            len(self.x_logical_basis) != self.logical_qubits
            or len(self.z_logical_basis) != self.logical_qubits
        ):
            raise _validation_error(
                "logical_frame_dimension", "both logical families must have size k"
            )
        register = self.check_space.qubit_register
        n = len(register.qubit_ids)
        source_rows = self.check_space.basis
        if any(
            not isinstance(row, PhaseFreeQubitPauli)
            or row.qubit_register != register
            or len(row.x_bits) != n
            or len(row.z_bits) != n
            or any(bit not in (0, 1) for bit in (*row.x_bits, *row.z_bits))
            for row in source_rows
        ):
            raise _validation_error(
                "logical_frame_check_space",
                "check rows must be valid phase-free values on the source register",
            )
        logical_rows = (*self.x_logical_basis, *self.z_logical_basis)
        if any(
            not isinstance(row, PhaseFreeQubitPauli)
            or row.qubit_register != register
            or len(row.x_bits) != n
            or len(row.z_bits) != n
            or any(bit not in (0, 1) for bit in (*row.x_bits, *row.z_bits))
            for row in logical_rows
        ):
            raise _validation_error(
                "logical_frame_register",
                "logical representatives must be valid values on the source register",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        check_space: CheckSpaceValue,
        x_logical_basis: tuple[PhaseFreeQubitPauli, ...],
        z_logical_basis: tuple[PhaseFreeQubitPauli, ...],
    ) -> Self:
        return cls.model_construct(
            check_space=check_space,
            x_logical_basis=x_logical_basis,
            z_logical_basis=z_logical_basis,
            logical_qubits=len(x_logical_basis),
        )


class CSSLogicalPauliFrame(StrictModel):
    """Register-bound representatives of a symplectic basis of ``S-perp/S``."""

    css_check_space: CSSCheckSpaceValue
    x_logical_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_QUBITS)
    z_logical_basis: tuple[PhaseFreeQubitPauli, ...] = Field(max_length=MAX_QUBITS)
    logical_qubits: StrictInt = Field(ge=0, le=MAX_QUBITS)

    @model_validator(mode="after")
    def require_frame_shape(self) -> Self:
        register = self.css_check_space.qubit_register
        if (
            len(self.x_logical_basis) != self.logical_qubits
            or len(self.z_logical_basis) != self.logical_qubits
        ):
            raise _validation_error(
                "css_logical_frame_rank", "both CSS logical families must match k"
            )
        if any(
            row.qubit_register != register
            for row in (*self.x_logical_basis, *self.z_logical_basis)
        ):
            raise _validation_error(
                "css_logical_frame_register", "logical Paulis must use the CSS register"
            )
        if any(any(row.z_bits) for row in self.x_logical_basis) or any(
            any(row.x_bits) for row in self.z_logical_basis
        ):
            raise _validation_error(
                "css_logical_frame_roles",
                "logical X/Z representatives must retain their roles",
            )
        return self


class CSSDistanceResult(StrictModel):
    """Exact minimum X/Z logical weights, each with one minimum representative."""

    css_check_space: CSSCheckSpaceValue
    logical_qubits: StrictInt = Field(ge=0, le=MAX_QUBITS)
    x_distance: StrictInt | None = Field(default=None, ge=1, le=MAX_QUBITS)
    x_representative: PhaseFreeQubitPauli | None = None
    z_distance: StrictInt | None = Field(default=None, ge=1, le=MAX_QUBITS)
    z_representative: PhaseFreeQubitPauli | None = None

    @model_validator(mode="after")
    def require_exact_or_degenerate_branch(self) -> Self:
        if self.logical_qubits == 0:
            if any(
                value is not None
                for value in (
                    self.x_distance,
                    self.x_representative,
                    self.z_distance,
                    self.z_representative,
                )
            ):
                raise _validation_error(
                    "css_distance_no_logicals",
                    "a code with no logical qubits has no logical distances",
                )
            return self
        if any(
            value is None
            for value in (
                self.x_distance,
                self.x_representative,
                self.z_distance,
                self.z_representative,
            )
        ):
            raise _validation_error(
                "css_distance_missing_sector",
                "both logical sectors require an exact distance and representative",
            )
        register = self.css_check_space.qubit_register
        for distance, representative, role in (
            (self.x_distance, self.x_representative, "x"),
            (self.z_distance, self.z_representative, "z"),
        ):
            if distance is None or representative is None:
                raise _validation_error(
                    "css_distance_missing_sector",
                    "both logical sectors require an exact distance and representative",
                )
            if representative.qubit_register != register:
                raise _validation_error(
                    "css_distance_representative",
                    "minimum representative must have the declared weight on the source register",
                )
            if (role == "x" and any(representative.z_bits)) or (
                role == "z" and any(representative.x_bits)
            ):
                raise _validation_error(
                    "css_distance_role",
                    "X and Z representatives must retain their CSS coordinate roles",
                )
        return self


class StabilizerDistanceResult(StrictModel):
    """Exact minimum mixed-Pauli logical weight for one check space."""

    check_space: CheckSpaceValue
    logical_qubits: StrictInt = Field(ge=0, le=MAX_QUBITS)
    distance: StrictInt | None = Field(default=None, ge=1, le=MAX_QUBITS)
    representative: PhaseFreeQubitPauli | None = None

    @model_validator(mode="after")
    def require_exact_or_degenerate_branch(self) -> Self:
        if self.logical_qubits == 0:
            if self.distance is not None or self.representative is not None:
                raise _validation_error(
                    "distance_no_logicals", "a code with k=0 has no logical distance"
                )
            return self
        if self.distance is None or self.representative is None:
            raise _validation_error(
                "distance_missing_representative",
                "a positive-k code needs a minimum logical Pauli",
            )
        if self.representative.qubit_register != self.check_space.qubit_register:
            raise _validation_error(
                "distance_representative",
                "minimum representative must have the declared weight on the source register",
            )
        return self


class StabilizerErasureCorrectabilityRequest(StrictModel):
    """A supplied erasure subset on one register-bound stabilizer check space."""

    check_space: CheckSpaceValue
    erased_qubit_ids: tuple[QubitId, ...] = Field(max_length=MAX_QUBITS)

    @model_validator(mode="after")
    def require_erasure_subset(self) -> Self:
        register = self.check_space.qubit_register
        if len(set(self.erased_qubit_ids)) != len(self.erased_qubit_ids) or any(
            qubit_id not in register.qubit_ids for qubit_id in self.erased_qubit_ids
        ):
            raise _validation_error(
                "erasure_subset", "erased qubit IDs must be a unique register subset"
            )
        return self


class StabilizerErasureCorrectabilityResult(StrictModel):
    """Exact supported-logical criterion for one erasure subset."""

    source: StabilizerErasureCorrectabilityRequest
    supported_normalizer_dimension: StrictInt = Field(ge=0, le=2 * MAX_QUBITS)
    supported_stabilizer_dimension: StrictInt = Field(ge=0, le=MAX_CHECK_ROWS)
    supported_logical_dimension: StrictInt = Field(ge=0, le=2 * MAX_QUBITS)
    correctable: StrictBool
    witness: PhaseFreeQubitPauli | None = None

    @model_validator(mode="after")
    def require_exact_witness_branch(self) -> Self:
        if self.supported_logical_dimension != (
            self.supported_normalizer_dimension - self.supported_stabilizer_dimension
        ):
            raise _validation_error(
                "erasure_logical_dimension",
                "supported logical dimension must be the normalizer/stabilizer difference",
            )
        if self.correctable != (self.supported_logical_dimension == 0):
            raise _validation_error(
                "erasure_correctability",
                "correctability must match the exact dimension",
            )
        if self.correctable:
            if self.witness is not None:
                raise _validation_error(
                    "erasure_witness", "a correctable erasure has no logical witness"
                )
        else:
            if self.witness is None:
                raise _validation_error(
                    "erasure_witness",
                    "an uncorrectable erasure needs a supported logical witness",
                )
            register = self.source.check_space.qubit_register
            erased = set(self.source.erased_qubit_ids)
            if self.witness.qubit_register != register or any(
                (x or z) and qubit_id not in erased
                for qubit_id, x, z in zip(
                    register.qubit_ids,
                    self.witness.x_bits,
                    self.witness.z_bits,
                    strict=True,
                )
            ):
                raise _validation_error(
                    "erasure_witness",
                    "logical witness must be supported inside the erasure",
                )
        return self


__all__ = [
    "MAX_CHECK_ROWS",
    "MAX_QUBITS",
    "MAX_QUBIT_LABEL_LENGTH",
    "BinaryPauliRow",
    "CSSCheckSpaceRequest",
    "CSSCheckSpaceResult",
    "CSSCheckSpaceValue",
    "CSSDistanceResult",
    "CSSLogicalPauliFrame",
    "CSSNonOrthogonalWitness",
    "CanonicalCheckRow",
    "CheckSpaceCanonicalizeRequest",
    "CheckSpaceCanonicalizeResult",
    "CheckSpaceStatus",
    "CheckSpaceValue",
    "ExactQubitPauli",
    "LogicalPauliFrame",
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
    "StabilizerErasureCorrectabilityRequest",
    "StabilizerErasureCorrectabilityResult",
    "StabilizerErrorEquivalenceRequest",
    "StabilizerErrorEquivalenceResult",
    "StabilizerSyndromeRequest",
    "StabilizerSyndromeResult",
]
