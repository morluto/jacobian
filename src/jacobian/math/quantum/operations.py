"""Native exact binary-qubit stabilizer and logical-space operations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product
from typing import Literal, NoReturn, cast

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBIT_LABEL_LENGTH,
    MAX_QUBITS,
    BinaryPauliRow,
    CanonicalCheckRow,
    CheckSpaceCanonicalizeResult,
    CheckSpaceValue,
    CSSCheckSpaceResult,
    CSSCheckSpaceValue,
    CSSDistanceResult,
    CSSLogicalPauliFrame,
    CSSNonOrthogonalWitness,
    ExactQubitPauli,
    ExactStabilizerGroup,
    ExactStabilizerGroupRequest,
    LogicalPauliFrame,
    NonCommutingWitness,
    NormalizerResult,
    PauliFamilyCommutationRequest,
    PauliFamilyCommutationResult,
    PauliFamilyEntry,
    PauliInverseResult,
    PauliPairingResult,
    PauliProductResult,
    PauliToLabelsResult,
    PhaseFreeQubitPauli,
    QubitRegister,
    StabilizerCodeValue,
    StabilizerDistanceResult,
    StabilizerErrorEquivalenceResult,
    StabilizerSyndromeResult,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _symplectic_pairing(
    first: Sequence[int], second: Sequence[int], qubits: int
) -> int:
    """Binary symplectic pairing ``x.z' + z.x' mod 2`` of two ``(x|z)`` rows."""

    total = 0
    for q in range(qubits):
        total += first[q] * second[qubits + q] + first[qubits + q] * second[q]
    return total % 2


def _gf2_rref(rows: list[list[int]], width: int) -> tuple[list[list[int]], list[int]]:
    """Reduced row-echelon form over GF(2) with pivot columns."""

    basis = [row[:] for row in rows]
    pivots: list[int] = []
    target = 0
    for column in range(width):
        pivot = next(
            (r for r in range(target, len(basis)) if basis[r][column] == 1), None
        )
        if pivot is None:
            continue
        basis[target], basis[pivot] = basis[pivot], basis[target]
        for r in range(len(basis)):
            if r != target and basis[r][column] == 1:
                basis[r] = [
                    (a + b) % 2 for a, b in zip(basis[r], basis[target], strict=True)
                ]
        pivots.append(column)
        target += 1
    return basis[:target], pivots


def _admit_canonicalize(
    qubit_ids: object, generators: object
) -> tuple[tuple[str, ...], tuple[BinaryPauliRow, ...]]:
    """Enforce the shared envelope for native and catalog calls."""

    if not isinstance(qubit_ids, (tuple, list)) or not qubit_ids:
        _reject(
            "qubit_ids",
            "stabilizer.check_space.register_not_a_qubit_family",
            "check-space register must be a nonempty qubit family",
        )
    if not isinstance(generators, (tuple, list)) or not generators:
        _reject(
            "generators",
            "stabilizer.check_space.generators_not_a_row_family",
            "check-space generators must be a nonempty Pauli row family",
        )
    validated_rows: list[BinaryPauliRow] = []
    for row in generators:
        if not isinstance(row, BinaryPauliRow):
            _reject(
                "generators",
                "stabilizer.check_space.generator_not_a_pauli_row",
                "every check-space generator must be a phase-free Pauli row",
            )
        validated_rows.append(row)
    rows = tuple(validated_rows)
    ids = tuple(qubit_ids)
    if any(
        not isinstance(qubit_id, str)
        or not qubit_id
        or len(qubit_id) > MAX_QUBIT_LABEL_LENGTH
        or any(0xD800 <= ord(character) <= 0xDFFF for character in qubit_id)
        for qubit_id in ids
    ):
        _reject(
            "qubit_ids",
            "stabilizer.check_space.qubit_id_not_strict_string",
            "qubit IDs must be nonempty Unicode scalar strings",
        )
    if len(set(ids)) != len(ids):
        _reject(
            "qubit_ids",
            "stabilizer.check_space.qubit_ids_not_unique",
            "qubit IDs must be unique",
        )
    row_ids = tuple(row.row_id for row in rows)
    if any(
        not isinstance(row_id, str)
        or not row_id
        or len(row_id) > MAX_QUBIT_LABEL_LENGTH
        or any(0xD800 <= ord(character) <= 0xDFFF for character in row_id)
        for row_id in row_ids
    ):
        _reject(
            "generators",
            "stabilizer.check_space.generator_row_id_not_strict_string",
            "generator row IDs must be nonempty Unicode scalar strings",
        )
    if tuple(sorted(row_ids)) != row_ids or len(set(row_ids)) != len(row_ids):
        _reject(
            "generators",
            "stabilizer.check_space.generator_row_ids",
            "generator row IDs must be unique and strictly ordered",
        )
    width = len(ids)
    if width > MAX_QUBITS:
        raise OperationResourceAdmissionError(
            location=("qubit_ids",),
            code="stabilizer.check_space.register_over_envelope",
            message=f"qubit register exceeds the {MAX_QUBITS}-qubit envelope",
        )
    if len(rows) > MAX_CHECK_ROWS:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="stabilizer.check_space.rows_over_envelope",
            message=f"generator rows exceed the {MAX_CHECK_ROWS}-row envelope",
        )
    for row in rows:
        if len(row.x_bits) != width or len(row.z_bits) != width:
            _reject(
                "generators",
                "stabilizer.check_space.register_binding_mismatch",
                "every generator bit row must match the register length",
            )
    return ids, rows


def _admit_register(register: object, location: str) -> QubitRegister:
    if not isinstance(register, QubitRegister):
        _reject(
            location, "quantum.pauli.invalid_register", "Pauli register is malformed"
        )
    ids = getattr(register, "qubit_ids", None)
    if (
        not isinstance(ids, tuple)
        or not 1 <= len(ids) <= MAX_QUBITS
        or any(
            type(value) is not str
            or not value
            or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            for value in ids
        )
        or len(set(ids)) != len(ids)
    ):
        _reject(
            location, "quantum.pauli.invalid_register", "Pauli register is malformed"
        )
    if any(len(value) > MAX_QUBIT_LABEL_LENGTH for value in ids):
        _reject(
            location,
            "quantum.pauli.invalid_register",
            "Pauli register labels exceed the envelope",
        )
    return register


def _admit_phase_free(value: object, location: str = "pauli") -> PhaseFreeQubitPauli:
    if not isinstance(value, PhaseFreeQubitPauli):
        _reject(
            location,
            "quantum.pauli.not_a_pauli",
            "Pauli operation requires typed exact Pauli values",
        )
    register = _admit_register(getattr(value, "qubit_register", None), location)
    x_bits = getattr(value, "x_bits", None)
    z_bits = getattr(value, "z_bits", None)
    width = len(register.qubit_ids)
    if (
        not isinstance(x_bits, tuple)
        or not isinstance(z_bits, tuple)
        or len(x_bits) != width
        or len(z_bits) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in (*x_bits, *z_bits))
    ):
        _reject(
            location,
            "quantum.pauli.invalid_bits",
            "Pauli coordinates must be binary rows on their register",
        )
    return value


def _admit_exact(value: object, location: str = "pauli") -> ExactQubitPauli:
    if not isinstance(value, ExactQubitPauli):
        _reject(
            location,
            "quantum.pauli.not_a_exact_pauli",
            "operation requires an exact phase-lifted Pauli",
        )
    _admit_phase_free(getattr(value, "phase_free", None), location)
    phase = getattr(value, "phase", None)
    if type(phase) is not int or not 0 <= phase <= 3:
        _reject(
            location,
            "quantum.pauli.invalid_phase",
            "Pauli phase must be an integer modulo four",
        )
    return value


def _admit_pauli_pair(left: PhaseFreeQubitPauli, right: PhaseFreeQubitPauli) -> None:
    _admit_phase_free(left, "left")
    _admit_phase_free(right, "right")
    if left.qubit_register != right.qubit_register:
        _reject(
            "right",
            "quantum.pauli.register_mismatch",
            "Paulis must use the identical ordered qubit register",
        )


def pauli_pairing(
    left: PhaseFreeQubitPauli, right: PhaseFreeQubitPauli
) -> PauliPairingResult:
    _admit_pauli_pair(left, right)
    pairing = (
        sum(
            x * z2 + z * x2
            for x, z, x2, z2 in zip(
                left.x_bits, left.z_bits, right.x_bits, right.z_bits, strict=True
            )
        )
        % 2
    )
    return PauliPairingResult(
        left=left, right=right, pairing=pairing, commute=pairing == 0
    )


def pauli_from_labels(
    register: QubitRegister,
    labels: tuple[Literal["I", "X", "Y", "Z"], ...] | list[Literal["I", "X", "Y", "Z"]],
    phase: int,
) -> ExactQubitPauli:
    """Lift local I/X/Y/Z labels to ``i^phase X^x Z^z`` exactly.

    In this convention each local Y contributes one factor of ``i`` because
    ``Y = i X Z``. The input phase is the scalar multiplying the labelled
    tensor product, so it is added to the number of Y entries modulo four.
    """
    admitted_register = _admit_register(register, "register")
    if not isinstance(labels, (tuple, list)) or not 1 <= len(labels) <= MAX_QUBITS:
        _reject(
            "labels",
            "quantum.pauli.invalid_labels",
            "Pauli labels must be a nonempty I/X/Y/Z family",
        )
    label_tuple = tuple(labels)
    if len(label_tuple) != len(admitted_register.qubit_ids) or any(
        type(label) is not str or label not in ("I", "X", "Y", "Z")
        for label in label_tuple
    ):
        _reject(
            "labels",
            "quantum.pauli.invalid_labels",
            "Pauli labels must cover the register with I/X/Y/Z symbols",
        )
    if type(phase) is not int or not 0 <= phase <= 3:
        _reject(
            "phase",
            "quantum.pauli.invalid_phase",
            "Pauli phase must be an integer modulo four",
        )
    x_bits: list[int] = []
    z_bits: list[int] = []
    y_count = 0
    for label in label_tuple:
        x, z = {
            "I": (0, 0),
            "X": (1, 0),
            "Y": (1, 1),
            "Z": (0, 1),
        }[label]
        x_bits.append(x)
        z_bits.append(z)
        y_count += label == "Y"
    phase_free = PhaseFreeQubitPauli(
        register=admitted_register, x_bits=tuple(x_bits), z_bits=tuple(z_bits)
    )
    return ExactQubitPauli(phase_free=phase_free, phase=(phase + y_count) % 4)


def pauli_to_labels(pauli: ExactQubitPauli) -> PauliToLabelsResult:
    """Return local labels and the unique scalar phase relative to them."""
    admitted = _admit_exact(pauli, "pauli")
    raw_labels = tuple(
        "Y" if x and z else "X" if x else "Z" if z else "I"
        for x, z in zip(
            admitted.phase_free.x_bits, admitted.phase_free.z_bits, strict=True
        )
    )
    labels = cast(tuple[Literal["I", "X", "Y", "Z"], ...], raw_labels)
    y_count = sum(label == "Y" for label in labels)
    return PauliToLabelsResult(
        source=admitted, labels=labels, phase=(admitted.phase - y_count) % 4
    )


def pauli_family_commutation_matrix(
    family: tuple[PauliFamilyEntry, ...] | list[PauliFamilyEntry],
) -> PauliFamilyCommutationResult:
    """Return all pairwise symplectic pairings on the retained named axis."""
    if not isinstance(family, (tuple, list)) or not 1 <= len(family) <= MAX_CHECK_ROWS:
        _reject(
            "family",
            "quantum.pauli.family.invalid_size",
            "Pauli family is outside its admitted size",
        )
    entries = []
    for index, entry in enumerate(family):
        pauli_id = getattr(entry, "pauli_id", None)
        if (
            type(pauli_id) is not str
            or not pauli_id
            or len(pauli_id) > MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(character) <= 0xDFFF for character in pauli_id)
        ):
            _reject(
                "family",
                "quantum.pauli.family.invalid_id",
                "Pauli IDs must be bounded Unicode scalar strings",
            )
        pauli = _admit_phase_free(getattr(entry, "pauli", None), f"family[{index}]")
        entries.append((pauli_id, pauli))
    if len({pauli_id for pauli_id, _ in entries}) != len(entries):
        _reject(
            "family", "quantum.pauli.family.duplicate_id", "Pauli IDs must be unique"
        )
    register = entries[0][1].qubit_register
    if any(pauli.qubit_register != register for _, pauli in entries):
        _reject(
            "family",
            "quantum.pauli.family.register_mismatch",
            "every Pauli must use the identical ordered register",
        )
    count = len(entries)
    width = len(register.qubit_ids)
    work = count * count * width
    # JSON escapes a Unicode code point to at most six ASCII bytes. Each Pauli
    # repeats its register carrier, so count that parent once per family row.
    register_bound = sum(6 * len(qubit_id) + 4 for qubit_id in register.qubit_ids)
    output_bound = (
        count
        * (
            register_bound
            + max(6 * len(identifier) + 4 for identifier, _ in entries)
            + 4 * width
            + 256
        )
        + 3 * count * count
        + 256
    )
    if work > 131_072 or output_bound > 750_000:
        raise OperationResourceAdmissionError(
            location=("family",),
            code="quantum.pauli.family.commutation_matrix.over_envelope",
            message="complete commutation matrix work or result size exceeds its envelope",
        )
    matrix = tuple(
        tuple(
            _symplectic_pairing(
                (*first.x_bits, *first.z_bits), (*second.x_bits, *second.z_bits), width
            )
            for _, second in entries
        )
        for _, first in entries
    )
    source = PauliFamilyCommutationRequest(
        family=tuple(
            PauliFamilyEntry(pauli_id=pauli_id, pauli=pauli)
            for pauli_id, pauli in entries
        )
    )
    return PauliFamilyCommutationResult.model_construct(
        source=source, commutation_matrix=matrix
    )


def pauli_multiply(left: ExactQubitPauli, right: ExactQubitPauli) -> PauliProductResult:
    _admit_exact(left, "left")
    _admit_exact(right, "right")
    _admit_pauli_pair(left.phase_free, right.phase_free)
    product_value = _product_pauli_after_admission(left, right)
    return PauliProductResult(left=left, right=right, product=product_value)


def _product_pauli_after_admission(
    left: ExactQubitPauli, right: ExactQubitPauli
) -> ExactQubitPauli:
    """Multiply same-register Paulis after a caller has admitted both values."""
    phase = (
        left.phase
        + right.phase
        + 2
        * sum(
            z * x
            for z, x in zip(
                left.phase_free.z_bits, right.phase_free.x_bits, strict=True
            )
        )
    ) % 4
    product_value = ExactQubitPauli.model_construct(
        phase_free=PhaseFreeQubitPauli.model_construct(
            qubit_register=left.register,
            x_bits=tuple(
                (x + y) % 2
                for x, y in zip(
                    left.phase_free.x_bits, right.phase_free.x_bits, strict=True
                )
            ),
            z_bits=tuple(
                (z + w) % 2
                for z, w in zip(
                    left.phase_free.z_bits, right.phase_free.z_bits, strict=True
                )
            ),
        ),
        phase=phase,
    )
    return product_value


def pauli_inverse(value: ExactQubitPauli) -> PauliInverseResult:
    _admit_exact(value)
    phase = (
        -value.phase
        - 2
        * sum(
            x * z
            for x, z in zip(
                value.phase_free.x_bits, value.phase_free.z_bits, strict=True
            )
        )
    ) % 4
    inverse = ExactQubitPauli(
        phase_free=value.phase_free,
        phase=phase,
    )
    return PauliInverseResult(source=value, inverse=inverse)


def stabilizer_group_from_generators(
    request: ExactStabilizerGroupRequest,
) -> ExactStabilizerGroup:
    """Validate exact stabilizer generators and retain an independent basis.

    A generator is Hermitian precisely when ``phase + x.z`` is even under
    ``i^phase X^x Z^z``. Pairwise commutation is checked before elimination.
    During incremental GF(2) reduction, each dependent row is multiplied by
    the selected Hermitian generators that cancel its vector. A zero vector
    must then have phase zero: phase two would put ``-I`` in the group.
    """
    if not isinstance(request, ExactStabilizerGroupRequest):
        _reject(
            "request",
            "quantum.stabilizer.exact_group.invalid_request",
            "request must contain a register and exact Pauli generators",
        )
    register = _admit_register(getattr(request, "qubit_register", None), "register")
    values = getattr(request, "generators", None)
    if not isinstance(values, tuple) or len(values) > MAX_CHECK_ROWS:
        _reject(
            "generators",
            "quantum.stabilizer.exact_group.invalid_size",
            "exact generator family exceeds its admitted row count",
        )

    # Admission is performed once before arithmetic. Charge the complete
    # pairwise scan and the worst-case elimination pass, plus bounded pivot
    # ordering overhead, before doing any generator arithmetic.
    width = len(register.qubit_ids)
    count = len(values)
    pair_count = count * (count - 1) // 2
    work_bound = (
        2 * pair_count * width
        + 8 * count * min(count, width) * width
        + width * (MAX_QUBIT_LABEL_LENGTH + 4)
        + 2 * count * width * (MAX_QUBIT_LABEL_LENGTH + 4)
        + count * width
    )
    if work_bound > 1_000_000:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="quantum.stabilizer.exact_group.over_envelope",
            message="exact generator validation exceeds its work envelope",
        )

    generators: list[ExactQubitPauli] = []
    for index, value in enumerate(values):
        pauli = _admit_exact(value, f"generators[{index}]")
        if pauli.register != register:
            _reject(
                f"generators[{index}]",
                "quantum.stabilizer.exact_group.register_mismatch",
                "every exact generator must use the identical ordered register",
            )
        if (
            pauli.phase
            + sum(
                x * z
                for x, z in zip(
                    pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True
                )
            )
        ) % 2:
            _reject(
                f"generators[{index}]",
                "quantum.stabilizer.exact_group.non_hermitian_generator",
                "stabilizer generators must be Hermitian Paulis",
            )
        generators.append(pauli)

    for i, first in enumerate(generators):
        for second in generators[i + 1 :]:
            if _symplectic_pairing(
                (*first.phase_free.x_bits, *first.phase_free.z_bits),
                (*second.phase_free.x_bits, *second.phase_free.z_bits),
                width,
            ):
                _reject(
                    "generators",
                    "quantum.stabilizer.exact_group.noncommuting_generators",
                    "stabilizer generators must commute pairwise",
                )

    # Each echelon row is an exact product of selected input generators.
    # Its leading coordinate is unique; the phase is carried through each
    # multiplication, so a dependent row detects the actual scalar relation.
    echelon: dict[int, ExactQubitPauli] = {}
    independent: list[ExactQubitPauli] = []
    for generator in generators:
        reduced = generator
        vector = (*reduced.phase_free.x_bits, *reduced.phase_free.z_bits)
        for existing_pivot in sorted(echelon):
            if vector[existing_pivot]:
                row = echelon[existing_pivot]
                product_pauli = _product_pauli_after_admission(reduced, row)
                reduced = product_pauli
                vector = (*reduced.phase_free.x_bits, *reduced.phase_free.z_bits)
        pivot = next((column for column, bit in enumerate(vector) if bit), None)
        if pivot is None:
            if reduced.phase != 0:
                _reject(
                    "generators",
                    "quantum.stabilizer.exact_group.forbidden_scalar",
                    "a generator dependency produces a nonidentity scalar",
                )
            continue
        echelon[pivot] = reduced
        independent.append(generator)

    return ExactStabilizerGroup(register=register, generators=tuple(independent))


def _admit_stabilizer_code_request(
    group: object, eigenvalues: object
) -> tuple[QubitRegister, tuple[ExactQubitPauli, ...], tuple[int, ...]]:
    """Validate a code's group, character, axes, and complete admitted work."""
    if not isinstance(group, ExactStabilizerGroup):
        _reject(
            "group",
            "quantum.stabilizer.code.invalid_group",
            "code construction requires an exact stabilizer group",
        )
    register = _admit_register(getattr(group, "qubit_register", None), "group")
    generators = getattr(group, "generators", None)
    if not isinstance(generators, tuple) or len(generators) > MAX_CHECK_ROWS:
        _reject(
            "group",
            "quantum.stabilizer.code.invalid_group",
            "group generators exceed the exact stabilizer envelope",
        )
    if (
        not isinstance(eigenvalues, tuple)
        or len(eigenvalues) != len(generators)
        or any(type(value) is not int or value not in (-1, 1) for value in eigenvalues)
    ):
        _reject(
            "generator_eigenvalues",
            "quantum.stabilizer.code.invalid_character",
            "one strict +1 or -1 eigenvalue is required per independent generator",
        )
    width = len(register.qubit_ids)
    count = len(generators)
    pair_count = count * (count - 1) // 2
    work_bound = (
        2 * pair_count * width
        + 6 * count * width * width
        + width * (MAX_QUBIT_LABEL_LENGTH + 4)
        + 3 * count * width * (MAX_QUBIT_LABEL_LENGTH + 4)
        + count * width
    )
    # A scalar label may occupy 12 characters when JSON escapes a
    # supplementary-plane code point as a surrogate pair. Each generator
    # repeats the complete register inside its phase-free Pauli value.
    output_bound = (
        (count + 1) * (width * (12 * MAX_QUBIT_LABEL_LENGTH + 3) + 128)
        + count * (4 * width + 128)
        + 512
    )
    if work_bound > 1_200_000 or output_bound > 2_000_000:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="quantum.stabilizer.code.over_envelope",
            message="canonical code construction exceeds its work or output envelope",
        )

    for index, pauli in enumerate(generators):
        _admit_exact(pauli, f"group.generators[{index}]")
        if pauli.register != register:
            _reject(
                f"group.generators[{index}]",
                "quantum.stabilizer.code.register_mismatch",
                "every exact generator must use the identical ordered register",
            )
        if (
            pauli.phase
            + sum(
                x * z
                for x, z in zip(
                    pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True
                )
            )
        ) % 2:
            _reject(
                f"group.generators[{index}]",
                "quantum.stabilizer.code.non_hermitian_generator",
                "stabilizer generators must be Hermitian Paulis",
            )
    for left_index, left in enumerate(generators):
        for right in generators[left_index + 1 :]:
            if _symplectic_pairing(
                (*left.phase_free.x_bits, *left.phase_free.z_bits),
                (*right.phase_free.x_bits, *right.phase_free.z_bits),
                width,
            ):
                _reject(
                    "group.generators",
                    "quantum.stabilizer.code.noncommuting_generators",
                    "stabilizer generators must commute pairwise",
                )
    return register, generators, eigenvalues


def stabilizer_code_compute(
    group: ExactStabilizerGroup, generator_eigenvalues: tuple[int, ...]
) -> StabilizerCodeValue:
    """Canonicalize an exact group together with its one-dimensional character.

    Each input generator ``g`` with eigenvalue ``lambda`` is replaced by
    ``lambda*g``. The returned operators therefore all stabilize the selected
    space with eigenvalue +1. Row operations carry their exact Pauli products,
    so RREF canonicalizes the subgroup without losing scalar signs.
    """
    register, generators, eigenvalues = _admit_stabilizer_code_request(
        group, generator_eigenvalues
    )
    width = len(register.qubit_ids)

    # Replace each generator g with chi(g) g. The resulting operators have
    # eigenvalue +1 on precisely the selected joint eigenspace.
    rows: list[tuple[list[int], ExactQubitPauli]] = []
    for pauli, eigenvalue in zip(generators, eigenvalues, strict=True):
        positive_generator = ExactQubitPauli.model_construct(
            phase_free=pauli.phase_free,
            phase=(pauli.phase + (2 if eigenvalue == -1 else 0)) % 4,
        )
        flat = [*pauli.phase_free.x_bits, *pauli.phase_free.z_bits]
        rows.append((flat, positive_generator))
    target = 0
    for column in range(2 * width):
        pivot = next(
            (index for index in range(target, len(rows)) if rows[index][0][column]),
            None,
        )
        if pivot is None:
            continue
        rows[target], rows[pivot] = rows[pivot], rows[target]
        pivot_vector, pivot_pauli = rows[target]
        for index in range(len(rows)):
            if index == target or not rows[index][0][column]:
                continue
            vector, pauli = rows[index]
            product_pauli = _product_pauli_after_admission(pauli, pivot_pauli)
            rows[index] = (
                [
                    (left + right) % 2
                    for left, right in zip(vector, pivot_vector, strict=True)
                ],
                product_pauli,
            )
        target += 1

    if target != len(generators):
        _reject(
            "group.generators",
            "quantum.stabilizer.code.group_not_independent",
            "an exact stabilizer group value must carry an independent generator family",
        )

    canonical_group = ExactStabilizerGroup(
        register=register,
        generators=tuple(row[1] for row in rows[:target]),
    )
    return StabilizerCodeValue(group=canonical_group)


def _gf2_nullspace(rows: list[list[int]], width: int) -> tuple[tuple[int, ...], ...]:
    rref, pivots = _gf2_rref(rows, width)
    pivot_set = set(pivots)
    free = [column for column in range(width) if column not in pivot_set]
    vectors: list[tuple[int, ...]] = []
    for free_column in free:
        vector = [0] * width
        vector[free_column] = 1
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = rref[row_index][free_column]
        vectors.append(tuple(vector))
    return tuple(vectors)


def stabilizer_normalizer(check_space: CheckSpaceValue) -> NormalizerResult:
    if not isinstance(check_space, CheckSpaceValue):
        _reject(
            "check_space",
            "quantum.stabilizer.not_a_check_space",
            "normalizer requires a typed register-bound check space",
        )
    register = _admit_register(
        getattr(check_space, "qubit_register", None), "check_space"
    )
    basis_value = getattr(check_space, "basis", None)
    if not isinstance(basis_value, tuple) or len(basis_value) > MAX_CHECK_ROWS:
        _reject(
            "check_space",
            "quantum.stabilizer.invalid_basis",
            "check-space basis is malformed",
        )
    basis = tuple(basis_value)
    for row in basis:
        _admit_phase_free(row, "check_space")
        if row.qubit_register != register:
            _reject(
                "check_space",
                "quantum.stabilizer.parent_mismatch",
                "all check rows must use the declared register",
            )
    for i, left in enumerate(basis):
        for right in basis[i + 1 :]:
            if pauli_pairing(left, right).pairing:
                _reject(
                    "check_space",
                    "quantum.stabilizer.not_isotropic",
                    "check space must be symplectically isotropic",
                )
    # Canonicalize the supplied row space before computing its orthogonal.
    flat = [[*row.x_bits, *row.z_bits] for row in basis]
    canonical_rows, _ = _gf2_rref(flat, 2 * len(register.qubit_ids))
    canonical_basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=tuple(row[: len(register.qubit_ids)]),
            z_bits=tuple(row[len(register.qubit_ids) :]),
        )
        for row in canonical_rows
    )
    constraints = [[*row.z_bits, *row.x_bits] for row in canonical_basis]
    orthogonal_flat = _gf2_nullspace(constraints, 2 * len(register.qubit_ids))
    orthogonal_basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=tuple(row[: len(register.qubit_ids)]),
            z_bits=tuple(row[len(register.qubit_ids) :]),
        )
        for row in orthogonal_flat
    )
    return NormalizerResult._from_kernel(
        check_space=CheckSpaceValue(register=register, basis=canonical_basis),
        orthogonal_basis=orthogonal_basis,
    )


def stabilizer_logical_frame(check_space: CheckSpaceValue) -> LogicalPauliFrame:
    """Choose a deterministic symplectic frame for the phase-free quotient."""
    normalizer = stabilizer_normalizer(check_space)
    canonical_space = normalizer.check_space
    register = canonical_space.qubit_register
    n = len(register.qubit_ids)
    stabilizer_rows = [[*row.x_bits, *row.z_bits] for row in canonical_space.basis]
    # Extend S to a basis of S-perp, making the selected complement a basis
    # for the quotient rather than enumerating its 4^k elements.
    span_rows = [row[:] for row in stabilizer_rows]
    span_rank = len(span_rows)
    complement: list[list[int]] = []
    for row in normalizer.orthogonal_basis:
        flat = [*row.x_bits, *row.z_bits]
        expanded, _ = _gf2_rref([*span_rows, flat], 2 * n)
        if len(expanded) > span_rank:
            complement.append(flat)
            span_rows = expanded
            span_rank += 1

    def pairing(left: list[int], right: list[int]) -> int:
        return (
            sum(left[i] * right[n + i] + left[n + i] * right[i] for i in range(n)) % 2
        )

    def add_scaled(vector: list[int], source: list[int], scale: int) -> list[int]:
        if not scale:
            return vector
        return [(a + b) % 2 for a, b in zip(vector, source, strict=True)]

    x_rows: list[list[int]] = []
    z_rows: list[list[int]] = []
    while complement:
        x_row = complement.pop(0)
        partner_index = next(
            (index for index, row in enumerate(complement) if pairing(x_row, row)),
            None,
        )
        if partner_index is None:
            _reject(
                "check_space",
                "quantum.stabilizer.logical_frame.degenerate_quotient",
                "the induced logical symplectic form must be nondegenerate",
            )
        z_row = complement.pop(partner_index)
        # Orthogonalize every remaining quotient vector against this pair.
        complement = [
            add_scaled(
                add_scaled(row, x_row, pairing(row, z_row)),
                z_row,
                pairing(row, x_row),
            )
            for row in complement
        ]
        x_rows.append(x_row)
        z_rows.append(z_row)

    logical_qubits = len(x_rows)
    if 2 * logical_qubits != normalizer.logical_dimension:
        _reject(
            "check_space",
            "quantum.stabilizer.logical_frame.dimension_mismatch",
            "the quotient frame dimension must equal dim(S-perp/S)",
        )
    x_values = tuple(
        PhaseFreeQubitPauli(
            register=register, x_bits=tuple(row[:n]), z_bits=tuple(row[n:])
        )
        for row in x_rows
    )
    z_values = tuple(
        PhaseFreeQubitPauli(
            register=register, x_bits=tuple(row[:n]), z_bits=tuple(row[n:])
        )
        for row in z_rows
    )
    if any(
        pairing(left, right) != int(i == j)
        for i, left in enumerate(x_rows)
        for j, right in enumerate(z_rows)
    ) or any(
        pairing(left, right)
        for family in (x_rows, z_rows)
        for i, left in enumerate(family)
        for right in family[i + 1 :]
    ):
        _reject(
            "check_space",
            "quantum.stabilizer.logical_frame.pairing_failure",
            "logical representatives must form a symplectic frame",
        )
    return LogicalPauliFrame._from_kernel(
        check_space=canonical_space,
        x_logical_basis=x_values,
        z_logical_basis=z_values,
    )


def canonicalize_check_space(
    qubit_ids: tuple[str, ...] | list[str],
    generators: tuple[BinaryPauliRow, ...] | list[BinaryPauliRow],
) -> CheckSpaceCanonicalizeResult:
    """Canonicalize a binary check matrix to RREF or witness non-isotropy.

    Phase-free Paulis commute iff their symplectic pairing is zero, so the
    kernel first replays the complete pairwise pairing table: the first
    pair with pairing one becomes the explicit ``NOT_ISOTROPIC`` witness.
    Otherwise GF(2) elimination yields the canonical RREF basis, rank, and
    ``ISOTROPIC_CHECK_SPACE`` decision. The RREF depends only on the GF(2)
    row space; zero rows contribute no pivot and reduce the rank.
    """

    ids, rows = _admit_canonicalize(qubit_ids, generators)
    qubits = len(ids)
    flat_rows = tuple((*row.x_bits, *row.z_bits) for row in rows)
    for i in range(len(flat_rows)):
        for j in range(i + 1, len(flat_rows)):
            if _symplectic_pairing(flat_rows[i], flat_rows[j], qubits):
                first, second = rows[i].row_id, rows[j].row_id
                ordered = (min(first, second), max(first, second))
                return CheckSpaceCanonicalizeResult._from_kernel(
                    status="NOT_ISOTROPIC",
                    qubit_ids=ids,
                    rank=0,
                    basis=(),
                    witness=NonCommutingWitness(
                        first_row_id=ordered[0],
                        second_row_id=ordered[1],
                        pairing=1,
                    ),
                )
    basis_rows, pivots = _gf2_rref([list(row) for row in flat_rows], 2 * qubits)
    basis = tuple(
        CanonicalCheckRow(
            pivot=pivot,
            x_bits=tuple(row[:qubits]),
            z_bits=tuple(row[qubits:]),
        )
        for row, pivot in zip(basis_rows, pivots, strict=True)
    )
    return CheckSpaceCanonicalizeResult._from_kernel(
        status="ISOTROPIC_CHECK_SPACE",
        qubit_ids=ids,
        rank=len(basis),
        basis=basis,
        witness=None,
    )


def stabilizer_syndrome(
    check_space: CheckSpaceValue, error: PhaseFreeQubitPauli
) -> StabilizerSyndromeResult:
    """Return the error's exact syndrome on the canonical RREF check axis."""

    if not isinstance(check_space, CheckSpaceValue):
        _reject(
            "check_space",
            "quantum.stabilizer.not_a_check_space",
            "syndrome requires a typed register-bound check space",
        )
    register = _admit_register(
        getattr(check_space, "qubit_register", None), "check_space"
    )
    basis_value = getattr(check_space, "basis", None)
    if not isinstance(basis_value, tuple) or len(basis_value) > MAX_CHECK_ROWS:
        _reject(
            "check_space",
            "quantum.stabilizer.invalid_basis",
            "check-space basis is malformed",
        )
    basis = tuple(basis_value)
    flat: list[list[int]] = []
    for row in basis:
        _admit_phase_free(row, "check_space")
        if row.qubit_register != register:
            _reject(
                "check_space",
                "quantum.stabilizer.parent_mismatch",
                "all check rows must use the declared register",
            )
        flat.append([*row.x_bits, *row.z_bits])
    for i, left in enumerate(flat):
        for right in flat[i + 1 :]:
            if _symplectic_pairing(left, right, len(register.qubit_ids)):
                _reject(
                    "check_space",
                    "quantum.stabilizer.not_isotropic",
                    "syndrome check space must be symplectically isotropic",
                )
    if not isinstance(error, PhaseFreeQubitPauli):
        _reject(
            "error",
            "quantum.pauli.not_a_pauli",
            "error must be a typed phase-free Pauli",
        )
    _admit_phase_free(error, "error")
    if error.qubit_register != register:
        _reject(
            "error",
            "quantum.pauli.register_mismatch",
            "error and check space must use the identical ordered register",
        )
    canonical, _ = _gf2_rref(flat, 2 * len(register.qubit_ids))
    canonical_basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=tuple(row[: len(register.qubit_ids)]),
            z_bits=tuple(row[len(register.qubit_ids) :]),
        )
        for row in canonical
    )
    error_flat = [*error.x_bits, *error.z_bits]
    syndrome = tuple(
        _symplectic_pairing(row, error_flat, len(register.qubit_ids))
        for row in canonical
    )
    return StabilizerSyndromeResult._from_kernel(
        check_space=CheckSpaceValue(register=register, basis=canonical_basis),
        error=error,
        syndrome=syndrome,
    )


def stabilizer_error_equivalence(
    check_space: CheckSpaceValue,
    left: PhaseFreeQubitPauli,
    right: PhaseFreeQubitPauli,
) -> StabilizerErrorEquivalenceResult:
    """Decide whether two errors differ by a check-space element."""
    if not isinstance(check_space, CheckSpaceValue):
        _reject(
            "check_space",
            "quantum.stabilizer.not_a_check_space",
            "error equivalence requires a typed register-bound check space",
        )
    register = _admit_register(
        getattr(check_space, "qubit_register", None), "check_space"
    )
    basis_value = getattr(check_space, "basis", None)
    if not isinstance(basis_value, tuple) or len(basis_value) > MAX_CHECK_ROWS:
        _reject(
            "check_space",
            "quantum.stabilizer.invalid_basis",
            "check-space basis is malformed",
        )
    basis = tuple(basis_value)
    n = len(register.qubit_ids)
    rows: list[list[int]] = []
    for basis_row in basis:
        _admit_phase_free(basis_row, "check_space")
        if basis_row.qubit_register != register:
            _reject(
                "check_space",
                "quantum.stabilizer.parent_mismatch",
                "all check rows must use the declared register",
            )
        rows.append([*basis_row.x_bits, *basis_row.z_bits])
    for i, flat_row in enumerate(rows):
        for other in rows[i + 1 :]:
            if _symplectic_pairing(flat_row, other, n):
                _reject(
                    "check_space",
                    "quantum.stabilizer.not_isotropic",
                    "check space must be symplectically isotropic",
                )
    for name, value in (("left", left), ("right", right)):
        _admit_phase_free(value, name)
        if value.qubit_register != register:
            _reject(
                name,
                "quantum.pauli.register_mismatch",
                "errors and check space must use the identical ordered register",
            )
    canonical, pivots = _gf2_rref(rows, 2 * n)
    difference_bits = tuple(
        (a + b) % 2
        for a, b in zip(
            (*left.x_bits, *left.z_bits), (*right.x_bits, *right.z_bits), strict=True
        )
    )
    residual = list(difference_bits)
    for canonical_row, pivot in zip(canonical, pivots, strict=True):
        if residual[pivot]:
            residual = [
                (a + b) % 2 for a, b in zip(residual, canonical_row, strict=True)
            ]
    difference = PhaseFreeQubitPauli(
        register=register,
        x_bits=tuple(difference_bits[:n]),
        z_bits=tuple(difference_bits[n:]),
    )
    canonical_space = CheckSpaceValue(
        register=register,
        basis=tuple(
            PhaseFreeQubitPauli(
                register=register, x_bits=tuple(row[:n]), z_bits=tuple(row[n:])
            )
            for row in canonical
        ),
    )
    return StabilizerErrorEquivalenceResult(
        check_space=canonical_space,
        left=left,
        right=right,
        difference=difference,
        equivalent_mod_stabilizers=not any(residual),
    )


def css_check_space(
    register: QubitRegister,
    x_checks: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    z_checks: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> CSSCheckSpaceResult:
    """Construct a binary CSS check space or return its exact obstruction."""
    admitted_register = _admit_register(register, "register")
    if not isinstance(x_checks, (tuple, list)) or not isinstance(
        z_checks, (tuple, list)
    ):
        _reject(
            "request",
            "quantum.stabilizer.css.invalid_checks",
            "CSS checks must be row tuples",
        )
    x_tuple = tuple(x_checks)
    z_tuple = tuple(z_checks)
    if len(x_tuple) + len(z_tuple) > MAX_CHECK_ROWS:
        raise OperationResourceAdmissionError(
            location=("x_checks", "z_checks"),
            code="quantum.stabilizer.css.too_many_checks",
            message=f"combined CSS check families exceed {MAX_CHECK_ROWS} rows",
        )
    n = len(admitted_register.qubit_ids)
    for family in (x_tuple, z_tuple):
        for row in family:
            if (
                not isinstance(row, tuple)
                or len(row) != n
                or any(type(bit) is not int or bit not in (0, 1) for bit in row)
            ):
                _reject(
                    "request",
                    "quantum.stabilizer.css.invalid_row",
                    "CSS rows must be binary vectors on the register",
                )
    # Admit pairing, both family reductions, the combined reduction, and their
    # maximum row workspace before any pair or matrix is expanded.
    x_count, z_count = len(x_tuple), len(z_tuple)
    row_count = x_count + z_count
    admitted_work = n * x_count * z_count + n * n * row_count + 4 * n * n * row_count
    if admitted_work > 400_000:
        raise OperationResourceAdmissionError(
            location=("x_checks", "z_checks"),
            code="quantum.stabilizer.css.work_over_envelope",
            message="CSS pairing and elimination work exceeds its admitted envelope",
        )
    obstruction = next(
        (
            CSSNonOrthogonalWitness._from_kernel(
                register=admitted_register,
                x_row=i,
                z_row=j,
                x_bits=xrow,
                z_bits=zrow,
            )
            for i, xrow in enumerate(x_tuple)
            for j, zrow in enumerate(z_tuple)
            if sum(a * b for a, b in zip(xrow, zrow, strict=True)) % 2
        ),
        None,
    )
    x_flat = _gf2_rref([list(row) for row in x_tuple], n)[0]
    z_flat = _gf2_rref([list(row) for row in z_tuple], n)[0]
    x_basis = tuple(
        PhaseFreeQubitPauli(
            register=admitted_register, x_bits=tuple(row), z_bits=(0,) * n
        )
        for row in x_flat
    )
    z_basis = tuple(
        PhaseFreeQubitPauli(
            register=admitted_register, x_bits=(0,) * n, z_bits=tuple(row)
        )
        for row in z_flat
    )
    if obstruction is not None:
        return CSSCheckSpaceResult(
            css_check_space=None,
            witness=obstruction,
        )
    combined, _ = _gf2_rref(
        [
            *([*row, *([0] * n)] for row in x_flat),
            *([*([0] * n), *row] for row in z_flat),
        ],
        2 * n,
    )
    check_basis = tuple(
        PhaseFreeQubitPauli(
            register=admitted_register, x_bits=tuple(row[:n]), z_bits=tuple(row[n:])
        )
        for row in combined
    )
    check_space = CheckSpaceValue(register=admitted_register, basis=check_basis)
    css_value = CSSCheckSpaceValue(
        register=admitted_register,
        x_check_basis=x_basis,
        z_check_basis=z_basis,
        check_space=check_space,
    )
    return CSSCheckSpaceResult(css_check_space=css_value)


def _solve_gf2(
    rows: Sequence[Sequence[int]], right_hand_side: Sequence[int], width: int
) -> tuple[int, ...] | None:
    """Return the free-zero solution of a consistent binary linear system."""
    augmented = [
        [*row, value] for row, value in zip(rows, right_hand_side, strict=True)
    ]
    pivot_columns: list[int] = []
    target = 0
    for column in range(width):
        pivot = next(
            (i for i in range(target, len(augmented)) if augmented[i][column]), None
        )
        if pivot is None:
            continue
        augmented[target], augmented[pivot] = augmented[pivot], augmented[target]
        for i in range(len(augmented)):
            if i != target and augmented[i][column]:
                augmented[i] = [
                    (a + b) % 2
                    for a, b in zip(augmented[i], augmented[target], strict=True)
                ]
        pivot_columns.append(column)
        target += 1
    if any(not any(row[:width]) and row[width] for row in augmented):
        return None
    solution = [0] * width
    for i, column in enumerate(pivot_columns):
        solution[column] = augmented[i][width]
    return tuple(solution)


def _admit_css_value(
    value: object,
) -> tuple[QubitRegister, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate source roles and the asserted combined check-space relation."""
    if not isinstance(value, CSSCheckSpaceValue):
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.not_a_css_space",
            "logical-frame construction requires a typed successful CSS check space",
        )
    register = _admit_register(
        getattr(value, "qubit_register", None), "css_check_space"
    )
    x_basis = getattr(value, "x_check_basis", None)
    z_basis = getattr(value, "z_check_basis", None)
    check_space = getattr(value, "check_space", None)
    combined_basis = (
        getattr(check_space, "basis", None)
        if isinstance(check_space, CheckSpaceValue)
        else None
    )
    if (
        not isinstance(x_basis, tuple)
        or not isinstance(z_basis, tuple)
        or not isinstance(check_space, CheckSpaceValue)
        or len(x_basis) + len(z_basis) > MAX_CHECK_ROWS
        or not isinstance(combined_basis, tuple)
        or len(combined_basis) > MAX_CHECK_ROWS
        or getattr(check_space, "qubit_register", None) != register
    ):
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.invalid_css_space",
            "CSS check families and combined check space are malformed",
        )
    n = len(register.qubit_ids)
    rx, rz = len(x_basis), len(z_basis)
    base_work = (
        n * rx * rz
        + 2 * n * n * (rx + rz)
        + 4 * n * n * (rx + rz + len(combined_basis))
    )
    # Include the nullspace, deterministic quotient complement, and all dual
    # linear solves in the same preflight. The bound assumes n candidates and
    # an n-row system for each elimination, including row scans and updates.
    frame_work = n * n * rz + 2 * n**3 * (rx + n) + 2 * n**4 + n**3
    if base_work + frame_work > 8_000_000:
        raise OperationResourceAdmissionError(
            location=("css_check_space",),
            code="quantum.stabilizer.css.logical.work_over_envelope",
            message="CSS quotient validation exceeds its admitted work envelope",
        )
    hx: list[list[int]] = []
    hz: list[list[int]] = []
    for family, target, role in (
        (x_basis, hx, "x"),
        (z_basis, hz, "z"),
    ):
        for row in family:
            _admit_phase_free(row, "css_check_space")
            if row.qubit_register != register:
                _reject(
                    "css_check_space",
                    "quantum.stabilizer.css.logical.parent_mismatch",
                    "CSS check rows must share the declared register",
                )
            if (role == "x" and any(row.z_bits)) or (role == "z" and any(row.x_bits)):
                _reject(
                    "css_check_space",
                    "quantum.stabilizer.css.logical.role_mismatch",
                    "CSS X and Z checks must retain their coordinate roles",
                )
            target.append(list(row.x_bits if role == "x" else row.z_bits))
    if tuple(tuple(row) for row in _gf2_rref(hx, n)[0]) != tuple(
        tuple(row) for row in hx
    ) or tuple(tuple(row) for row in _gf2_rref(hz, n)[0]) != tuple(
        tuple(row) for row in hz
    ):
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.noncanonical_checks",
            "CSS check bases must be canonical independent RREF rows",
        )
    if any(
        sum(a * b for a, b in zip(xrow, zrow, strict=True)) % 2
        for xrow in hx
        for zrow in hz
    ):
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.not_orthogonal",
            "CSS X/Z checks must satisfy H_X H_Z^T = 0",
        )
    source_combined = [[*row, *([0] * n)] for row in hx] + [
        [*([0] * n), *row] for row in hz
    ]
    source_rref = _gf2_rref(source_combined, 2 * n)[0]
    actual_rows: list[list[int]] = []
    for row in check_space.basis:
        _admit_phase_free(row, "css_check_space.check_space")
        if row.qubit_register != register:
            _reject(
                "css_check_space.check_space",
                "quantum.stabilizer.css.logical.parent_mismatch",
                "combined check rows must share the CSS register",
            )
        actual_rows.append([*row.x_bits, *row.z_bits])
    actual_rref = _gf2_rref(actual_rows, 2 * n)[0]
    if source_rref != actual_rref:
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.check_space_mismatch",
            "combined check space must equal the separate CSS check spans",
        )
    return register, tuple(tuple(row) for row in hx), tuple(tuple(row) for row in hz)


def css_logical_pauli_frame(value: CSSCheckSpaceValue) -> CSSLogicalPauliFrame:
    """Return deterministic dual X/Z representatives for the CSS quotient."""
    register, hx, hz = _admit_css_value(value)
    n = len(register.qubit_ids)
    logical_qubits = n - len(hx) - len(hz)
    if logical_qubits < 0:
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.negative_dimension",
            "orthogonal CSS check ranks cannot exceed the register dimension",
        )
    # Pick a deterministic complement of row(H_X) inside ker(H_Z).
    x_normalizer = _gf2_nullspace([list(row) for row in hz], n)
    span_rows = [list(row) for row in hx]
    span_rank = len(span_rows)
    x_logical: list[tuple[int, ...]] = []
    for candidate in x_normalizer:
        expanded, _ = _gf2_rref([*span_rows, list(candidate)], n)
        if len(expanded) > span_rank:
            x_logical.append(candidate)
            span_rows = expanded
            span_rank += 1
    if len(x_logical) != logical_qubits:
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.x_quotient_dimension",
            "X-normalizer quotient dimension disagrees with CSS rank parameters",
        )
    # Solve pairings against the independent basis (H_X, X_logical); this
    # yields Z representatives in ker(H_X), dual to the chosen quotient basis.
    equations = [*([row[:] for row in hx]), *([list(row) for row in x_logical])]
    z_logical: list[tuple[int, ...]] = []
    for logical_index in range(logical_qubits):
        rhs = [0] * len(hx) + [int(i == logical_index) for i in range(logical_qubits)]
        representative = _solve_gf2(equations, rhs, n)
        if representative is None:
            _reject(
                "css_check_space",
                "quantum.stabilizer.css.logical.dual_frame_failure",
                "CSS logical quotient has no exact symplectic dual representative",
            )
        z_logical.append(representative)
    x_values = tuple(
        PhaseFreeQubitPauli(register=register, x_bits=row, z_bits=(0,) * n)
        for row in x_logical
    )
    z_values = tuple(
        PhaseFreeQubitPauli(register=register, x_bits=(0,) * n, z_bits=row)
        for row in z_logical
    )
    if any(
        sum(a * b for a, b in zip(xrow, zrow, strict=True)) % 2 != int(i == j)
        for i, xrow in enumerate(x_logical)
        for j, zrow in enumerate(z_logical)
    ):
        _reject(
            "css_check_space",
            "quantum.stabilizer.css.logical.frame_pairing_failure",
            "constructed CSS logical representatives failed their exact dual-pairing identity",
        )
    return CSSLogicalPauliFrame(
        css_check_space=value,
        x_logical_basis=x_values,
        z_logical_basis=z_values,
        logical_qubits=logical_qubits,
    )


def _css_sector_minimum(
    *,
    check_rows: tuple[tuple[int, ...], ...],
    stabilizer_rows: tuple[tuple[int, ...], ...],
    register: QubitRegister,
    x_role: bool,
) -> tuple[int, PhaseFreeQubitPauli]:
    """Find the first minimum-weight normalizer vector outside stabilizers."""
    n = len(register.qubit_ids)
    check_masks = tuple(
        sum(bit << index for index, bit in enumerate(row)) for row in check_rows
    )
    stabilizer_masks = tuple(
        sum(bit << index for index, bit in enumerate(row)) for row in stabilizer_rows
    )
    pivots = tuple(
        next(index for index, bit in enumerate(row) if bit) for row in stabilizer_rows
    )
    for weight in range(1, n + 1):
        for support in combinations(range(n), weight):
            candidate = sum(1 << index for index in support)
            if any((candidate & check).bit_count() % 2 for check in check_masks):
                continue
            residual = candidate
            for row_mask, pivot in zip(stabilizer_masks, pivots, strict=True):
                if residual & (1 << pivot):
                    residual ^= row_mask
            if residual:
                bits = tuple((candidate >> index) & 1 for index in range(n))
                representative = PhaseFreeQubitPauli(
                    register=register,
                    x_bits=bits if x_role else (0,) * n,
                    z_bits=(0,) * n if x_role else bits,
                )
                return weight, representative
    _reject(
        "css_check_space",
        "quantum.stabilizer.css.distance.missing_logical_representative",
        "nonzero CSS logical quotient must contain a finite-weight representative",
    )


def css_exact_distance(value: CSSCheckSpaceValue) -> CSSDistanceResult:
    r"""Exhaustively compute CSS X/Z distances within an admitted envelope.

    X distance is ``min wt(x)`` for ``x in ker(H_Z) \ row(H_X)``; Z distance
    is ``min wt(z)`` for ``z in ker(H_X) \ row(H_Z)``. Minimum ties are resolved
    by the lexicographically first tuple of register positions.
    """
    register, hx, hz = _admit_css_value(value)
    n = len(register.qubit_ids)
    logical_qubits = n - len(hx) - len(hz)
    if logical_qubits == 0:
        return CSSDistanceResult(
            css_check_space=value,
            logical_qubits=0,
            x_distance=None,
            x_representative=None,
            z_distance=None,
            z_representative=None,
        )
    # Each sector may inspect every binary vector. The count includes both
    # sectors and is checked before constructing support tuples or masks.
    candidate_count = 1 << (n + 1)
    search_work = candidate_count * (n + len(hx) + len(hz))
    if candidate_count > 1_100_000 or search_work > 100_000_000:
        raise OperationResourceAdmissionError(
            location=("css_check_space",),
            code="quantum.stabilizer.css.distance.search_over_envelope",
            message=(
                "complete CSS distance search exceeds the admitted candidate/work envelope"
            ),
        )
    x_distance, x_representative = _css_sector_minimum(
        check_rows=hz,
        stabilizer_rows=hx,
        register=register,
        x_role=True,
    )
    z_distance, z_representative = _css_sector_minimum(
        check_rows=hx,
        stabilizer_rows=hz,
        register=register,
        x_role=False,
    )
    return CSSDistanceResult(
        css_check_space=value,
        logical_qubits=logical_qubits,
        x_distance=x_distance,
        x_representative=x_representative,
        z_distance=z_distance,
        z_representative=z_representative,
    )


def stabilizer_exact_distance(value: CheckSpaceValue) -> StabilizerDistanceResult:
    """Return the minimum mixed-Pauli weight outside the isotropic checks.

    The bounded exhaustive search is over physical Pauli supports and labels,
    testing membership in S-perp and excluding S. Weight ties are ordered by
    register support, then local labels X, Z, Y.
    """
    normalizer = stabilizer_normalizer(value)
    canonical = normalizer.check_space
    register = canonical.qubit_register
    n = len(register.qubit_ids)
    k = normalizer.logical_dimension // 2
    if k == 0:
        return StabilizerDistanceResult(check_space=value, logical_qubits=0)

    candidate_count = (1 << (2 * n)) - 1
    row_count = len(canonical.basis)
    # Per candidate: at most n coordinate steps plus a full row pass for
    # commutation and another for stabilizer-span reduction.
    search_work = candidate_count * (n + 2 * row_count)
    if n > 10 or candidate_count > 1_100_000 or search_work > 50_000_000:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.stabilizer.distance.search_over_envelope",
            message="complete mixed-Pauli distance search exceeds the admitted candidate/work envelope",
        )

    stabilizer_rows = tuple(
        sum(bit << j for j, bit in enumerate(row.x_bits))
        | (sum(bit << j for j, bit in enumerate(row.z_bits)) << n)
        for row in canonical.basis
    )
    pivots = tuple((row & -row).bit_length() - 1 for row in stabilizer_rows)
    check_constraints = tuple(
        sum(bit << j for j, bit in enumerate(row.z_bits))
        | (sum(bit << j for j, bit in enumerate(row.x_bits)) << n)
        for row in canonical.basis
    )
    for weight in range(1, n + 1):
        for support in combinations(range(n), weight):
            for labels in product((1, 2, 3), repeat=weight):
                x_mask = 0
                z_mask = 0
                for position, label in zip(support, labels, strict=True):
                    if label & 1:
                        x_mask |= 1 << position
                    if label & 2:
                        z_mask |= 1 << position
                candidate = x_mask | (z_mask << n)
                if any((candidate & row).bit_count() & 1 for row in check_constraints):
                    continue
                residual = candidate
                for row, pivot in zip(stabilizer_rows, pivots, strict=True):
                    if residual & (1 << pivot):
                        residual ^= row
                if residual:
                    representative = PhaseFreeQubitPauli(
                        register=register,
                        x_bits=tuple((x_mask >> j) & 1 for j in range(n)),
                        z_bits=tuple((z_mask >> j) & 1 for j in range(n)),
                    )
                    return StabilizerDistanceResult(
                        check_space=value,
                        logical_qubits=k,
                        distance=weight,
                        representative=representative,
                    )
    _reject(
        "check_space",
        "quantum.stabilizer.distance.missing_logical_representative",
        "positive-dimensional stabilizer quotient must contain a nontrivial logical Pauli",
    )


__all__ = [
    "canonicalize_check_space",
    "css_check_space",
    "pauli_inverse",
    "pauli_multiply",
    "pauli_pairing",
    "stabilizer_error_equivalence",
    "stabilizer_exact_distance",
    "stabilizer_normalizer",
    "stabilizer_syndrome",
]
