"""Exact register-bound values for the phase-free logical Pauli quotient."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
)
from jacobian.math.quantum._models import (
    MAX_QUBITS,
    CheckSpaceValue,
    PhaseFreeQubitPauli,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quantum.logical_pauli_space.{reason}", message)


_GF2 = FiniteFieldPresentation(
    characteristic=2,
    modulus_coefficients=(0, 1),
    generator="a",
)


def _ambient_axis(check_space: CheckSpaceValue) -> Axis:
    ids = check_space.qubit_register.qubit_ids
    return Axis(
        name="qubit_pauli_symplectic_coordinates",
        labels=tuple(f"X:{qubit}" for qubit in ids)
        + tuple(f"Z:{qubit}" for qubit in ids),
    )


def _check_axis(check_space: CheckSpaceValue) -> Axis:
    return Axis(
        name="canonical_stabilizer_check_basis",
        labels=tuple(f"s{index}" for index in range(len(check_space.basis))),
    )


def _normalizer_axis(size: int) -> Axis:
    return Axis(
        name="canonical_stabilizer_normalizer_basis",
        labels=tuple(f"n{index}" for index in range(size)),
    )


def _quotient_axis(size: int) -> Axis:
    return Axis(
        name="canonical_logical_pauli_quotient_basis",
        labels=tuple(f"l{index}" for index in range(size)),
    )


def _binary_product(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    if not left:
        return ()
    columns = len(right[0]) if right else 0
    inner = len(right)
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(inner)) % 2
            for column in range(columns)
        )
        for row in range(len(left))
    )


def _binary_rank(rows: tuple[tuple[int, ...], ...], columns: int) -> int:
    """Return GF(2) rank with work bounded by the admitted 64 coordinate cap."""
    packed = [sum(value << column for column, value in enumerate(row)) for row in rows]
    rank = 0
    for column in range(columns):
        pivot = next(
            (
                index
                for index in range(rank, len(packed))
                if (packed[index] >> column) & 1
            ),
            None,
        )
        if pivot is None:
            continue
        packed[rank], packed[pivot] = packed[pivot], packed[rank]
        for index in range(len(packed)):
            if index != rank and (packed[index] >> column) & 1:
                packed[index] ^= packed[rank]
        rank += 1
    return rank


def _check_basis_is_canonical(check_space: CheckSpaceValue) -> bool:
    """Check the producer's RREF postcondition without reducing the rows again."""
    n = len(check_space.qubit_register.qubit_ids)
    pivots: list[int] = []
    rows = tuple(
        sum(bit << index for index, bit in enumerate((*row.x_bits, *row.z_bits)))
        for row in check_space.basis
    )
    for row in rows:
        pivot = next((index for index in range(2 * n) if (row >> index) & 1), None)
        if pivot is None or (pivots and pivot <= pivots[-1]):
            return False
        if any((other >> pivot) & 1 for other in rows if other != row):
            return False
        pivots.append(pivot)
    return True


def _ambient_pairing(left: tuple[int, ...], right: tuple[int, ...], width: int) -> int:
    return (
        sum(
            left[index] * right[width + index] + left[width + index] * right[index]
            for index in range(width)
        )
        % 2
    )


class LogicalPauliSpace(StrictModel):
    """A source-bound coordinate model of ``S^perp / S`` over ``GF(2)``.

    The quotient basis consists of ambient Pauli representatives. The maps
    retain the stabilizer inclusion, normalizer embedding, quotient projection,
    and a linear section selecting those representatives.
    """

    check_space: CheckSpaceValue
    stabilizer_inclusion: FiniteLinearMap
    normalizer_embedding: FiniteLinearMap
    quotient_projection: FiniteLinearMap
    quotient_lift: FiniteLinearMap
    induced_symplectic_form: AxisBoundMatrix
    logical_qubits: StrictInt = Field(ge=0, le=MAX_QUBITS)

    @model_validator(mode="after")
    def require_bound_quotient_shape(self) -> Self:
        register = self.check_space.qubit_register
        width = len(register.qubit_ids)
        normalizer_dimension = len(self.normalizer_embedding.source_axis.labels)
        quotient_dimension = len(self.quotient_projection.target_axis.labels)
        check_count = len(self.check_space.basis)
        ambient = _ambient_axis(self.check_space)
        check_axis = _check_axis(self.check_space)
        normalizer_axis = _normalizer_axis(normalizer_dimension)
        quotient_axis = _quotient_axis(quotient_dimension)

        if normalizer_dimension != 2 * width - check_count:
            raise _validation_error(
                "normalizer_dimension",
                "normalizer basis size must be 2n minus check rank",
            )
        if quotient_dimension != normalizer_dimension - check_count:
            raise _validation_error(
                "quotient_dimension",
                "quotient basis size must be dim(S-perp) minus dim(S)",
            )
        if quotient_dimension != 2 * self.logical_qubits:
            raise _validation_error(
                "logical_dimension",
                "logical quotient dimension must equal twice the logical qubit count",
            )
        expected_maps = (
            (self.stabilizer_inclusion, check_axis, normalizer_axis),
            (self.normalizer_embedding, normalizer_axis, ambient),
            (self.quotient_projection, normalizer_axis, quotient_axis),
            (self.quotient_lift, quotient_axis, normalizer_axis),
        )
        for linear_map, source_axis, target_axis in expected_maps:
            if (
                not isinstance(linear_map, FiniteLinearMap)
                or linear_map.source_axis != source_axis
                or linear_map.target_axis != target_axis
                or linear_map.matrix.prime != 2
            ):
                raise _validation_error(
                    "map_axes",
                    "quotient maps must use their source-bound binary coordinate axes",
                )
        form = self.induced_symplectic_form
        if (
            form.presentation != _GF2
            or form.row_axis != quotient_axis
            or form.column_axis != quotient_axis
            or len(form.entries) != quotient_dimension
            or any(
                len(row) != quotient_dimension
                or any(
                    not isinstance(entry, FiniteFieldElement)
                    or entry.presentation != _GF2
                    or entry.coordinates not in ((0,), (1,))
                    for entry in row
                )
                for row in form.entries
            )
        ):
            raise _validation_error(
                "symplectic_form_axes",
                "induced form must be a GF(2) matrix on the quotient coordinate axis",
            )
        self._require_semantic_relations(
            width, normalizer_dimension, quotient_dimension
        )
        return self

    def _require_semantic_relations(
        self, width: int, normalizer_dimension: int, quotient_dimension: int
    ) -> None:
        check_count = len(self.check_space.basis)
        if not _check_basis_is_canonical(self.check_space):
            raise _validation_error(
                "check_basis_canonical",
                "source check rows must be an independent canonical RREF basis",
            )

        inclusion = self.stabilizer_inclusion.matrix.entries
        embedding = self.normalizer_embedding.matrix.entries
        projection = self.quotient_projection.matrix.entries
        lift = self.quotient_lift.matrix.entries
        form = self.induced_symplectic_form
        check_rows = tuple((*row.x_bits, *row.z_bits) for row in self.check_space.basis)

        # The embedding has one ambient coordinate selector per normalizer
        # coordinate, so its columns are independent. Requiring E*A=C^T binds
        # the inclusion to the actual source checks and, since C is independent,
        # makes A injective as well.
        embedded_checks = _binary_product(embedding, inclusion)
        transposed_checks = tuple(
            tuple(check_rows[column][row] for column in range(check_count))
            for row in range(2 * width)
        )
        if embedded_checks != transposed_checks:
            raise _validation_error(
                "stabilizer_inclusion_relation",
                "embedded stabilizer inclusion must equal the source check basis",
            )

        # Every embedded normalizer vector must commute with every check. The
        # dimension and independence bounds above then identify its image with
        # S-perp, rather than merely a subspace of S-perp.
        for check in check_rows:
            for column in range(normalizer_dimension):
                normalizer_vector = tuple(row[column] for row in embedding)
                if _ambient_pairing(check, normalizer_vector, width):
                    raise _validation_error(
                        "normalizer_orthogonality",
                        "embedded normalizer columns must be orthogonal to every check",
                    )

        projection_of_checks = _binary_product(projection, inclusion)
        if any(value for row in projection_of_checks for value in row):
            raise _validation_error(
                "quotient_kernel_relation",
                "stabilizer inclusion must lie in the quotient projection kernel",
            )
        projection_of_lift = _binary_product(projection, lift)
        identity = tuple(
            tuple(int(row == column) for column in range(quotient_dimension))
            for row in range(quotient_dimension)
        )
        if projection_of_lift != identity:
            raise _validation_error(
                "quotient_section_relation",
                "quotient projection composed with its section must be identity",
            )

        # P is surjective by P*L=I, A is injective by E*A=C^T, and
        # rank(A)+rank(P)=N. Together with P*A=0 this proves ker(P)=image(A).
        # Also, E maps onto S-perp by its declared dimension and orthogonality;
        # hence the induced pairing on S-perp/S is nondegenerate.
        quotient_representatives = _binary_product(embedding, lift)
        ambient_form = tuple(
            tuple(
                _ambient_pairing(
                    tuple(
                        quotient_representatives[row][left] for row in range(2 * width)
                    ),
                    tuple(
                        quotient_representatives[row][right] for row in range(2 * width)
                    ),
                    width,
                )
                for right in range(quotient_dimension)
            )
            for left in range(quotient_dimension)
        )
        form_values = tuple(
            tuple(entry.coordinates[0] for entry in row) for row in form.entries
        )
        if _binary_rank(form_values, quotient_dimension) != quotient_dimension:
            raise _validation_error(
                "induced_form_nondegenerate",
                "induced quotient symplectic form must be nondegenerate",
            )
        if form_values != ambient_form:
            raise _validation_error(
                "induced_form_pairing",
                "induced quotient form must equal ambient Pauli pairings",
            )

        for basis_index in range(normalizer_dimension):
            unit = tuple(int(row == basis_index) for row in range(normalizer_dimension))
            selector = next(
                (
                    position
                    for position, row in enumerate(embedding)
                    if tuple(row[column] for column in range(normalizer_dimension))
                    == unit
                ),
                None,
            )
            if selector is None:
                raise _validation_error(
                    "normalizer_embedding_coordinates",
                    "normalizer embedding must retain coordinate selector rows",
                )

    @classmethod
    def _from_kernel(
        cls,
        *,
        check_space: CheckSpaceValue,
        stabilizer_inclusion: FiniteLinearMap,
        normalizer_embedding: FiniteLinearMap,
        quotient_projection: FiniteLinearMap,
        quotient_lift: FiniteLinearMap,
        induced_symplectic_form: AxisBoundMatrix,
    ) -> Self:
        return cls(
            check_space=check_space,
            stabilizer_inclusion=stabilizer_inclusion,
            normalizer_embedding=normalizer_embedding,
            quotient_projection=quotient_projection,
            quotient_lift=quotient_lift,
            induced_symplectic_form=induced_symplectic_form,
            logical_qubits=len(quotient_projection.target_axis.labels) // 2,
        )

    @property
    def normalizer_coordinate_positions(self) -> tuple[int, ...]:
        matrix = self.normalizer_embedding.matrix.entries
        dimension = len(self.normalizer_embedding.source_axis.labels)
        return tuple(
            next(
                position
                for position, row in enumerate(matrix)
                if tuple(row[column] for column in range(dimension))
                == tuple(int(index == coordinate) for index in range(dimension))
            )
            for coordinate in range(dimension)
        )

    @property
    def normalizer_basis(self) -> tuple[PhaseFreeQubitPauli, ...]:
        register = self.check_space.qubit_register
        n = len(register.qubit_ids)
        matrix = self.normalizer_embedding.matrix.entries
        return tuple(
            PhaseFreeQubitPauli(
                register=register,
                x_bits=tuple(matrix[index][column] for index in range(n)),
                z_bits=tuple(matrix[n + index][column] for index in range(n)),
            )
            for column in range(len(self.normalizer_embedding.source_axis.labels))
        )

    @property
    def quotient_basis(self) -> tuple[PhaseFreeQubitPauli, ...]:
        register = self.check_space.qubit_register
        n = len(register.qubit_ids)
        embedding = self.normalizer_embedding.matrix.entries
        lift = self.quotient_lift.matrix.entries
        quotient_dimension = len(self.quotient_lift.source_axis.labels)
        normalizer_dimension = len(self.normalizer_embedding.source_axis.labels)
        ambient_rows = tuple(
            tuple(
                sum(
                    embedding[ambient][normalizer] * lift[normalizer][quotient]
                    for normalizer in range(normalizer_dimension)
                )
                % 2
                for quotient in range(quotient_dimension)
            )
            for ambient in range(2 * n)
        )
        return tuple(
            PhaseFreeQubitPauli(
                register=register,
                x_bits=tuple(ambient_rows[index][column] for index in range(n)),
                z_bits=tuple(ambient_rows[n + index][column] for index in range(n)),
            )
            for column in range(quotient_dimension)
        )


__all__ = ["LogicalPauliSpace"]
