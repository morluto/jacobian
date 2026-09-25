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
        embedding = self.normalizer_embedding.matrix.entries
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
        return self

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
        return cls.model_construct(
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
