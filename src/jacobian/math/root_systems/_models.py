"""Typed wire contracts for root system operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.root_systems._cartan import require_finite_type

MAX_RANK = 8
MAX_REFLECTION_COORDINATE = ((1 << 53) - 1) // (1 + 3 * MAX_RANK)
# E8 is the largest finite crystallographic Weyl group at the admitted rank.
MAX_WEYL_GROUP_ORDER = 696_729_600


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by root-system contracts."""

    return PydanticCustomError(f"root_system.{reason}", message)


class CartanMatrixRequest(StrictModel):
    """A bounded finite-type Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...] = Field(
        description=(
            "Finite-type generalized Cartan matrix of rank 1 through "
            f"{MAX_RANK}: square, diagonal entries 2, non-positive "
            "off-diagonal entries with paired products in {0, 1, 2, 3}, "
            "and positive-definite symmetrization."
        )
    )

    @model_validator(mode="after")
    def require_valid_cartan(self) -> Self:
        n = len(self.matrix)
        if n < 1 or n > MAX_RANK:
            raise _validation_error(
                "rank_out_of_range", f"rank must be between 1 and {MAX_RANK}"
            )
        for row in self.matrix:
            if len(row) != n:
                raise _validation_error("not_square", "Cartan matrix must be square")
        for i in range(n):
            if self.matrix[i][i] != 2:
                raise _validation_error("diagonal_entry", "diagonal entries must be 2")
        self._check_off_diagonal(n)
        require_finite_type(self.matrix)
        return self

    def _check_off_diagonal(self, n: int) -> None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                aij = self.matrix[i][j]
                aji = self.matrix[j][i]
                if aij > 0:
                    raise _validation_error(
                        "positive_off_diagonal",
                        "off-diagonal entries must be non-positive",
                    )
                if aij * aji not in (0, 1, 2, 3):
                    raise _validation_error(
                        "off_diagonal_product",
                        "off-diagonal product must be 0, 1, 2, or 3",
                    )
                if (aij == 0) != (aji == 0):
                    raise _validation_error(
                        "zero_pattern",
                        "generalized Cartan matrix requires a_ij == 0 iff a_ji == 0",
                    )


class PositiveRootsResult(CartanMatrixRequest):
    """The positive roots of a root system."""

    rank: int
    positive_roots: tuple[tuple[int, ...], ...]
    num_positive_roots: int

    @model_validator(mode="after")
    def bind_roots(self) -> Self:
        from jacobian.math.root_systems._cartan import positive_roots

        expected = positive_roots(self.matrix)
        if self.positive_roots != expected or self.num_positive_roots != len(expected):
            raise _validation_error(
                "positive_roots_mismatch",
                "positive roots are not bound to the Cartan matrix",
            )
        return self


class RootComponentData(StrictModel):
    simple_root_indices: tuple[int, ...]
    positive_roots: tuple[tuple[int, ...], ...]
    highest_root: tuple[int, ...]
    marks: tuple[int, ...]
    coxeter_number: int


class RootSystemDataResult(StrictModel):
    """Complete root system data from a Cartan matrix."""

    rank: int
    cartan_matrix: tuple[tuple[int, ...], ...]
    positive_roots: tuple[tuple[int, ...], ...]
    negative_roots: tuple[tuple[int, ...], ...]
    simple_roots: tuple[tuple[int, ...], ...]
    num_positive_roots: int
    components: tuple[RootComponentData, ...]

    @model_validator(mode="after")
    def bind_root_data(self) -> Self:
        from jacobian.math.root_systems._cartan import (
            connected_components,
            positive_roots,
        )

        CartanMatrixRequest(matrix=self.cartan_matrix)
        expected = positive_roots(self.cartan_matrix)
        rank = len(self.cartan_matrix)
        simple = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
        if (
            self.rank != rank
            or self.simple_roots != simple
            or self.positive_roots != expected
            or self.num_positive_roots != len(expected)
        ):
            raise _validation_error(
                "root_data_mismatch",
                "root-system data is not bound to its Cartan matrix",
            )
        if self.negative_roots != tuple(
            tuple(-value for value in root) for root in expected
        ):
            raise _validation_error(
                "negative_roots_mismatch",
                "negative roots must be the negatives of positive roots",
            )
        expected_components = []
        for indices in connected_components(self.cartan_matrix):
            roots = tuple(
                root
                for root in expected
                if any(root[index] for index in indices)
                and all(
                    root[index] == 0 for index in range(rank) if index not in indices
                )
            )
            highest = max(roots, key=lambda root: sum(root))
            marks = tuple(highest[index] for index in indices)
            expected_components.append(
                RootComponentData(
                    simple_root_indices=indices,
                    positive_roots=roots,
                    highest_root=highest,
                    marks=marks,
                    coxeter_number=sum(marks) + 1,
                )
            )
        if self.components != tuple(expected_components):
            raise _validation_error(
                "component_data_mismatch",
                "component data is not bound to the Cartan matrix",
            )
        return self


class SimpleReflectionRequest(StrictModel):
    """One bounded simple reflection on a finite-type root lattice."""

    matrix: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_RANK,
        description=(
            "Finite-type generalized Cartan matrix of rank 1 through "
            f"{MAX_RANK}; it must meet the same Cartan conditions as "
            "``CartanMatrixRequest.matrix``."
        ),
    )
    vector: tuple[
        Annotated[
            int,
            Field(ge=-MAX_REFLECTION_COORDINATE, le=MAX_REFLECTION_COORDINATE),
        ],
        ...,
    ] = Field(
        min_length=1,
        description=(
            "Root-lattice coordinates in the matrix's simple-root basis; "
            "length must equal the Cartan-matrix rank, and each coordinate "
            f"must lie in [-{MAX_REFLECTION_COORDINATE}, "
            f"{MAX_REFLECTION_COORDINATE}]."
        ),
    )
    simple_index: int = Field(
        ge=0,
        description=(
            "Zero-based simple-root index; it must be smaller than the "
            "Cartan-matrix rank."
        ),
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = len(self.matrix)
        if n < 1 or n > MAX_RANK:
            raise _validation_error(
                "rank_out_of_range", f"rank must be between 1 and {MAX_RANK}"
            )
        if self.simple_index >= n:
            raise _validation_error(
                "simple_index_out_of_range", "simple_index out of range"
            )
        if len(self.vector) != n:
            raise _validation_error(
                "vector_length_mismatch", "vector length must match rank"
            )
        if any(
            abs(coordinate) > MAX_REFLECTION_COORDINATE for coordinate in self.vector
        ):
            raise _validation_error(
                "vector_coordinate_out_of_range",
                "vector coordinates must fit the bounded reflected-coordinate domain",
            )
        CartanMatrixRequest(matrix=self.matrix)
        return self


class SimpleReflectionResult(StrictModel):
    """Result of applying a simple reflection to a vector."""

    matrix: tuple[tuple[int, ...], ...]
    vector: tuple[int, ...]
    simple_index: int
    reflected_vector: tuple[int, ...]

    @model_validator(mode="after")
    def bind_reflection(self) -> Self:
        from jacobian.math.root_systems._operations import _apply_reflection

        SimpleReflectionRequest(
            matrix=self.matrix,
            vector=self.vector,
            simple_index=self.simple_index,
        )
        reflected = _apply_reflection(
            [list(row) for row in self.matrix], list(self.vector), self.simple_index
        )
        if self.reflected_vector != tuple(reflected):
            raise _validation_error(
                "reflected_vector_mismatch", "reflected_vector must be s_i(vector)"
            )
        return self


class WeylGroupOrderResult(StrictModel):
    """The exact order of the Weyl group generated by one Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]
    group_order: int = Field(ge=1, le=MAX_WEYL_GROUP_ORDER)
    method: Literal["SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION"] = (
        "SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION"
    )

    @model_validator(mode="after")
    def bind_weyl_group_order(self) -> Self:
        from jacobian.math.root_systems._operations import _weyl_group_order

        CartanMatrixRequest(matrix=self.matrix)
        if self.group_order != _weyl_group_order(self.matrix):
            raise _validation_error(
                "group_order_mismatch",
                "group_order must equal the order of the root action",
            )
        return self
