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
    def require_root_shape(self) -> Self:
        if self.rank != len(self.matrix):
            raise _validation_error(
                "positive_roots_rank", "rank must equal the Cartan-matrix rank"
            )
        if self.num_positive_roots != len(self.positive_roots):
            raise _validation_error(
                "positive_roots_count",
                "num_positive_roots must equal the number of positive roots",
            )
        if len(self.positive_roots) > 120 or any(
            len(root) != self.rank
            or any(coordinate < 0 or coordinate > 6 for coordinate in root)
            or not any(root)
            for root in self.positive_roots
        ):
            raise _validation_error(
                "positive_roots_shape",
                "positive roots must be nonzero bounded vectors on the Cartan axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: CartanMatrixRequest, positive_roots: tuple[tuple[int, ...], ...]
    ) -> Self:
        return cls.model_construct(
            matrix=request.matrix,
            rank=len(request.matrix),
            positive_roots=positive_roots,
            num_positive_roots=len(positive_roots),
        )


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
    def require_root_data_shape(self) -> Self:
        CartanMatrixRequest(matrix=self.cartan_matrix)
        rank = len(self.cartan_matrix)
        if self.rank != rank or self.num_positive_roots != len(self.positive_roots):
            raise _validation_error(
                "root_data_count",
                "root data must agree with its declared rank and count",
            )
        if any(
            len(root) != rank
            or any(coordinate < 0 or coordinate > 6 for coordinate in root)
            or not any(root)
            for root in self.positive_roots
        ) or any(
            len(root) != rank
            or any(coordinate < -6 or coordinate > 0 for coordinate in root)
            or not any(root)
            for root in self.negative_roots
        ):
            raise _validation_error(
                "root_data_root_shape",
                "roots must be nonzero bounded vectors on the Cartan axis",
            )
        if self.simple_roots != tuple(
            tuple(int(i == j) for j in range(rank)) for i in range(rank)
        ):
            raise _validation_error(
                "simple_roots", "simple roots must be the canonical Cartan basis"
            )
        indices = [
            index
            for component in self.components
            for index in component.simple_root_indices
        ]
        if tuple(sorted(indices)) != tuple(range(rank)) or any(
            not component.positive_roots
            or tuple(sorted(set(component.simple_root_indices)))
            != component.simple_root_indices
            or any(not 0 <= index < rank for index in component.simple_root_indices)
            or any(
                len(root) != rank
                or any(coordinate < 0 or coordinate > 6 for coordinate in root)
                or not any(root)
                for root in component.positive_roots
            )
            or len(component.highest_root) != rank
            or any(
                coordinate < 0 or coordinate > 6
                for coordinate in component.highest_root
            )
            or len(component.marks) != len(component.simple_root_indices)
            or any(mark < 0 or mark > 6 for mark in component.marks)
            or component.coxeter_number < 2
            or component.coxeter_number > 49
            for component in self.components
        ):
            raise _validation_error(
                "component_data_shape",
                "components must be a nonempty partition with bounded root data",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: CartanMatrixRequest,
        *,
        positive_roots: tuple[tuple[int, ...], ...],
        negative_roots: tuple[tuple[int, ...], ...],
        simple_roots: tuple[tuple[int, ...], ...],
        components: tuple[RootComponentData, ...],
    ) -> Self:
        return cls.model_construct(
            rank=len(request.matrix),
            cartan_matrix=request.matrix,
            positive_roots=positive_roots,
            negative_roots=negative_roots,
            simple_roots=simple_roots,
            num_positive_roots=len(positive_roots),
            components=components,
        )


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
    def require_reflection_shape(self) -> Self:
        SimpleReflectionRequest(
            matrix=self.matrix,
            vector=self.vector,
            simple_index=self.simple_index,
        )
        if len(self.reflected_vector) != len(self.vector) or any(
            abs(coordinate) > MAX_REFLECTION_COORDINATE
            for coordinate in self.reflected_vector
        ):
            raise _validation_error(
                "reflected_vector_shape",
                "reflected_vector must fit the bounded root-lattice axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: SimpleReflectionRequest, reflected_vector: tuple[int, ...]
    ) -> Self:
        return cls.model_construct(
            matrix=request.matrix,
            vector=request.vector,
            simple_index=request.simple_index,
            reflected_vector=reflected_vector,
        )


class WeylGroupOrderResult(StrictModel):
    """The exact order of the Weyl group generated by one Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]
    group_order: int = Field(ge=1, le=MAX_WEYL_GROUP_ORDER)
    method: Literal["SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION"] = (
        "SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION"
    )

    @model_validator(mode="after")
    def require_order_domain(self) -> Self:
        CartanMatrixRequest(matrix=self.matrix)
        return self

    @classmethod
    def _from_kernel(cls, request: CartanMatrixRequest, group_order: int) -> Self:
        return cls.model_construct(
            matrix=request.matrix,
            group_order=group_order,
            method="SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION",
        )
