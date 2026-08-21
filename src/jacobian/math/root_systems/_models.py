"""Typed wire contracts for root system operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.root_systems._cartan import require_finite_type

MAX_RANK = 8
MAX_REFLECTION_COORDINATE = ((1 << 53) - 1) // (1 + 3 * MAX_RANK)


class CartanMatrixRequest(StrictModel):
    """A bounded finite-type Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_valid_cartan(self) -> Self:
        n = len(self.matrix)
        if n < 1 or n > MAX_RANK:
            raise ValueError(f"rank must be between 1 and {MAX_RANK}")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("Cartan matrix must be square")
        for i in range(n):
            if self.matrix[i][i] != 2:
                raise ValueError("diagonal entries must be 2")
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
                    raise ValueError("off-diagonal entries must be non-positive")
                if aij * aji not in (0, 1, 2, 3):
                    raise ValueError("off-diagonal product must be 0, 1, 2, or 3")
                if (aij == 0) != (aji == 0):
                    raise ValueError(
                        "generalized Cartan matrix requires a_ij == 0 iff a_ji == 0"
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
            raise ValueError("positive roots are not bound to the Cartan matrix")
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
            raise ValueError("root-system data is not bound to its Cartan matrix")
        if self.negative_roots != tuple(
            tuple(-value for value in root) for root in expected
        ):
            raise ValueError("negative roots must be the negatives of positive roots")
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
            raise ValueError("component data is not bound to the Cartan matrix")
        return self


class SimpleReflectionRequest(StrictModel):
    """Request to apply a simple reflection s_i to a root lattice vector."""

    matrix: tuple[tuple[int, ...], ...]
    vector: tuple[int, ...] = Field(min_length=1)
    simple_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = len(self.matrix)
        if n < 1 or n > MAX_RANK:
            raise ValueError(f"rank must be between 1 and {MAX_RANK}")
        if self.simple_index >= n:
            raise ValueError("simple_index out of range")
        if len(self.vector) != n:
            raise ValueError("vector length must match rank")
        if any(
            abs(coordinate) > MAX_REFLECTION_COORDINATE for coordinate in self.vector
        ):
            raise ValueError(
                "vector coordinates must fit the bounded reflected-coordinate domain"
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
            raise ValueError("reflected_vector must be s_i(vector)")
        return self


class WeylGroupDataRequest(CartanMatrixRequest):
    """Request Weyl group data from a Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]


class WeylGroupDataResult(StrictModel):
    """Weyl group data from a Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]
    rank: int
    group_order: int
    longest_element: tuple[int, ...]
    coxeter_number: int

    @model_validator(mode="after")
    def bind_weyl_data(self) -> Self:
        from jacobian.math.root_systems._operations import _weyl_group_data

        CartanMatrixRequest(matrix=self.matrix)
        order, longest, coxeter = _weyl_group_data([list(row) for row in self.matrix])
        if self.rank != len(self.matrix):
            raise ValueError("rank must match the Cartan matrix")
        if self.group_order != order:
            raise ValueError("group_order must be |W|")
        if self.longest_element != longest:
            raise ValueError("longest_element must be the longest Weyl group element")
        if self.coxeter_number != coxeter:
            raise ValueError("coxeter_number must be h")
        return self
