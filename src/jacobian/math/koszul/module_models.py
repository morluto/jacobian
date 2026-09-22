"""Finite commutative algebra/module carriers for module Koszul complexes."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MODULE_ALGEBRA_DIMENSION = 6
MAX_MODULE_DIMENSION = 8
MAX_MODULE_SEQUENCE_LENGTH = 6


def _err(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"koszul.module.{reason}", message)


class FiniteCommutativeAlgebra(StrictModel):
    basis: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_MODULE_ALGEBRA_DIMENSION
    )
    multiplication: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]
    unit: tuple[CanonicalRational, ...] | None = None

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.basis)
        if len(set(self.basis)) != n or len(self.multiplication) != n:
            raise _err(
                "algebra_shape",
                "algebra basis and multiplication dimensions must agree",
            )
        if any(
            len(row) != n or any(len(cell) != n for cell in row)
            for row in self.multiplication
        ):
            raise _err("algebra_shape", "multiplication must be an n by n by n tensor")
        if self.unit is not None and len(self.unit) != n:
            raise _err("unit_shape", "unit coordinates must match the algebra basis")
        return self


class BasedFiniteModule(StrictModel):
    algebra: FiniteCommutativeAlgebra
    basis: tuple[str, ...] = Field(min_length=1, max_length=MAX_MODULE_DIMENSION)
    # action[a][target][source] is multiplication by algebra basis a
    action: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        n, m = len(self.algebra.basis), len(self.basis)
        if len(set(self.basis)) != m or len(self.action) != n:
            raise _err(
                "module_shape",
                "module action must have one matrix per algebra basis element",
            )
        if any(
            len(matrix) != m or any(len(row) != m for row in matrix)
            for matrix in self.action
        ):
            raise _err(
                "module_shape",
                "module action matrices must be square on the module basis",
            )
        return self


class ModuleKoszulRequest(StrictModel):
    algebra: FiniteCommutativeAlgebra
    module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MODULE_SEQUENCE_LENGTH
    )

    @model_validator(mode="after")
    def binding(self) -> Self:
        if self.module.algebra != self.algebra:
            raise _err(
                "parent_mismatch", "module must be based over the supplied algebra"
            )
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "sequence_shape", "sequence coordinates must use the algebra basis"
            )
        return self


class ModuleDifferential(StrictModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    entries: tuple[tuple[int, int, CanonicalRational], ...] = ()

    @model_validator(mode="after")
    def entry_shape(self) -> Self:
        keys = tuple((row, column) for row, column, _ in self.entries)
        if keys != tuple(sorted(set(keys))):
            raise _err(
                "differential_entries", "differential entries must be unique and sorted"
            )
        if any(
            row < 0
            or row >= self.row_count
            or column < 0
            or column >= self.column_count
            for row, column, _ in self.entries
        ):
            raise _err(
                "differential_entries",
                "differential entries must lie on the declared axes",
            )
        return self


class ModuleKoszulComplex(StrictModel):
    algebra: FiniteCommutativeAlgebra
    module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    basis_sizes: tuple[int, ...]
    differentials: tuple[ModuleDifferential, ...]
    square_zero: bool = True

    @model_validator(mode="after")
    def result_shape(self) -> Self:
        if self.module.algebra != self.algebra or len(self.differentials) != len(
            self.sequence
        ):
            raise _err(
                "result_binding",
                "complex must retain its algebra, module, and sequence",
            )
        if len(self.basis_sizes) != len(self.sequence) + 1:
            raise _err("result_shape", "basis sizes must cover every Koszul degree")
        if any(
            (differential.row_count, differential.column_count)
            != (self.basis_sizes[index], self.basis_sizes[index + 1])
            for index, differential in enumerate(self.differentials)
        ):
            raise _err("result_shape", "differentials must use consecutive Koszul axes")
        return self


class ModuleKoszulHomologyRequest(StrictModel):
    complex: ModuleKoszulComplex


class ModuleKoszulHomology(StrictModel):
    complex: ModuleKoszulComplex
    dimensions: tuple[int, ...]
    cycle_dimensions: tuple[int, ...]
    boundary_dimensions: tuple[int, ...]

    @model_validator(mode="after")
    def profile_shape(self) -> Self:
        if len(self.dimensions) != len(self.complex.basis_sizes):
            raise _err("homology_shape", "homology profile must cover every degree")
        return self
