"""Typed contracts for bounded finite-model finding."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class FunctionSymbol(ContractModel):
    """One declared function symbol in a finite-model signature."""

    name: str = Field(min_length=1, max_length=64)
    arity: StrictInt = Field(ge=0, le=4)


class RelationSymbol(ContractModel):
    """One declared relation symbol in a finite-model signature."""

    name: str = Field(min_length=1, max_length=64)
    arity: StrictInt = Field(ge=1, le=4)


class FiniteModelSignature(ContractModel):
    """The signature of a bounded finite-model search."""

    functions: tuple[FunctionSymbol, ...] = Field(max_length=8)
    relations: tuple[RelationSymbol, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def require_distinct_names(self) -> Self:
        names: set[str] = set()
        for sym in self.functions:
            if sym.name in names:
                raise ValueError(f"duplicate function name: {sym.name}")
            names.add(sym.name)
        for sym in self.relations:
            if sym.name in names:
                raise ValueError(f"duplicate relation name: {sym.name}")
            names.add(sym.name)
        return self


class FiniteModelAxiom(ContractModel):
    """One SMT-LIB axiom that must hold in the model."""

    name: str = Field(min_length=1, max_length=64)
    smtlib: str = Field(min_length=1, max_length=4096)


class FiniteModelFindRequest(ContractModel):
    """Request to find one finite model or countermodel."""

    signature: FiniteModelSignature
    axioms: tuple[FiniteModelAxiom, ...] = Field(min_length=1, max_length=16)
    carrier_order: StrictInt = Field(ge=1, le=16)
    timeout_ms: StrictInt = Field(default=5000, ge=100, le=30000)

    @model_validator(mode="after")
    def validate_axioms(self) -> Self:
        if len(set(ax.name for ax in self.axioms)) != len(self.axioms):
            raise ValueError("axiom names must be distinct")
        return self


class FiniteModelFunctionTable(ContractModel):
    """The table of one function symbol in a found model."""

    name: str = Field(min_length=1, max_length=64)
    values: tuple[StrictInt, ...] = Field(min_length=1)


class FiniteModelRelationTable(ContractModel):
    """The table of one relation symbol in a found model."""

    name: str = Field(min_length=1, max_length=64)
    entries: tuple[tuple[StrictInt, ...], ...] = Field(default=())


class FiniteModelFindResult(ContractModel):
    """The outcome of a bounded finite-model search."""

    status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN", "INVALID"]
    carrier_order: StrictInt = Field(ge=1, le=16)
    function_tables: tuple[FiniteModelFunctionTable, ...] = Field(default=())
    relation_tables: tuple[FiniteModelRelationTable, ...] = Field(default=())
    examined_count: StrictInt = Field(default=0, ge=0)
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_status_to_tables(self) -> Self:
        sat = self.status == "SATISFIABLE"
        if sat and not self.function_tables:
            raise ValueError("a satisfiable result requires function tables")
        if sat:
            for tbl in self.function_tables:
                if any(
                    v < 0 or v >= self.carrier_order for v in tbl.values
                ):
                    raise ValueError(
                        f"function table {tbl.name} has out-of-range values"
                    )
        return self
