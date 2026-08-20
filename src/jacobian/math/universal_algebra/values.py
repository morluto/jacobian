"""Provider-independent values for exact finite universal-algebra operations.

A *finite algebra* is a pair ``A = (|A|, (f_i^A)_i)`` where ``|A|`` is a
finite carrier and each ``f_i^A`` is a complete operation table ``A^r -> A``.
The signature is single-sorted and finitary.  These are direct finite
mathematical values; no theorem prover, model finder, or variety classifier
is introduced.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CARRIER_SIZE = 32
MAX_SIGNATURE_SIZE = 16
MAX_ARITY = 4


class OperationSymbol(StrictModel):
    """One finitary operation symbol."""

    operation_id: str = Field(min_length=1, max_length=64)
    arity: int = Field(ge=0, le=MAX_ARITY)


class FiniteAlgebra(StrictModel):
    """An immutable single-sorted finite algebra with complete operation tables.

    ``carrier`` is a tuple of unique carrier labels.  ``operations`` is a tuple
    of ``(operation_id, arity)`` symbols.  ``tables`` is a tuple of one table
    per operation, in the same order as ``operations``; each table is a tuple
    of carrier-index outputs indexed by the dense Cartesian product of input
    positions in row-major order.
    """

    carrier: tuple[str, ...] = Field(min_length=1)
    operations: tuple[OperationSymbol, ...] = Field(max_length=MAX_SIGNATURE_SIZE)
    tables: tuple[tuple[int, ...], ...] = ()

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.carrier) > MAX_CARRIER_SIZE:
            raise ValueError("carrier size exceeds the bounded budget")
        if len(set(self.carrier)) != len(self.carrier):
            raise ValueError("carrier labels must be unique")
        if len(self.tables) != len(self.operations):
            raise ValueError("tables must have one entry per operation symbol")
        for symbol, table in zip(self.operations, self.tables, strict=True):
            expected_cells = len(self.carrier) ** symbol.arity
            if len(table) != expected_cells:
                raise ValueError(
                    f"operation {symbol.operation_id} table has wrong cell count"
                )
            for output in table:
                if not 0 <= output < len(self.carrier):
                    raise ValueError("table output out of carrier range")
        return self


class Term(StrictModel):
    """A closed source-bound AST node for a finite-algebra term.

    ``kind`` is ``"variable"`` or ``"application"``.  For variables,
    ``variable_id`` is the variable index.  For applications, ``operation`` is
    the operation index in the algebra's signature, and ``children`` are child
    term indices (recursively, but Pydantic does not allow recursive models
    directly — use a flat node list with parent/child indices instead).
    """

    kind: str
    variable_id: int | None = None
    operation: int | None = None
    children: tuple[int, ...] = ()


class FlatTerm(StrictModel):
    """A flat term representation: a list of nodes where each application node
    references its children by index."""

    nodes: tuple[Term, ...] = Field(min_length=1)
    root: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_root(self) -> Self:
        if self.root >= len(self.nodes):
            raise ValueError("root index out of range")
        return self


__all__ = [
    "MAX_ARITY",
    "MAX_CARRIER_SIZE",
    "MAX_SIGNATURE_SIZE",
    "FiniteAlgebra",
    "FlatTerm",
    "OperationSymbol",
    "Term",
]
