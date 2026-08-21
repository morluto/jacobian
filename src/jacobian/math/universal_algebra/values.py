"""Provider-independent values for exact finite universal-algebra operations.

A *finite algebra* is a pair ``A = (|A|, (f_i^A)_i)`` where ``|A|`` is a
finite carrier and each ``f_i^A`` is a complete operation table ``A^r -> A``.
The signature is single-sorted and finitary.  These are direct finite
mathematical values; no theorem prover, model finder, or variety classifier
is introduced.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CARRIER_SIZE = 32
MAX_SIGNATURE_SIZE = 16
MAX_ARITY = 4
MAX_TERM_NODES = 256
MAX_TERM_DEPTH = 64
MAX_TABLE_CELLS = 65_536


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
        if len({symbol.operation_id for symbol in self.operations}) != len(
            self.operations
        ):
            raise ValueError("operation identifiers must be unique")
        if sum(len(table) for table in self.tables) > MAX_TABLE_CELLS:
            raise ValueError("operation tables exceed the bounded cell budget")
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


class VariableTerm(StrictModel):
    kind: Literal["variable"]
    variable_id: int = Field(ge=0, le=255, strict=True)


class ApplicationTerm(StrictModel):
    kind: Literal["application"]
    operation: int = Field(ge=0, le=MAX_SIGNATURE_SIZE - 1, strict=True)
    children: tuple[int, ...] = Field(default=(), max_length=MAX_ARITY)


Term = Annotated[VariableTerm | ApplicationTerm, Field(discriminator="kind")]


class FlatTerm(StrictModel):
    """A flat term representation: a list of nodes where each application node
    references its children by index."""

    nodes: tuple[Term, ...] = Field(min_length=1, max_length=MAX_TERM_NODES)
    root: int = Field(ge=0)

    @model_validator(mode="after")
    def require_closed_acyclic_ast(self) -> Self:
        if self.root >= len(self.nodes):
            raise ValueError("root index out of range")
        for index, node in enumerate(self.nodes):
            if isinstance(node, ApplicationTerm) and any(
                child < 0 or child >= index for child in node.children
            ):
                raise ValueError("application children must reference earlier nodes")
        reachable = _reachable_nodes(self.nodes, self.root)
        if reachable != set(range(len(self.nodes))):
            raise ValueError("every term node must be reachable from the root")
        if _term_depths(self.nodes)[self.root] > MAX_TERM_DEPTH:
            raise ValueError("term depth exceeds the bounded budget")
        return self

    @property
    def variable_count(self) -> int:
        identifiers = tuple(
            node.variable_id for node in self.nodes if isinstance(node, VariableTerm)
        )
        return max(identifiers, default=-1) + 1


def _reachable_nodes(nodes: tuple[Term, ...], root: int) -> set[int]:
    reachable: set[int] = set()
    pending = [root]
    while pending:
        index = pending.pop()
        if index in reachable:
            continue
        reachable.add(index)
        node = nodes[index]
        if isinstance(node, ApplicationTerm):
            pending.extend(node.children)
    return reachable


def _term_depths(nodes: tuple[Term, ...]) -> tuple[int, ...]:
    depths: list[int] = []
    for node in nodes:
        if isinstance(node, VariableTerm) or not node.children:
            depths.append(1)
        else:
            depths.append(1 + max(depths[child] for child in node.children))
    return tuple(depths)


def require_term_for_algebra(term: FlatTerm, algebra: FiniteAlgebra) -> None:
    """Bind application nodes to one finite signature before evaluation."""

    for node in term.nodes:
        if not isinstance(node, ApplicationTerm):
            continue
        if node.operation >= len(algebra.operations):
            raise ValueError("term operation index out of range")
        if len(node.children) != algebra.operations[node.operation].arity:
            raise ValueError("term application arity does not match the operation")


__all__ = [
    "MAX_ARITY",
    "MAX_CARRIER_SIZE",
    "MAX_SIGNATURE_SIZE",
    "ApplicationTerm",
    "FiniteAlgebra",
    "FlatTerm",
    "OperationSymbol",
    "Term",
    "VariableTerm",
    "require_term_for_algebra",
]
