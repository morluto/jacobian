"""Provider-independent values for exact finite universal-algebra operations.

A *finite algebra* is a pair ``A = (|A|, (f_i^A)_i)`` where ``|A|`` is a
finite carrier and each ``f_i^A`` is a complete operation table ``A^r -> A``.
The signature is single-sorted and finitary.  These are direct finite
mathematical values; no theorem prover, model finder, or variety classifier
is introduced.
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import Annotated, Literal, NamedTuple, Self

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


class FiniteAlgebraCarrierMap(StrictModel):
    """One total carrier map between finite algebras with the same signature.

    ``mapping[a]`` is the target carrier index assigned to source carrier
    index ``a``.  This value establishes only totality and exact signature
    binding; :class:`FiniteAlgebraHomomorphism` additionally establishes
    preservation of every basic operation.
    """

    source: FiniteAlgebra = Field(
        description="Source finite algebra whose carrier positions index the map."
    )
    target: FiniteAlgebra = Field(
        description=(
            "Target finite algebra with exactly the source operation identifiers "
            "and arities in the same order."
        )
    )
    mapping: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_CARRIER_SIZE,
        description=(
            "Complete target carrier index for each source carrier position, in "
            "source carrier order."
        ),
    )

    @model_validator(mode="after")
    def require_total_signature_bound_map(self) -> Self:
        if self.source.operations != self.target.operations:
            raise ValueError(
                "source and target operation identifiers and arities must match exactly"
            )
        if len(self.mapping) != len(self.source.carrier):
            raise ValueError("mapping must contain one target index per source element")
        if any(not 0 <= image < len(self.target.carrier) for image in self.mapping):
            raise ValueError("mapping image is outside the target carrier")
        return self


class _PreservationFailure(NamedTuple):
    operation: int
    source_arguments: tuple[int, ...]
    target_arguments: tuple[int, ...]
    source_output: int
    mapped_source_output: int
    target_output: int


def _dense_table_index(arguments: tuple[int, ...], carrier_size: int) -> int:
    index = 0
    for argument in arguments:
        index = index * carrier_size + argument
    return index


def _first_homomorphism_failure(
    carrier_map: FiniteAlgebraCarrierMap,
) -> _PreservationFailure | None:
    """Return the first operation-preservation failure in canonical order."""

    source_size = len(carrier_map.source.carrier)
    target_size = len(carrier_map.target.carrier)
    for operation, symbol in enumerate(carrier_map.source.operations):
        for source_arguments in iproduct(range(source_size), repeat=symbol.arity):
            target_arguments = tuple(
                carrier_map.mapping[argument] for argument in source_arguments
            )
            source_output = carrier_map.source.tables[operation][
                _dense_table_index(source_arguments, source_size)
            ]
            mapped_source_output = carrier_map.mapping[source_output]
            target_output = carrier_map.target.tables[operation][
                _dense_table_index(target_arguments, target_size)
            ]
            if mapped_source_output != target_output:
                return _PreservationFailure(
                    operation=operation,
                    source_arguments=source_arguments,
                    target_arguments=target_arguments,
                    source_output=source_output,
                    mapped_source_output=mapped_source_output,
                    target_output=target_output,
                )
    return None


def _homomorphism_kernel_and_image(
    mapping: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return kernel fibers and image in canonical carrier-position order."""

    fibers: dict[int, list[int]] = {}
    for source_element, target_element in enumerate(mapping):
        fibers.setdefault(target_element, []).append(source_element)
    return tuple(tuple(block) for block in fibers.values()), tuple(sorted(fibers))


class FiniteAlgebraHomomorphism(FiniteAlgebraCarrierMap):
    """A checked carrier map preserving every basic operation exactly."""

    @model_validator(mode="after")
    def require_operation_preservation(self) -> Self:
        failure = _first_homomorphism_failure(self)
        if failure is not None:
            operation_id = self.source.operations[failure.operation].operation_id
            raise ValueError(
                "carrier map does not preserve operation "
                f"{operation_id!r} at source arguments "
                f"{failure.source_arguments}"
            )
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
    "FiniteAlgebraCarrierMap",
    "FiniteAlgebraHomomorphism",
    "FlatTerm",
    "OperationSymbol",
    "Term",
    "VariableTerm",
    "require_term_for_algebra",
]
