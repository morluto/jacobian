"""Provider-independent exact values for finite relational structures.

A structure binds a finite carrier with canonical labels ``0..n-1``, a ranked
signature of relation symbols with unique IDs and bounded arities, and one
COMPLETE tuple table per symbol. Tuple tables are exact finite sets: a tuple
omitted from a table is definitively NOT in the relation — omission is exact
nonmembership, never unknown. Row order is transport data only; tables are
canonicalized once into strictly increasing lexicographic order without
duplicates. Many-sorted carriers, function symbols, infinite carriers, and
partial/weighted relations are out of scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers

MAX_RELATIONAL_CARRIER = 64
MAX_RELATIONAL_SYMBOLS = 8
MAX_RELATIONAL_ARITY = 4
MAX_RELATIONAL_TABLE_ROWS = 4_096
# Exhaustive homomorphism transport work: every source tuple of every symbol
# is replayed once. This aggregate ceiling is strictly tighter than the
# implied per-table structural maximum so the published transport envelope
# is a real admission bound.
MAX_RELATIONAL_TRANSPORT_TUPLES = 16_384
# An m-ary operation is represented by its complete table on A^m. The
# operation-specific envelope bounds arity and table cells independently of
# the larger carrier envelope.
MAX_RELATIONAL_POLYMORPHISM_ARITY = 8
MAX_RELATIONAL_OPERATION_TABLE_CELLS = 16_384

RelationSymbolId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.structure.{reason}", message)


class FiniteRelationSymbol(StrictModel):
    """One ranked relation symbol of a finite relational signature."""

    symbol_id: RelationSymbolId
    arity: StrictInt = Field(
        ge=0,
        le=MAX_RELATIONAL_ARITY,
        description=(
            "The declared rank. Arity 0 is a nullary relation whose complete "
            f"table is either empty (false) or the single empty tuple (true); "
            f"the construction admits arity at most {MAX_RELATIONAL_ARITY}."
        ),
    )


class FiniteRelationalStructure(StrictModel):
    """One exact finite relational structure over a ranked signature.

    The carrier is the canonical label set ``0..carrier_size-1``.
    ``relation_tables[i]`` is the COMPLETE tuple table of ``signature[i]``:
    strictly increasing unique rows of exactly the declared arity with every
    coordinate in the carrier. A tuple absent from a table is exactly not in
    the relation. Two structures with identical table shapes but different
    signatures are distinct values; the signature is structural identity, not
    presentation.
    """

    carrier_size: StrictInt = Field(
        ge=0,
        le=MAX_RELATIONAL_CARRIER,
        description=(
            "The cardinality of the canonical carrier 0..n-1; at most "
            f"{MAX_RELATIONAL_CARRIER}. The empty carrier is a legitimate "
            "degenerate structure whose tables can only contain nullary rows."
        ),
    )
    signature: tuple[FiniteRelationSymbol, ...] = Field(
        default=(),
        max_length=MAX_RELATIONAL_SYMBOLS,
        description=(
            "The ranked relation symbols in deterministic order, with unique "
            f"IDs; at most {MAX_RELATIONAL_SYMBOLS} symbols. The empty "
            "signature is valid and makes every carrier map a homomorphism."
        ),
    )
    relation_tables: tuple[tuple[tuple[StrictInt, ...], ...], ...] = Field(
        default=(),
        description=(
            "One complete tuple table per signature symbol, in signature "
            f"order; at most {MAX_RELATIONAL_TABLE_ROWS} rows each. Duplicate "
            "rows normalize once and row order canonicalizes to increasing "
            "lexicographic order. An omitted tuple is exactly NOT in the "
            "relation."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def canonicalize_table_rows(cls, data: object) -> object:
        """Normalize duplicate rows and row order once, structurally.

        Only ordering and multiplicity are transport data; malformed rows are
        passed through untouched so the typed validators own their rejection.
        """

        if not isinstance(data, Mapping):
            return data
        canonical = canonicalize_json_containers(data)
        if not isinstance(canonical, Mapping):
            return canonical
        if "relation_tables" not in canonical:
            return canonical
        tables = canonical["relation_tables"]
        if not isinstance(tables, (list, tuple)):
            return canonical
        canonical_tables = []
        for table in tables:
            if not isinstance(table, (list, tuple)) or any(
                not isinstance(row, (list, tuple))
                or any(
                    not isinstance(coordinate, int) or isinstance(coordinate, bool)
                    for coordinate in row
                )
                for row in table
            ):
                canonical_tables.append(table)
                continue
            canonical_tables.append(tuple(sorted({tuple(row) for row in table})))
        normalized = dict(canonical)
        normalized["relation_tables"] = tuple(canonical_tables)
        return normalized

    @model_validator(mode="after")
    def require_complete_canonical_tables(self) -> Self:
        ids = tuple(symbol.symbol_id for symbol in self.signature)
        if len(set(ids)) != len(ids):
            raise _validation_error(
                "symbol_identity", "relation symbol IDs must be unique in a signature"
            )
        if len(self.relation_tables) != len(self.signature):
            raise _validation_error(
                "table_count",
                "exactly one complete tuple table is required per relation "
                "symbol, in signature order",
            )
        for symbol, table in zip(self.signature, self.relation_tables, strict=True):
            if len(table) > MAX_RELATIONAL_TABLE_ROWS:
                raise _validation_error(
                    "table_rows",
                    f"a relation table exceeds the {MAX_RELATIONAL_TABLE_ROWS}-row "
                    "envelope",
                )
            if table != tuple(sorted(set(table))):
                raise _validation_error(
                    "table_order",
                    "relation tables canonicalize to strictly increasing "
                    "unique rows; duplicate tuples normalize once",
                )
            for row in table:
                if len(row) != symbol.arity:
                    raise _validation_error(
                        "tuple_arity",
                        f"every tuple of {symbol.symbol_id} must have exactly "
                        f"the declared arity {symbol.arity}",
                    )
                if any(not 0 <= coordinate < self.carrier_size for coordinate in row):
                    raise _validation_error(
                        "tuple_carrier",
                        "every tuple coordinate must be a canonical carrier "
                        "label 0..carrier_size-1",
                    )
        return self


__all__ = [
    "MAX_RELATIONAL_ARITY",
    "MAX_RELATIONAL_CARRIER",
    "MAX_RELATIONAL_OPERATION_TABLE_CELLS",
    "MAX_RELATIONAL_POLYMORPHISM_ARITY",
    "MAX_RELATIONAL_SYMBOLS",
    "MAX_RELATIONAL_TABLE_ROWS",
    "MAX_RELATIONAL_TRANSPORT_TUPLES",
    "FiniteRelationSymbol",
    "FiniteRelationalStructure",
    "RelationSymbolId",
]
