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
from typing import Annotated, Literal, Self

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
MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE = 65_536
# Primitive-positive formula evaluation is exhaustive over assignments. These
# independent ceilings bound candidate assignments, atom replays, and the
# materialized defined relation before any Cartesian expansion.
MAX_PP_VARIABLES = 8
MAX_PP_ATOMS = 64
MAX_PP_EVALUATION_ASSIGNMENTS = 1_048_576
MAX_PP_EVALUATION_ATOM_CHECKS = 8_388_608
MAX_PP_EVALUATION_COORDINATE_WORK = 16_777_216
MAX_PP_DEFINED_TUPLES = 65_536
MAX_RELATIONAL_INVARIANT_CLOSURE_TUPLES = 4_096
MAX_RELATIONAL_INVARIANT_CLOSURE_WORK = 8_388_608
MAX_RELATIONAL_INVARIANT_CLOSURE_OUTPUT_BYTES = 8 * 1_048_576

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


class PPRelationAtom(StrictModel):
    """A relation-symbol application to variables of a pp formula."""

    kind: Literal["relation"]
    symbol_id: RelationSymbolId
    variables: tuple[StrictInt, ...] = Field(max_length=MAX_RELATIONAL_ARITY)


class PPEqualityAtom(StrictModel):
    """Logical equality between two formula variables."""

    kind: Literal["equality"]
    left: StrictInt
    right: StrictInt


PPAtom = PPRelationAtom | PPEqualityAtom


class PrimitivePositiveFormula(StrictModel):
    """A finite single-sorted pp formula in explicit variable coordinates.

    The formula is a conjunction of relation and equality atoms. Variables in
    ``free_variables`` are ordered result axes; every other declared variable
    is existentially quantified. An empty conjunction is true, which also
    allows a sentence to express existence of isolated quantified variables.
    The atom tuple is a canonical conjunction presentation: equality endpoints
    are ordered, duplicate atoms are removed, and atoms are sorted. This does
    not reorder variable axes or terms of relation atoms.
    """

    variable_count: StrictInt = Field(ge=0, le=MAX_PP_VARIABLES)
    free_variables: tuple[StrictInt, ...] = Field(
        max_length=MAX_PP_VARIABLES,
        description=(
            "Distinct declared variables in result-axis order; all other "
            "declared variables are existentially quantified."
        ),
    )
    atoms: tuple[PPAtom, ...] = Field(
        max_length=MAX_PP_ATOMS,
        description=(
            "Conjunctive relation/equality atoms. Construction sorts and "
            "deduplicates atoms and orders equality endpoints; relation "
            "argument and free-variable axis order are preserved."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def canonicalize_conjunction(cls, data: object) -> object:
        """Normalize only a bounded, fully well-shaped raw atom sequence.

        Overlong or malformed syntax is left untouched for the typed field
        validators to reject; in particular, no sorting or deduplication runs
        before the input atom-count ceiling has been checked.
        """

        if not isinstance(data, Mapping) or "atoms" not in data:
            return data
        atoms = data["atoms"]
        if not isinstance(atoms, (list, tuple)) or len(atoms) > MAX_PP_ATOMS:
            return data

        keyed_atoms: list[tuple[tuple[object, ...], object]] = []
        for atom in atoms:
            if isinstance(atom, PPRelationAtom):
                symbol_id = atom.symbol_id
                variables = atom.variables
                if type(symbol_id) is not str or any(
                    type(v) is not int for v in variables
                ):
                    return data
                key = ("relation", symbol_id, tuple(variables), 0, 0)
                value: object = {
                    "kind": "relation",
                    "symbol_id": symbol_id,
                    "variables": tuple(variables),
                }
            elif isinstance(atom, PPEqualityAtom):
                left, right = sorted((atom.left, atom.right))
                key = ("equality", "", (), left, right)
                value = {"kind": "equality", "left": left, "right": right}
            elif isinstance(atom, Mapping):
                if len(atom) > 3:
                    return data
                kind = atom.get("kind")
                if kind == "relation" and set(atom) == {
                    "kind",
                    "symbol_id",
                    "variables",
                }:
                    symbol_id = atom["symbol_id"]
                    variables = atom["variables"]
                    if (
                        type(symbol_id) is not str
                        or not isinstance(variables, (list, tuple))
                        or len(variables) > MAX_RELATIONAL_ARITY
                        or any(type(variable) is not int for variable in variables)
                    ):
                        return data
                    variables = tuple(variables)
                    key = ("relation", symbol_id, variables, 0, 0)
                    value = {
                        "kind": "relation",
                        "symbol_id": symbol_id,
                        "variables": variables,
                    }
                elif kind == "equality" and set(atom) == {
                    "kind",
                    "left",
                    "right",
                }:
                    left, right = atom["left"], atom["right"]
                    if type(left) is not int or type(right) is not int:
                        return data
                    left, right = sorted((left, right))
                    key = ("equality", "", (), left, right)
                    value = {"kind": "equality", "left": left, "right": right}
                else:
                    return data
            else:
                return data
            keyed_atoms.append((key, value))

        normalized = tuple(
            value
            for _, value in sorted(dict(keyed_atoms).items(), key=lambda item: item[0])
        )
        canonical = dict(data)
        free_variables = canonical.get("free_variables")
        if isinstance(free_variables, list) and len(free_variables) <= MAX_PP_VARIABLES:
            canonical["free_variables"] = tuple(free_variables)
        canonical["atoms"] = normalized
        return canonical

    @model_validator(mode="after")
    def require_valid_variable_axes(self) -> Self:
        if len(set(self.free_variables)) != len(self.free_variables):
            raise _validation_error(
                "pp.free_variables", "free-variable axes must be distinct"
            )
        if any(
            not 0 <= variable < self.variable_count for variable in self.free_variables
        ):
            raise _validation_error(
                "pp.free_variable_range", "free variables must name declared variables"
            )
        for atom in self.atoms:
            if isinstance(atom, PPRelationAtom):
                variables = atom.variables
            else:
                variables = (atom.left, atom.right)
            if any(not 0 <= variable < self.variable_count for variable in variables):
                raise _validation_error(
                    "pp.atom_variable_range", "every atom variable must be declared"
                )
        return self


class PPDefinedRelation(StrictModel):
    """The exact relation defined by a formula on one retained structure.

    Tuple position ``i`` denotes formula variable ``free_variables[i]``.
    Rows are sorted and unique; structure, formula, and axis remain attached
    so the finite relation is interpretable after serialization.
    """

    structure: FiniteRelationalStructure
    formula: PrimitivePositiveFormula
    tuples: tuple[tuple[StrictInt, ...], ...] = Field(max_length=MAX_PP_DEFINED_TUPLES)

    @model_validator(mode="after")
    def require_exact_relation_shape(self) -> Self:
        symbol_arities = {
            symbol.symbol_id: symbol.arity for symbol in self.structure.signature
        }
        for index, atom in enumerate(self.formula.atoms):
            if isinstance(atom, PPRelationAtom):
                arity = symbol_arities.get(atom.symbol_id)
                if arity is None:
                    raise _validation_error(
                        "pp.unknown_symbol",
                        f"formula atom {index} names no symbol in the retained structure",
                    )
                if len(atom.variables) != arity:
                    raise _validation_error(
                        "pp.atom_arity",
                        f"formula atom {index} has the wrong relation arity",
                    )
        axis_width = len(self.formula.free_variables)
        if self.tuples != tuple(sorted(set(self.tuples))):
            raise _validation_error(
                "pp.result_canonical", "defined tuples must be sorted and unique"
            )
        for row in self.tuples:
            if len(row) != axis_width or any(
                not 0 <= value < self.structure.carrier_size for value in row
            ):
                raise _validation_error(
                    "pp.result_tuple", "each defined tuple must lie on the free axes"
                )
        return self


__all__ = [
    "MAX_PP_ATOMS",
    "MAX_PP_DEFINED_TUPLES",
    "MAX_PP_EVALUATION_ASSIGNMENTS",
    "MAX_PP_EVALUATION_ATOM_CHECKS",
    "MAX_PP_EVALUATION_COORDINATE_WORK",
    "MAX_PP_VARIABLES",
    "MAX_RELATIONAL_ARITY",
    "MAX_RELATIONAL_CARRIER",
    "MAX_RELATIONAL_INVARIANT_CLOSURE_OUTPUT_BYTES",
    "MAX_RELATIONAL_INVARIANT_CLOSURE_TUPLES",
    "MAX_RELATIONAL_INVARIANT_CLOSURE_WORK",
    "MAX_RELATIONAL_OPERATION_TABLE_CELLS",
    "MAX_RELATIONAL_POLYMORPHISM_ARITY",
    "MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE",
    "MAX_RELATIONAL_SYMBOLS",
    "MAX_RELATIONAL_TABLE_ROWS",
    "MAX_RELATIONAL_TRANSPORT_TUPLES",
    "FiniteRelationSymbol",
    "FiniteRelationalStructure",
    "PPDefinedRelation",
    "PPEqualityAtom",
    "PPRelationAtom",
    "PrimitivePositiveFormula",
    "RelationSymbolId",
]
