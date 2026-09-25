"""Relational homomorphism admission shared by the native and catalog paths.

The defining invariant — for every relation symbol ``R`` and every tuple
``t`` in ``R^A``, ``h(t)`` lies in ``R^B`` — is replayed exhaustively by the
kernel, so the complete transport work ``sum |R^A|`` is preflighted here
before any tuple is transported. Structural carrier-map defects and
signature mismatches are typed domain rejections; a structurally valid
request whose transport work exceeds the published envelope is a resource
admission failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._models import RelationalPolymorphism
from jacobian.math.logic.relational_structures.values import (
    MAX_PP_DEFINED_TUPLES,
    MAX_PP_EVALUATION_ASSIGNMENTS,
    MAX_PP_EVALUATION_ATOM_CHECKS,
    MAX_PP_EVALUATION_COORDINATE_WORK,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_INVARIANT_CLOSURE_OUTPUT_BYTES,
    MAX_RELATIONAL_INVARIANT_CLOSURE_TUPLES,
    MAX_RELATIONAL_INVARIANT_CLOSURE_WORK,
    MAX_RELATIONAL_OPERATION_TABLE_CELLS,
    MAX_RELATIONAL_POLYMORPHISM_ARITY,
    MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
    PPRelationAtom,
    PrimitivePositiveFormula,
)

# Exhaustive homomorphism search work: every candidate carrier map is
# replayed over every source tuple. The candidate cap bounds enumeration
# overhead (map construction is linear in the source carrier); the joint
# work cap bounds tuple replays, the dominant cost.
MAX_SEARCH_CANDIDATES = 65_536
MAX_SEARCH_TUPLE_REPLAYS = 1_048_576
MAX_HOMOMORPHISM_ENUMERATION_OUTPUT_BYTES = 8 * 1_048_576
# An induced embedding must decide membership for every tuple in every
# Cartesian relation domain, not only the sparse positive rows.
MAX_EMBEDDING_REFLECTION_CELLS = 1_048_576
# A supplied m-ary polymorphism checks every element of each relation power
# R^m. These caps independently bound relation-product enumeration and the
# coordinate operations needed to build each output tuple.
MAX_POLYMORPHISM_RELATION_COMBINATIONS = 65_536
MAX_POLYMORPHISM_COORDINATE_WORK = 1_000_000
# Complete fixed-arity family enumeration charges candidate table generation,
# all worst-case relation-product checks, and coordinatewise table work.
MAX_POLYMORPHISM_FAMILY_WORK = 8_388_608
MAX_POLYMORPHISM_FAMILY_OUTPUT_BYTES = 8 * 1_048_576
MAX_INDUCED_SUBSTRUCTURE_WORK = 81_920
MAX_RELATIONAL_REDUCT_WORK = 81_920
MAX_RELATIONAL_PRODUCT_WORK = 1_048_576


def admit_invariant_relation_closure(
    source: FiniteRelationalStructure,
    relation_arity: int,
    generator_tuples: Sequence[tuple[int, ...]],
    polymorphisms: Sequence[RelationalPolymorphism],
) -> tuple[int, int]:
    """Preflight closure state cardinality and full preservation/closure work.

    The returned values are (relation-state bound, coordinate-work bound).
    Every supplied operation is required to preserve the source relations;
    this check is charged here before any relation or power is expanded.
    """

    if not 0 <= relation_arity <= 4:
        raise OperationDomainValidationError(
            location=("relation_arity",),
            code="relational.invariant_closure.arity",
            message="the generated relation arity must be between 0 and 4",
        )
    state_count = source.carrier_size**relation_arity
    if state_count > MAX_RELATIONAL_INVARIANT_CLOSURE_TUPLES:
        raise OperationResourceAdmissionError(
            location=("relation_arity",),
            code="relational.invariant_closure.state_bound",
            message=(
                f"the generated relation has at most {state_count} tuples, exceeding "
                f"the {MAX_RELATIONAL_INVARIANT_CLOSURE_TUPLES}-tuple envelope"
            ),
        )

    # With no seeds, positive-arity operations cannot produce a tuple, so the
    # closure kernel returns immediately without scanning the ambient power.
    empty_closure = not generator_tuples

    # The queue sorts the current tuple set once per discovered row. The
    # factor eight covers up to log2(4096) comparisons for every row reference.
    work = 0 if empty_closure else 8 * state_count * state_count
    for operation in polymorphisms:
        arity = operation.arity
        table_cells, check_work = admit_polymorphism_check(source, arity)
        del table_cells
        preservation_combinations = sum(
            len(table) ** arity for table in source.relation_tables
        )
        # Each discovered tuple is combined with every possible tuple in the
        # other m-1 coordinates, in each argument position. This bounds the
        # incremental fixed-point algorithm without rescanning old products.
        closure_work = (
            0 if empty_closure else arity * arity * relation_arity * state_count**arity
        )
        work += check_work + preservation_combinations + closure_work
    retained_source_bytes = len(
        encode_strict_json({"source": source.model_dump(mode="json")})
    )
    operation_bytes = sum(
        len(encode_strict_json(operation.model_dump(mode="json")))
        for operation in polymorphisms
    )
    generator_bytes = len(encode_strict_json([list(row) for row in generator_tuples]))
    # Charge one conservative row frame per possible tuple, including the
    # widest supported relation coordinate and JSON punctuation.
    closure_bytes = state_count * (3 * relation_arity + 3) + 2
    output_bound = (
        retained_source_bytes
        + operation_bytes
        + generator_bytes
        + closure_bytes
        + 2_048
    )
    if output_bound > MAX_RELATIONAL_INVARIANT_CLOSURE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("polymorphisms",),
            code="relational.invariant_closure.output_bound",
            message=(
                f"the complete closure result has a conservative {output_bound}-byte "
                f"bound, exceeding the {MAX_RELATIONAL_INVARIANT_CLOSURE_OUTPUT_BYTES}-byte envelope"
            ),
        )
    if work > MAX_RELATIONAL_INVARIANT_CLOSURE_WORK:
        raise OperationResourceAdmissionError(
            location=("polymorphisms",),
            code="relational.invariant_closure.work_bound",
            message=(
                "complete operation-preservation and generated-relation closure "
                f"requires at most {work} coordinate steps, exceeding the "
                f"{MAX_RELATIONAL_INVARIANT_CLOSURE_WORK}-step envelope"
            ),
        )
    return state_count, work


def admit_pp_evaluation(
    structure: FiniteRelationalStructure, formula: PrimitivePositiveFormula
) -> None:
    """Admit exhaustive pp assignment replay and output materialization."""

    symbol_arities = {symbol.symbol_id: symbol.arity for symbol in structure.signature}
    for index, atom in enumerate(formula.atoms):
        if isinstance(atom, PPRelationAtom):
            arity = symbol_arities.get(atom.symbol_id)
            if arity is None:
                raise OperationDomainValidationError(
                    location=("formula", "atoms", index, "symbol_id"),
                    code="relational.pp.unknown_symbol",
                    message="every relation atom must name a symbol of the structure",
                )
            if len(atom.variables) != arity:
                raise OperationDomainValidationError(
                    location=("formula", "atoms", index, "variables"),
                    code="relational.pp.atom_arity",
                    message="relation atom variables must match the symbol arity",
                )

    size = structure.carrier_size
    assignments = 1 if formula.variable_count == 0 else size**formula.variable_count
    output_tuples = (
        1 if len(formula.free_variables) == 0 else size ** len(formula.free_variables)
    )
    atom_checks = assignments * len(formula.atoms)
    coordinate_work = assignments * formula.variable_count
    coordinate_work += assignments * sum(
        len(atom.variables) if isinstance(atom, PPRelationAtom) else 2
        for atom in formula.atoms
    )
    coordinate_work += output_tuples * len(formula.free_variables)
    if assignments > MAX_PP_EVALUATION_ASSIGNMENTS:
        raise OperationResourceAdmissionError(
            location=("formula", "variable_count"),
            code="relational.pp.assignment_bound",
            message=(
                f"complete formula evaluation has {assignments} assignments, "
                f"exceeding the {MAX_PP_EVALUATION_ASSIGNMENTS}-assignment envelope"
            ),
        )
    if atom_checks > MAX_PP_EVALUATION_ATOM_CHECKS:
        raise OperationResourceAdmissionError(
            location=("formula", "atoms"),
            code="relational.pp.atom_check_bound",
            message=(
                f"complete formula evaluation has {atom_checks} atom checks, "
                f"exceeding the {MAX_PP_EVALUATION_ATOM_CHECKS}-check envelope"
            ),
        )
    if coordinate_work > MAX_PP_EVALUATION_COORDINATE_WORK:
        raise OperationResourceAdmissionError(
            location=("formula",),
            code="relational.pp.coordinate_work_bound",
            message=(
                f"complete formula evaluation has {coordinate_work} coordinate "
                f"steps, exceeding the {MAX_PP_EVALUATION_COORDINATE_WORK}-step envelope"
            ),
        )
    if output_tuples > MAX_PP_DEFINED_TUPLES:
        raise OperationResourceAdmissionError(
            location=("formula", "free_variables"),
            code="relational.pp.output_bound",
            message=(
                f"the defined relation has at most {output_tuples} tuples, "
                f"exceeding the {MAX_PP_DEFINED_TUPLES}-tuple envelope"
            ),
        )


def candidate_space(source_size: int, target_size: int) -> int:
    """Return ``|B|^|A|``, the complete carrier-map count.

    The empty source admits exactly the empty map; a nonempty source
    into the empty carrier admits no map at all.
    """

    if source_size == 0:
        return 1
    if target_size == 0:
        return 0
    return int(pow(target_size, source_size))


def admit_polymorphism_check(
    source: FiniteRelationalStructure, arity: int
) -> tuple[int, int]:
    """Preflight complete operation-table size and relation-power work.

    Returns (operation table cells, coordinate work). No relation product or
    membership index is constructed before this estimate succeeds.
    """

    if not 1 <= arity <= MAX_RELATIONAL_POLYMORPHISM_ARITY:
        raise OperationDomainValidationError(
            location=("arity",),
            code="relational.polymorphism.arity",
            message=(
                f"polymorphism arity must lie in 1..{MAX_RELATIONAL_POLYMORPHISM_ARITY}"
            ),
        )
    table_cells = source.carrier_size**arity
    if table_cells > MAX_RELATIONAL_OPERATION_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("operation_table",),
            code="relational.polymorphism.table_bound",
            message=(
                f"the complete operation table has {table_cells} cells, exceeding "
                f"the {MAX_RELATIONAL_OPERATION_TABLE_CELLS}-cell envelope"
            ),
        )
    combinations = 0
    coordinate_work = 0
    for symbol, table in zip(source.signature, source.relation_tables, strict=True):
        count = len(table) ** arity
        combinations += count
        coordinate_work += count * symbol.arity * arity
    if combinations > MAX_POLYMORPHISM_RELATION_COMBINATIONS:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.polymorphism.relation_product_bound",
            message=(
                "complete relation products exceed the "
                f"{MAX_POLYMORPHISM_RELATION_COMBINATIONS}-combination envelope"
            ),
        )
    if coordinate_work > MAX_POLYMORPHISM_COORDINATE_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.polymorphism.coordinate_work_bound",
            message=(
                "coordinatewise operation work exceeds the "
                f"{MAX_POLYMORPHISM_COORDINATE_WORK}-step envelope"
            ),
        )
    return table_cells, coordinate_work


def admit_polymorphism_family(
    source: FiniteRelationalStructure, arity: int
) -> tuple[int, int, int, int]:
    """Preflight the complete function family, preservation replay, and output.

    Returns table cells per operation, total candidate tables, aggregate work,
    and a conservative serialized-output bound. No function table or relation
    membership index is constructed before this succeeds.
    """

    if not 1 <= arity <= MAX_RELATIONAL_POLYMORPHISM_ARITY:
        raise OperationDomainValidationError(
            location=("arity",),
            code="relational.polymorphism.arity",
            message=(
                f"polymorphism arity must lie in 1..{MAX_RELATIONAL_POLYMORPHISM_ARITY}"
            ),
        )
    carrier_size = source.carrier_size
    table_cells = carrier_size**arity
    if table_cells > MAX_RELATIONAL_OPERATION_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("operation_table",),
            code="relational.polymorphism.table_bound",
            message=(
                f"each complete operation table has {table_cells} cells, exceeding "
                f"the {MAX_RELATIONAL_OPERATION_TABLE_CELLS}-cell envelope"
            ),
        )

    # A positive-arity operation on the empty carrier is the unique empty
    # function. For other carriers, stop multiplying as soon as the family cap
    # is crossed so rejection never formats or retains an enormous n^(n^m).
    candidate_tables = (
        1
        if carrier_size == 0
        else _capped_power(
            carrier_size, table_cells, MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE
        )
    )
    if candidate_tables > MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE:
        raise OperationResourceAdmissionError(
            location=("arity",),
            code="relational.polymorphism.family_candidate_bound",
            message=(
                "the complete function space exceeds the "
                f"{MAX_RELATIONAL_POLYMORPHISM_FAMILY_SIZE}-candidate envelope"
            ),
        )

    work_per_candidate = table_cells + carrier_size + 1 + len(source.signature)
    for symbol, relation in zip(source.signature, source.relation_tables, strict=True):
        combinations = len(relation) ** arity
        # itertools.product copies one relation pool and initializes its
        # arity-sized odometer for each candidate. The loop then materializes
        # each m-row input tuple. Charge both before admitting enumeration.
        work_per_candidate += len(relation) + arity
        search_depth = len(relation).bit_length()
        per_combination_work = (
            arity
            + 1
            + symbol.arity * (3 * arity + 2)
            + search_depth * 4 * (symbol.arity + 1)
        )
        work_per_candidate += combinations * per_combination_work
    work = candidate_tables * work_per_candidate
    if work > MAX_POLYMORPHISM_FAMILY_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.polymorphism.family_work_bound",
            message=(
                f"complete family enumeration requires at most {work} table-generation, "
                f"relation-check, and coordinate steps, exceeding the "
                f"{MAX_POLYMORPHISM_FAMILY_WORK}-step envelope"
            ),
        )

    source_bytes = len(source.model_dump_json().encode("utf-8"))
    coordinate_digits = len(str(carrier_size - 1)) if carrier_size else 1
    per_table_bytes = 2 + table_cells * (coordinate_digits + 1)
    output_bound = (
        source_bytes
        + 256
        + candidate_tables * per_table_bytes
        + max(candidate_tables - 1, 0)
    )
    if output_bound > MAX_POLYMORPHISM_FAMILY_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("operation_tables",),
            code="relational.polymorphism.family_output_bound",
            message=(
                f"the complete family has a conservative {output_bound}-byte output "
                f"bound, exceeding the {MAX_POLYMORPHISM_FAMILY_OUTPUT_BYTES}-byte envelope"
            ),
        )
    return table_cells, candidate_tables, work, output_bound


def _capped_power(base: int, exponent: int, cap: int) -> int:
    """Return ``base**exponent`` exactly through cap, else ``cap + 1``."""

    result = 1
    for _ in range(exponent):
        result *= base
        if result > cap:
            return cap + 1
    return result


def core_search_work(source_size: int, transport_tuples: int) -> int:
    """Return the worst-case core-iteration replay work.

    Retraction strictly shrinks the carrier, so at most one exhaustive
    endomorphism scan runs per size from ``source_size`` down to 1; the
    transport tables only shrink along restrictions, hence the source
    count bounds every level.
    """

    return int(
        sum(pow(size, size) for size in range(1, source_size + 1))
        * max(transport_tuples, 1)
    )


def admit_homomorphism_check(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> int:
    """Preflight one exhaustive homomorphism replay; return its tuple count.

    Raises ``OperationDomainValidationError`` when the two structures do not
    share one signature or the candidate map is not a total function from the
    source carrier into the target carrier, and
    ``OperationResourceAdmissionError`` when the complete transport work
    exceeds the published envelope.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    if len(carrier_map) != source.carrier_size:
        raise OperationDomainValidationError(
            location=("carrier_map",),
            code="relational.homomorphism.carrier_map_axis",
            message=(
                "a candidate homomorphism must be a total function on the "
                f"complete source carrier; expected {source.carrier_size} "
                f"images, got {len(carrier_map)}"
            ),
        )
    for position, image in enumerate(carrier_map):
        if not isinstance(image, int) or isinstance(image, bool):
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message="carrier map images must be exact integers",
            )
        if not 0 <= image < target.carrier_size:
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message=(
                    "every carrier map image must belong to the target "
                    f"carrier 0..{target.carrier_size - 1}"
                ),
            )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    if transport_tuples > MAX_RELATIONAL_TRANSPORT_TUPLES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.transport_bound",
            message=(
                f"the exhaustive replay transports {transport_tuples} source "
                f"relation tuples, exceeding the "
                f"{MAX_RELATIONAL_TRANSPORT_TUPLES}-tuple envelope"
            ),
        )
    return transport_tuples


def embedding_reflection_cells(source: FiniteRelationalStructure) -> int:
    """Return the complete Cartesian cells replayed by an embedding."""

    return sum(
        1 if symbol.arity == 0 else source.carrier_size**symbol.arity
        for symbol in source.signature
    )


def admit_induced_substructure(
    source: FiniteRelationalStructure, inclusion: Sequence[int]
) -> tuple[int, int]:
    """Preflight induced-table restriction and a conservative JSON result bound.

    The ordered inclusion is the new-carrier-to-source map. Every source row
    and each of its coordinates are charged. The
    result repeats the source and returns a substructure no larger than it;
    therefore twice the canonical source JSON size, plus the inclusion and
    fixed framing, is a conservative serialized-output estimate.
    """

    if not isinstance(inclusion, Sequence) or isinstance(
        inclusion, (str, bytes, bytearray)
    ):
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="relational.induced_substructure.inclusion_shape",
            message="inclusion must be an ordered finite sequence of source labels",
        )
    if len(inclusion) > source.carrier_size:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="relational.induced_substructure.inclusion_size",
            message="the selected carrier cannot exceed the source carrier",
        )
    if any(
        not isinstance(label, int) or isinstance(label, bool) for label in inclusion
    ):
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="relational.induced_substructure.inclusion_label",
            message="every inclusion label must be an exact integer",
        )
    if len(set(inclusion)) != len(inclusion):
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="relational.induced_substructure.inclusion_injective",
            message="the ordered carrier selection must contain distinct labels",
        )
    if any(not 0 <= label < source.carrier_size for label in inclusion):
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="relational.induced_substructure.inclusion_range",
            message="every selected label must belong to the source carrier",
        )

    work = sum(
        len(table) * (symbol.arity + 1)
        for symbol, table in zip(source.signature, source.relation_tables, strict=True)
    )
    if work > MAX_INDUCED_SUBSTRUCTURE_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.induced_substructure.work_bound",
            message=(
                f"induced relation transport needs {work} row/coordinate visits, "
                "exceeding the "
                f"{MAX_INDUCED_SUBSTRUCTURE_WORK}-visit envelope"
            ),
        )

    source_json_bytes = len(source.model_dump_json().encode("utf-8"))
    output_bound = 2 * source_json_bytes + 4 * len(inclusion) + 256
    max_output_bytes = CanonicalLimits().max_output_bytes
    if output_bound > max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.induced_substructure.output_bound",
            message=(
                f"the conservative induced-substructure JSON bound is "
                f"{output_bound} bytes, exceeding the {max_output_bytes}-byte "
                "canonical output limit"
            ),
        )
    return work, output_bound


def admit_relational_reduct(
    source: FiniteRelationalStructure, symbol_ids: Sequence[str]
) -> tuple[tuple[int, ...], int, int]:
    """Preflight selected relation-table copying for one reduct.

    Returns source signature indices in canonical source order, total row and
    coordinate-copy work, and a conservative bound for the complete result.
    The result contains both the original source and the copied reduct.
    """

    if not isinstance(symbol_ids, Sequence) or isinstance(
        symbol_ids, (str, bytes, bytearray)
    ):
        raise OperationDomainValidationError(
            location=("symbol_ids",),
            code="relational.reduct.symbol_ids_shape",
            message="symbol_ids must be an ordered finite sequence of relation IDs",
        )
    if len(symbol_ids) > MAX_RELATIONAL_SYMBOLS:
        raise OperationDomainValidationError(
            location=("symbol_ids",),
            code="relational.reduct.symbol_ids_size",
            message="selected relation IDs exceed the signature-size bound",
        )
    if any(type(symbol_id) is not str for symbol_id in symbol_ids):
        raise OperationDomainValidationError(
            location=("symbol_ids",),
            code="relational.reduct.symbol_id_type",
            message="selected relation IDs must be exact strings",
        )
    if len(set(symbol_ids)) != len(symbol_ids):
        raise OperationDomainValidationError(
            location=("symbol_ids",),
            code="relational.reduct.symbol_ids_not_unique",
            message="selected relation IDs must be unique",
        )

    selected = set(symbol_ids)
    source_ids = tuple(symbol.symbol_id for symbol in source.signature)
    unknown = selected.difference(source_ids)
    if unknown:
        raise OperationDomainValidationError(
            location=("symbol_ids",),
            code="relational.reduct.symbol_id_unknown",
            message="every selected relation ID must belong to the source signature",
        )
    indices = tuple(
        index for index, symbol_id in enumerate(source_ids) if symbol_id in selected
    )
    work = sum(
        1 + len(source.relation_tables[index]) * (source.signature[index].arity + 1)
        for index in indices
    )
    if work > MAX_RELATIONAL_REDUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.reduct.work_bound",
            message=(
                f"reduct copying needs {work} row and coordinate visits, "
                f"exceeding the {MAX_RELATIONAL_REDUCT_WORK}-visit envelope"
            ),
        )

    source_bytes = len(source.model_dump_json().encode("utf-8"))
    # The result serializes the source once and a reduct no larger than that
    # source, plus object keys and at most eight small integer map entries.
    output_bound = 2 * source_bytes + 256 + 12 * len(indices)
    max_output_bytes = CanonicalLimits().max_output_bytes
    if output_bound > max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.reduct.output_bound",
            message=(
                f"the reduct output may need {output_bound} bytes, exceeding "
                f"the {max_output_bytes}-byte canonical output limit"
            ),
        )
    return indices, work, output_bound


def admit_relational_product(
    left: FiniteRelationalStructure, right: FiniteRelationalStructure
) -> tuple[int, int]:
    """Preflight pairwise relation rows, coordinate work, and result bytes."""
    if left.signature != right.signature:
        raise OperationDomainValidationError(
            location=("right", "signature"),
            code="relational.product.signature_mismatch",
            message="direct product factors must have identical ranked signatures",
        )
    carrier_size = left.carrier_size * right.carrier_size
    if carrier_size > MAX_RELATIONAL_CARRIER:
        raise OperationResourceAdmissionError(
            location=("product", "carrier_size"),
            code="relational.product.carrier_bound",
            message=f"Cartesian carrier has {carrier_size} labels, exceeding {MAX_RELATIONAL_CARRIER}",
        )
    row_pairs = tuple(
        len(left_table) * len(right_table)
        for left_table, right_table in zip(
            left.relation_tables, right.relation_tables, strict=True
        )
    )
    if any(rows > MAX_RELATIONAL_TABLE_ROWS for rows in row_pairs):
        raise OperationResourceAdmissionError(
            location=("product", "relation_tables"),
            code="relational.product.table_rows_bound",
            message=f"a product relation exceeds the {MAX_RELATIONAL_TABLE_ROWS}-row table bound",
        )
    work = (
        sum(
            rows * (symbol.arity + 1)
            for rows, symbol in zip(row_pairs, left.signature, strict=True)
        )
        + 2 * carrier_size
    )
    if work > MAX_RELATIONAL_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("product",),
            code="relational.product.work_bound",
            message=f"direct product needs {work} row/coordinate visits, exceeding {MAX_RELATIONAL_PRODUCT_WORK}",
        )
    product_bytes = 512 + len(left.signature) * 64
    label_width = len(str(max(0, carrier_size - 1)))
    for rows, symbol in zip(row_pairs, left.signature, strict=True):
        product_bytes += rows * (symbol.arity * (label_width + 1) + 3) + 8
    output_bound = (
        len(left.model_dump_json().encode("utf-8"))
        + len(right.model_dump_json().encode("utf-8"))
        + product_bytes
        + carrier_size * 12
    )
    maximum = CanonicalLimits().max_output_bytes
    if output_bound > maximum:
        raise OperationResourceAdmissionError(
            location=("product",),
            code="relational.product.output_bound",
            message=f"direct product result needs at most {output_bound} bytes, exceeding {maximum}",
        )
    return work, output_bound


def admit_embedding_search(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> tuple[int, int, int]:
    """Admit a complete induced-embedding search.

    In addition to positive homomorphism rows, every candidate pays for the
    full source Cartesian relation domains.  This is what distinguishes an
    induced embedding from an injective homomorphism.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    space = candidate_space(source.carrier_size, target.carrier_size)
    if space > MAX_SEARCH_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_space",
            message=(
                f"the exhaustive search spans {space} carrier maps from a "
                f"{source.carrier_size}-element source into a "
                f"{target.carrier_size}-element target, exceeding the "
                f"{MAX_SEARCH_CANDIDATES}-candidate envelope"
            ),
        )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    reflection_cells = embedding_reflection_cells(source)
    target_materialization = sum(len(table) for table in target.relation_tables)
    work = target_materialization + space * (
        max(transport_tuples, 1) + reflection_cells
    )
    if reflection_cells > MAX_EMBEDDING_REFLECTION_CELLS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.embedding.reflection_bound",
            message=(
                f"induced embedding reflection inspects {reflection_cells} "
                f"Cartesian cells, exceeding the "
                f"{MAX_EMBEDDING_REFLECTION_CELLS}-cell envelope"
            ),
        )
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.embedding.search_work",
            message=(
                f"the induced embedding search materializes {target_materialization} "
                f"target rows and replays {reflection_cells} reflection cells and "
                f"{transport_tuples} positive rows across {space} maps, exceeding "
                f"the {MAX_SEARCH_TUPLE_REPLAYS}-unit envelope"
            ),
        )
    return space, transport_tuples, reflection_cells


def admit_homomorphism_search(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> tuple[int, int]:
    """Preflight one exhaustive homomorphism search; return space and tuples.

    Returns the ``(|B|^|A| carrier-map count, source tuple count)`` pair.
    Raises ``OperationDomainValidationError`` when the two structures do
    not share one signature, and ``OperationResourceAdmissionError`` when
    the candidate space or the joint replay work exceeds the published
    search envelope.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    space = candidate_space(source.carrier_size, target.carrier_size)
    if space > MAX_SEARCH_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_space",
            message=(
                f"the exhaustive search spans {space} carrier maps from a "
                f"{source.carrier_size}-element source into a "
                f"{target.carrier_size}-element target, exceeding the "
                f"{MAX_SEARCH_CANDIDATES}-candidate envelope"
            ),
        )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    work = space * max(transport_tuples, 1)
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_work",
            message=(
                f"the exhaustive search replays {transport_tuples} source "
                f"relation tuples across {space} carrier maps, exceeding the "
                f"{MAX_SEARCH_TUPLE_REPLAYS}-replay envelope"
            ),
        )
    return space, transport_tuples


def admit_homomorphism_enumeration(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> tuple[int, int]:
    """Admit complete homomorphism enumeration including a conservative output bound."""
    space, transport_tuples = admit_homomorphism_search(source, target)
    # The result echoes both structures, so charge their exact canonical JSON
    # size as well as a conservative integer/list bound for every map.
    retained_sources = len(
        encode_strict_json(
            {
                "source": source.model_dump(mode="json"),
                "target": target.model_dump(mode="json"),
            }
        )
    )
    # The factor intentionally exceeds the largest canonical carrier label,
    # including commas, nested list framing, and map tuple framing.
    output_bound = retained_sources + space * (4 + 8 * source.carrier_size) + 128
    if output_bound > MAX_HOMOMORPHISM_ENUMERATION_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.enumeration_output",
            message=(
                f"the complete map list has a conservative {output_bound}-byte "
                f"output bound, exceeding the "
                f"{MAX_HOMOMORPHISM_ENUMERATION_OUTPUT_BYTES}-byte envelope"
            ),
        )
    return space, transport_tuples


def admit_core_computation(source: FiniteRelationalStructure) -> int:
    """Preflight one core iteration; return its worst-case replay work.

    Raises ``OperationResourceAdmissionError`` when the summed
    level-by-level endomorphism replay work exceeds the published
    envelope shared with homomorphism search.
    """

    transport_tuples = sum(len(table) for table in source.relation_tables)
    work = core_search_work(source.carrier_size, transport_tuples)
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.core.search_work",
            message=(
                f"the core iteration replays {transport_tuples} source "
                f"relation tuples across shrinking endomorphism spaces, "
                f"exceeding the {MAX_SEARCH_TUPLE_REPLAYS}-replay envelope"
            ),
        )
    return work


__all__ = [
    "MAX_EMBEDDING_REFLECTION_CELLS",
    "MAX_INDUCED_SUBSTRUCTURE_WORK",
    "MAX_POLYMORPHISM_COORDINATE_WORK",
    "MAX_POLYMORPHISM_RELATION_COMBINATIONS",
    "MAX_RELATIONAL_TRANSPORT_TUPLES",
    "MAX_SEARCH_CANDIDATES",
    "MAX_SEARCH_TUPLE_REPLAYS",
    "admit_core_computation",
    "admit_embedding_search",
    "admit_homomorphism_check",
    "admit_homomorphism_search",
    "admit_induced_substructure",
    "admit_polymorphism_check",
    "candidate_space",
    "core_search_work",
    "embedding_reflection_cells",
]
