"""Exact bounded native kernel for relational homomorphism checking."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import lcm
from typing import Literal, NoReturn

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._admission import (
    admit_core_computation,
    admit_embedding_search,
    admit_homomorphism_check,
    admit_homomorphism_enumeration,
    admit_homomorphism_search,
    admit_induced_substructure,
    admit_polymorphism_check,
    admit_relational_reduct,
)
from jacobian.math.logic.relational_structures._models import (
    MAX_CSP_CONSTRAINTS,
    MAX_CSP_SCOPE_ENTRIES,
    CspAssignmentProfile,
    CspConstraintEvaluation,
    EmbeddingSearchResult,
    FiniteCspConstraint,
    FiniteCspInstance,
    HomomorphismCheckResult,
    HomomorphismCoreResult,
    HomomorphismCountResult,
    HomomorphismEnumerationResult,
    HomomorphismSearchResult,
    HomomorphismSearchStatus,
    HomomorphismStatus,
    HomomorphismViolationWitness,
    InducedEmbeddingCheckResult,
    InducedRelationProfile,
    InducedSubstructureResult,
    RelationalPolymorphism,
    RelationalPolymorphismCheckResult,
    RelationalPolymorphismRelationProfile,
    RelationalPolymorphismStatus,
    RelationalPolymorphismWitness,
    RelationalQuotient,
    RelationalReductResult,
    SymbolTransportProfile,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)


def _admit_structure(value: object, field: str) -> FiniteRelationalStructure:
    if not isinstance(value, FiniteRelationalStructure):
        raise OperationDomainValidationError(
            location=(field,),
            code="relational.homomorphism.structure_type",
            message=f"{field} must be a finite relational structure",
        )
    try:
        return FiniteRelationalStructure.model_validate(value.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(field,),
            code="relational.homomorphism.structure_shape",
            message=f"{field} must satisfy its complete canonical relation tables",
        ) from exc


def induced_substructure(
    source: FiniteRelationalStructure, inclusion: Sequence[int]
) -> InducedSubstructureResult:
    """Return the exact induced structure on an ordered source-carrier subset.

    New carrier label ``i`` denotes ``inclusion[i]`` in the source. Every
    relation table is restricted to tuples wholly in that image and transported
    through the selected order; nullary relations retain their truth value.
    """

    source = _admit_structure(source, "source")
    inclusion = tuple(inclusion) if isinstance(inclusion, Sequence) else inclusion
    admit_induced_substructure(source, inclusion)
    source_to_induced = {
        source_label: induced_label
        for induced_label, source_label in enumerate(inclusion)
    }
    tables: list[tuple[tuple[int, ...], ...]] = []
    rows_seen = 0
    for table in source.relation_tables:
        induced_rows = []
        for row in table:
            rows_seen += 1
            if rows_seen % 4_096 == 0:
                request_checkpoint(
                    "during induced relational substructure construction"
                )
            if all(label in source_to_induced for label in row):
                induced_rows.append(tuple(source_to_induced[label] for label in row))
        tables.append(tuple(induced_rows))
    substructure = FiniteRelationalStructure(
        carrier_size=len(inclusion),
        signature=source.signature,
        relation_tables=tuple(tables),
    )
    return InducedSubstructureResult(
        source=source,
        substructure=substructure,
        inclusion=inclusion,
    )


def reduct_structure(
    source: FiniteRelationalStructure,
    symbol_ids: Sequence[str],
) -> RelationalReductResult:
    """Return the selected source relations with an explicit symbol-axis map."""

    source = _admit_structure(source, "source")
    indices, _work = admit_relational_reduct(source, symbol_ids)
    request_checkpoint("before relational reduct construction")
    reduct = FiniteRelationalStructure(
        carrier_size=source.carrier_size,
        signature=tuple(source.signature[index] for index in indices),
        relation_tables=tuple(source.relation_tables[index] for index in indices),
    )
    return RelationalReductResult._from_kernel(
        source=source,
        reduct=reduct,
        source_symbol_indices=indices,
    )


def quotient_structure(
    source: FiniteRelationalStructure,
    classes: Sequence[int],
) -> RelationalQuotient:
    """Form an exact quotient when every relation is saturated by a partition."""

    source = _admit_structure(source, "source")
    if not isinstance(classes, Sequence) or isinstance(
        classes, (str, bytes, bytearray)
    ):
        raise OperationDomainValidationError(
            location=("classes",),
            code="relational.quotient.partition_shape",
            message="classes must be a finite sequence of exact integers",
        )
    if len(classes) != source.carrier_size:
        raise OperationDomainValidationError(
            location=("classes",),
            code="relational.quotient.partition_axis",
            message="classes must give exactly one label per source element",
        )
    if any(not isinstance(label, int) or isinstance(label, bool) for label in classes):
        raise OperationDomainValidationError(
            location=("classes",),
            code="relational.quotient.partition_label",
            message="every class label must be an exact integer",
        )
    work = sum(map(len, source.relation_tables))
    if work > MAX_RELATIONAL_TRANSPORT_TUPLES:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.quotient.work_limit",
            message="quotient construction exceeds the admitted tuple-work limit",
        )
    members: dict[int, list[int]] = {}
    for element, label in enumerate(classes):
        members.setdefault(label, []).append(element)
    ordered_labels = sorted(members, key=lambda label: members[label][0])
    quotient_index = {label: index for index, label in enumerate(ordered_labels)}
    quotient_map = tuple(quotient_index[label] for label in classes)

    quotient_tables: list[tuple[tuple[int, ...], ...]] = []
    for symbol, table in zip(source.signature, source.relation_tables, strict=True):
        fibers: dict[tuple[int, ...], int] = {}
        for row in table:
            quotient_row = tuple(quotient_map[element] for element in row)
            fibers[quotient_row] = fibers.get(quotient_row, 0) + 1
        quotient_rows: list[tuple[int, ...]] = []
        for row, present_count in fibers.items():
            fiber_size = 1
            for quotient_label in row:
                fiber_size *= len(members[ordered_labels[quotient_label]])
            if present_count != fiber_size:
                raise OperationDomainValidationError(
                    location=("classes",),
                    code="relational.quotient.not_saturated",
                    message=(
                        f"partition is not a congruence for relation "
                        f"{symbol.symbol_id}: a quotient tuple has a partial "
                        "source fiber"
                    ),
                )
            quotient_rows.append(row)
        quotient_tables.append(tuple(sorted(quotient_rows)))

    quotient = FiniteRelationalStructure(
        carrier_size=len(members),
        signature=source.signature,
        relation_tables=tuple(quotient_tables),
    )
    return RelationalQuotient.model_construct(
        source=source, quotient=quotient, quotient_map=quotient_map
    )


def check_polymorphism(
    source: FiniteRelationalStructure,
    arity: int,
    operation_table: Sequence[int],
) -> RelationalPolymorphismCheckResult:
    """Check complete preservation of every finite basic relation by f:A^m→A.

    The caller's table is indexed by lexicographic tuples in A^m. For each
    relation, the kernel checks every ordered m-tuple of relation rows and
    applies f coordinatewise. Admission covers the complete relation powers
    before any row combinations or lookup indexes are expanded.
    """

    source = _admit_structure(source, "source")
    admit_polymorphism_check(source, arity, operation_table)
    operation_table = tuple(operation_table)

    carrier_size = source.carrier_size
    relation_profiles: list[RelationalPolymorphismRelationProfile] = []
    witness: RelationalPolymorphismWitness | None = None
    checked_combinations = 0
    for symbol, relation in zip(source.signature, source.relation_tables, strict=True):
        relation_set = set(relation)
        preserved = 0
        for input_rows in product(relation, repeat=arity):
            checked_combinations += 1
            if checked_combinations % 4_096 == 0:
                request_checkpoint("during relational polymorphism preservation check")
            output_row: tuple[int, ...]
            if symbol.arity == 0:
                output_row = ()
            else:
                coordinates: list[int] = []
                for coordinate in range(symbol.arity):
                    table_index = 0
                    for row in input_rows:
                        table_index = table_index * carrier_size + row[coordinate]
                    coordinates.append(operation_table[table_index])
                output_row = tuple(coordinates)
            if output_row in relation_set:
                preserved += 1
            elif witness is None:
                witness = RelationalPolymorphismWitness(
                    symbol_id=symbol.symbol_id,
                    relation_arity=symbol.arity,
                    input_rows=tuple(input_rows),
                    output_row=output_row,
                )
        relation_profiles.append(
            RelationalPolymorphismRelationProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                input_combinations=len(relation) ** arity,
                preserved_combinations=preserved,
            )
        )

    status = (
        RelationalPolymorphismStatus.NOT_POLYMORPHISM
        if witness is not None
        else RelationalPolymorphismStatus.POLYMORPHISM
    )
    polymorphism = (
        RelationalPolymorphism.model_construct(
            source=source,
            arity=arity,
            operation_table=operation_table,
        )
        if witness is None
        else None
    )
    return RelationalPolymorphismCheckResult._from_kernel(
        source=source,
        arity=arity,
        operation_table=operation_table,
        status=status,
        polymorphism=polymorphism,
        witness=witness,
        relation_profiles=tuple(relation_profiles),
    )


def check_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> HomomorphismCheckResult:
    """Decide one candidate carrier map by exhaustive invariant replay.

    For every relation symbol ``R`` and every tuple ``t`` in the complete
    source table ``R^A``, the coordinatewise image ``h(t)`` is tested for
    membership in the complete target table ``R^B``. The replay is exhaustive
    (never sampled or short-circuited), records complete per-symbol transport
    counts, and retains the first violating ``(symbol, source tuple, image
    tuple)`` witness in deterministic signature/row order. The identity map on
    any structure is a homomorphism, and the composition of two checked
    homomorphisms is again checked as a homomorphism by this same replay.
    """

    source = _admit_structure(source, "source")
    target = _admit_structure(target, "target")
    if not isinstance(carrier_map, Sequence) or isinstance(
        carrier_map, (str, bytes, bytearray)
    ):
        raise OperationDomainValidationError(
            location=("carrier_map",),
            code="relational.homomorphism.carrier_map_shape",
            message="carrier_map must be a finite sequence of exact integers",
        )
    admit_homomorphism_check(source, target, carrier_map)
    target_tables = tuple(set(table) for table in target.relation_tables)
    return _check_homomorphism_admitted(
        source, target, tuple(carrier_map), target_tables
    )


def _check_homomorphism_admitted(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    checked_map: tuple[int, ...],
    target_tables: tuple[set[tuple[int, ...]], ...],
) -> HomomorphismCheckResult:
    """Decide a map between already-admitted structures.

    Search and count callers validate the structures and candidate space once,
    then reuse the target membership indexes for each generated map. This
    avoids serializing and revalidating both complete structures per map.
    """

    witness: HomomorphismViolationWitness | None = None
    profiles: list[SymbolTransportProfile] = []
    for symbol_index, symbol in enumerate(source.signature):
        source_table = source.relation_tables[symbol_index]
        target_table = target_tables[symbol_index]
        preserved = 0
        for source_tuple in source_table:
            image_tuple = tuple(checked_map[coordinate] for coordinate in source_tuple)
            if image_tuple in target_table:
                preserved += 1
            elif witness is None:
                witness = HomomorphismViolationWitness(
                    symbol_id=symbol.symbol_id,
                    arity=symbol.arity,
                    source_tuple=source_tuple,
                    image_tuple=image_tuple,
                )
        profiles.append(
            SymbolTransportProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                source_tuples=len(source_table),
                preserved_tuples=preserved,
            )
        )

    status = (
        HomomorphismStatus.NOT_HOMOMORPHISM
        if witness is not None
        else HomomorphismStatus.HOMOMORPHISM
    )
    return HomomorphismCheckResult._from_kernel(
        status=status,
        source=source,
        target=target,
        carrier_map=checked_map,
        witness=witness,
        symbol_profiles=tuple(profiles),
    )


def csp_instance_to_source_structure(
    instance: FiniteCspInstance,
) -> FiniteRelationalStructure:
    """Convert a CSP instance to its canonical source relational structure.

    A relation tuple records a constraint's ordered variable scope. Repeated
    occurrences with the same symbol and scope collapse in the mathematical
    relation table, while their distinct IDs remain in the input instance.
    For every assignment into the template, satisfying all occurrences is
    equivalent to preserving every source relation tuple.
    """

    if not isinstance(instance, FiniteCspInstance):
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.instance_type",
            message="instance must be a finite CSP instance",
        )
    _preflight_csp_instance(instance)
    try:
        admitted = FiniteCspInstance.model_validate(instance.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.instance_shape",
            message="instance must have valid variables, constraints, and template relations",
        ) from exc
    table_by_symbol: dict[str, set[tuple[int, ...]]] = {
        symbol.symbol_id: set() for symbol in admitted.template.signature
    }
    for constraint in admitted.constraints:
        table_by_symbol[constraint.symbol_id].add(constraint.scope)
    return FiniteRelationalStructure(
        carrier_size=admitted.variable_count,
        signature=admitted.template.signature,
        relation_tables=tuple(
            tuple(sorted(table_by_symbol[symbol.symbol_id]))
            for symbol in admitted.template.signature
        ),
    )


def profile_csp_assignment(
    instance: FiniteCspInstance,
    assignment: Sequence[int],
) -> CspAssignmentProfile:
    """Evaluate a complete assignment at every named CSP constraint.

    This is a one-map check, not an unsatisfiability search. All occurrences
    are evaluated in declaration order so duplicate constraints and the first
    violated occurrence remain observable. The complete canonical source
    structure is available separately via ``csp.instance.to_source_structure``.
    """

    if not isinstance(instance, FiniteCspInstance):
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.instance_type",
            message="instance must be a finite CSP instance",
        )
    _preflight_csp_instance(instance)
    if (
        not isinstance(assignment, Sequence)
        or isinstance(assignment, (str, bytes, bytearray))
        or len(assignment) > MAX_RELATIONAL_CARRIER
        or any(type(value) is not int for value in assignment)
    ):
        raise OperationDomainValidationError(
            location=("assignment",),
            code="relational.csp.assignment_shape",
            message="assignment must be a bounded tuple of exact integer labels",
        )
    assignment = tuple(assignment)
    try:
        admitted = FiniteCspInstance.model_validate(instance.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.instance_shape",
            message="instance must have valid variables, constraints, and template relations",
        ) from exc
    if len(assignment) != admitted.variable_count or any(
        not 0 <= value < admitted.template.carrier_size for value in assignment
    ):
        raise OperationDomainValidationError(
            location=("assignment",),
            code="relational.csp.assignment_shape",
            message="assignment must contain one in-range template value per variable",
        )
    instance = admitted

    symbol_index = {
        symbol.symbol_id: index
        for index, symbol in enumerate(instance.template.signature)
    }
    relation_sets = tuple(map(set, instance.template.relation_tables))
    evaluations_list: list[CspConstraintEvaluation] = []
    for constraint in instance.constraints:
        request_checkpoint("during CSP assignment profile evaluation")
        target_tuple = tuple(assignment[variable] for variable in constraint.scope)
        evaluations_list.append(
            CspConstraintEvaluation(
                constraint_id=constraint.constraint_id,
                symbol_id=constraint.symbol_id,
                scope=constraint.scope,
                target_tuple=target_tuple,
                allowed=target_tuple
                in relation_sets[symbol_index[constraint.symbol_id]],
            )
        )
    evaluations = tuple(evaluations_list)
    first_violation = next((item for item in evaluations if not item.allowed), None)
    return CspAssignmentProfile(
        status="NOT_A_SOLUTION" if first_violation is not None else "SOLUTION",
        instance=instance,
        assignment=assignment,
        evaluations=evaluations,
        first_violation=first_violation,
    )


def _preflight_csp_instance(instance: FiniteCspInstance) -> None:
    """Bound native values before recursively copying them for revalidation."""

    template = instance.template
    constraints = instance.constraints
    if (
        type(template) is not FiniteRelationalStructure
        or type(instance.variable_count) is not int
        or not 0 <= instance.variable_count <= MAX_RELATIONAL_CARRIER
        or not isinstance(constraints, tuple)
        or len(constraints) > MAX_CSP_CONSTRAINTS
    ):
        _raise_invalid_csp_instance()
    if (
        type(template.carrier_size) is not int
        or not 0 <= template.carrier_size <= MAX_RELATIONAL_CARRIER
        or not isinstance(template.signature, tuple)
        or len(template.signature) > MAX_RELATIONAL_SYMBOLS
        or not isinstance(template.relation_tables, tuple)
        or len(template.relation_tables) != len(template.signature)
    ):
        _raise_invalid_csp_instance()
    template_rows = 0
    symbol_arities: dict[str, int] = {}
    for symbol, table in zip(template.signature, template.relation_tables, strict=True):
        if (
            type(symbol) is not FiniteRelationSymbol
            or type(symbol.symbol_id) is not str
            or type(symbol.arity) is not int
            or not 0 <= symbol.arity <= MAX_RELATIONAL_ARITY
            or not isinstance(table, tuple)
            or len(table) > MAX_RELATIONAL_TABLE_ROWS
        ):
            _raise_invalid_csp_instance()
        symbol_arities[symbol.symbol_id] = symbol.arity
        template_rows += len(table)
        if template_rows > MAX_RELATIONAL_TRANSPORT_TUPLES:
            _raise_invalid_csp_instance()
        for row in table:
            if (
                not isinstance(row, tuple)
                or len(row) != symbol.arity
                or any(
                    type(value) is not int or not 0 <= value < template.carrier_size
                    for value in row
                )
            ):
                _raise_invalid_csp_instance()
    scope_entries = 0
    for constraint in constraints:
        if (
            type(constraint) is not FiniteCspConstraint
            or type(constraint.symbol_id) is not str
            or not isinstance(constraint.scope, tuple)
            or len(constraint.scope) > MAX_RELATIONAL_ARITY
        ):
            _raise_invalid_csp_instance()
        arity = symbol_arities.get(constraint.symbol_id)
        if arity is None or len(constraint.scope) != arity:
            _raise_invalid_csp_instance()
        scope_entries += len(constraint.scope)
        if scope_entries > MAX_CSP_SCOPE_ENTRIES:
            _raise_invalid_csp_instance()
        if any(
            type(variable) is not int or not 0 <= variable < instance.variable_count
            for variable in constraint.scope
        ):
            _raise_invalid_csp_instance()


def _raise_invalid_csp_instance() -> NoReturn:
    raise OperationDomainValidationError(
        location=("instance",),
        code="relational.csp.instance_shape",
        message="instance must satisfy the bounded finite CSP structure contract",
    )


def search_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> HomomorphismSearchResult:
    """Search two structures for a homomorphism by exhaustive replay.

    Candidate carrier maps run in lexicographic order (source label 0
    varying slowest) through the complete ``|B|^|A|`` space admitted up
    front. Every candidate is decided by the reused homomorphism replay,
    so a FOUND map carries its complete check and an EXHAUSTED scan is
    a proved negative: every map was examined and none transports.
    """

    source = _admit_structure(source, "source")
    target = _admit_structure(target, "target")
    total_candidates, _transport_tuples = admit_homomorphism_search(source, target)
    found = _first_homomorphism(
        source, target, total_candidates, require_injective=False
    )
    if found is not None:
        check, examined = found
        return HomomorphismSearchResult._from_kernel(
            status=HomomorphismSearchStatus.FOUND,
            source=source,
            target=target,
            check=check,
            candidates_examined=examined,
            total_candidates=total_candidates,
        )
    return HomomorphismSearchResult._from_kernel(
        status=HomomorphismSearchStatus.EXHAUSTED,
        source=source,
        target=target,
        check=None,
        candidates_examined=total_candidates,
        total_candidates=total_candidates,
    )


def _first_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    total_candidates: int,
    *,
    require_injective: bool,
) -> tuple[HomomorphismCheckResult, int] | None:
    """Scan carrier maps in lexicographic order for the first replay that
    transports, optionally requiring distinct images.

    Returns the winning check with its one-based examination count, or
    ``None`` after a complete scan. Shared by homomorphism and embedding
    search so both see identical order and receipts.
    """

    target_tables = tuple(set(table) for table in target.relation_tables)
    examined = 0
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during homomorphism search enumeration")
        if require_injective and len(set(candidate)) != source.carrier_size:
            continue
        check = _check_homomorphism_admitted(source, target, candidate, target_tables)
        if check.status is HomomorphismStatus.HOMOMORPHISM:
            return check, examined
    if examined != total_candidates:
        raise RuntimeError("homomorphism search did not scan its admitted space")
    return None


def _check_embedding(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: tuple[int, ...],
    *,
    target_tables: tuple[set[tuple[int, ...]], ...] | None = None,
    source_tables: tuple[set[tuple[int, ...]], ...] | None = None,
) -> InducedEmbeddingCheckResult:
    """Replay the induced-substructure invariant for one injective map.

    Unlike homomorphism checking, this scans the complete Cartesian domain of
    every relation.  With an injective map, membership of a target tuple in
    the image has a unique source preimage, so the two truth values must agree.
    """

    admit_homomorphism_check(source, target, carrier_map)
    if len(set(carrier_map)) != len(carrier_map):
        raise OperationDomainValidationError(
            location=("carrier_map",),
            code="relational.embedding.injectivity",
            message="an induced embedding must be injective",
        )
    target_tables = target_tables or tuple(
        set(table) for table in target.relation_tables
    )
    source_sets = source_tables or tuple(set(table) for table in source.relation_tables)
    witness: HomomorphismViolationWitness | None = None
    witness_kind: Literal["PRESERVATION", "REFLECTION"] = "PRESERVATION"
    profiles: list[SymbolTransportProfile] = []
    reflection_profiles: list[InducedRelationProfile] = []
    for symbol_index, symbol in enumerate(source.signature):
        source_table = source_sets[symbol_index]
        target_table = target_tables[symbol_index]
        preserved = 0
        matching = 0
        for coordinates in product(range(source.carrier_size), repeat=symbol.arity):
            source_member = coordinates in source_table
            image = tuple(carrier_map[index] for index in coordinates)
            target_member = image in target_table
            if source_member and target_member:
                preserved += 1
            if source_member == target_member:
                matching += 1
            elif witness is None and source_member != target_member:
                witness = HomomorphismViolationWitness(
                    symbol_id=symbol.symbol_id,
                    arity=symbol.arity,
                    source_tuple=tuple(coordinates),
                    image_tuple=image,
                )
                if target_member and not source_member:
                    witness_kind = "REFLECTION"
        profiles.append(
            SymbolTransportProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                source_tuples=len(source_table),
                preserved_tuples=preserved,
            )
        )
        reflection_profiles.append(
            InducedRelationProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                relation_cells=(
                    1 if symbol.arity == 0 else source.carrier_size**symbol.arity
                ),
                matching_cells=matching,
            )
        )
    status = (
        HomomorphismStatus.NOT_HOMOMORPHISM
        if witness is not None
        else HomomorphismStatus.HOMOMORPHISM
    )
    return InducedEmbeddingCheckResult._from_induced_kernel(
        status=status,
        source=source,
        target=target,
        carrier_map=carrier_map,
        witness=witness,
        witness_kind=witness_kind,
        symbol_profiles=tuple(profiles),
        reflection_profiles=tuple(reflection_profiles),
    )


def search_embedding(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> EmbeddingSearchResult:
    """Search for an induced (reflecting) injective embedding."""

    source = _admit_structure(source, "source")
    target = _admit_structure(target, "target")
    total_candidates, _transport_tuples, _reflection_cells = admit_embedding_search(
        source, target
    )
    found = _first_embedding(source, target, total_candidates)
    if found is not None:
        check, examined = found
        return EmbeddingSearchResult._from_kernel(
            status=HomomorphismSearchStatus.FOUND,
            source=source,
            target=target,
            check=check,
            candidates_examined=examined,
            total_candidates=total_candidates,
        )
    return EmbeddingSearchResult._from_kernel(
        status=HomomorphismSearchStatus.EXHAUSTED,
        source=source,
        target=target,
        check=None,
        candidates_examined=total_candidates,
        total_candidates=total_candidates,
    )


def _first_embedding(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    total_candidates: int,
) -> tuple[InducedEmbeddingCheckResult, int] | None:
    target_tables = tuple(set(table) for table in target.relation_tables)
    source_tables = tuple(set(table) for table in source.relation_tables)
    examined = 0
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during induced embedding search enumeration")
        if len(set(candidate)) != source.carrier_size:
            continue
        check = _check_embedding(
            source,
            target,
            tuple(candidate),
            target_tables=target_tables,
            source_tables=source_tables,
        )
        if check.status is HomomorphismStatus.HOMOMORPHISM:
            return check, examined
    if examined != total_candidates:
        raise RuntimeError("embedding search did not scan its admitted space")
    return None


def count_homomorphisms(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> HomomorphismCountResult:
    """Count every homomorphism by exhaustive replay without early stopping.

    The same admitted space the search scans is replayed completely:
    every carrier map is decided by the reused homomorphism replay and
    transporting maps are counted. The count is exact and complete.
    """

    source = _admit_structure(source, "source")
    target = _admit_structure(target, "target")
    total_candidates, _transport_tuples = admit_homomorphism_search(source, target)
    count = 0
    examined = 0
    target_tables = tuple(set(table) for table in target.relation_tables)
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during homomorphism count enumeration")
        if _check_homomorphism_admitted(
            source, target, candidate, target_tables
        ).status is (HomomorphismStatus.HOMOMORPHISM):
            count += 1
    if examined != total_candidates:
        raise RuntimeError("homomorphism count did not scan its admitted space")
    return HomomorphismCountResult._from_kernel(
        source=source,
        target=target,
        count=count,
        total_candidates=total_candidates,
    )


def enumerate_homomorphisms(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> HomomorphismEnumerationResult:
    """Return every relation-preserving carrier map in canonical order."""
    source = _admit_structure(source, "source")
    target = _admit_structure(target, "target")
    total_candidates, _transport_tuples = admit_homomorphism_enumeration(source, target)
    target_tables = tuple(set(table) for table in target.relation_tables)
    maps: list[tuple[int, ...]] = []
    examined = 0
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during homomorphism enumeration")
        if (
            _check_homomorphism_admitted(
                source, target, candidate, target_tables
            ).status
            is HomomorphismStatus.HOMOMORPHISM
        ):
            maps.append(candidate)
    if examined != total_candidates:
        raise RuntimeError("homomorphism enumeration did not scan its admitted space")
    return HomomorphismEnumerationResult._from_kernel(
        source=source,
        target=target,
        carrier_maps=tuple(maps),
        total_candidates=total_candidates,
    )


def _induced_substructure(
    structure: FiniteRelationalStructure, image: tuple[int, ...]
) -> FiniteRelationalStructure:
    """Restrict to the sorted image labels with canonical relabeling.

    Core label ``c`` denotes image label ``image[c]``; a table tuple
    survives exactly when every coordinate lies in the image, relabeled
    by rank. Tables stay canonical: filtering and relabeling preserve
    strictly increasing unique rows.
    """

    rank = {label: position for position, label in enumerate(image)}
    tables = tuple(
        tuple(
            tuple(rank[coordinate] for coordinate in row)
            for row in table
            if all(coordinate in rank for coordinate in row)
        )
        for table in structure.relation_tables
    )
    return FiniteRelationalStructure(
        carrier_size=len(image),
        signature=structure.signature,
        relation_tables=tables,
    )


def _idempotent_retraction(
    composed: tuple[int, ...],
    inclusion: tuple[int, ...],
    source_size: int,
) -> tuple[int, ...]:
    """Return the core-label retraction induced by an endomorphism onto the core.

    ``composed`` is an endomorphism of the source whose image is exactly the
    strictly increasing ``inclusion`` image. On that image it is a bijection,
    so a power of it acts as the identity there while keeping the same image;
    that power is an idempotent endomorphism, and reading its values as core
    labels gives the retraction with ``retraction[inclusion[c]] == c``.
    """

    if not inclusion:
        return ()
    position = {label: index for index, label in enumerate(inclusion)}
    permutation = tuple(position[composed[label]] for label in inclusion)
    seen = [False] * len(inclusion)
    order = 1
    for start in range(len(inclusion)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = permutation[node]
            length += 1
        order = lcm(order, length)

    def compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(first[second[label]] for label in range(source_size))

    power = tuple(range(source_size))
    base = composed
    exponent = order
    while exponent:
        if exponent & 1:
            power = compose(power, base)
        base = compose(base, base)
        exponent >>= 1
    return tuple(position[power[label]] for label in range(source_size))


def compute_core(
    source: FiniteRelationalStructure,
) -> HomomorphismCoreResult:
    """Compute the minimal retract with its witnessing maps.

    Each level scans endomorphisms in lexicographic order; the first
    map with a smaller image restricts the structure to its sorted
    image with canonical relabeling, and the scan repeats. The final
    level examines every endomorphism and finds no smaller image, so
    the retained structure is minimal. The composed retraction is
    replayed once through the reused check before construction.
    """

    source = _admit_structure(source, "source")
    admit_core_computation(source)
    current = source
    inclusion = tuple(range(source.carrier_size))
    retraction = tuple(range(source.carrier_size))
    scanned = 0
    while True:
        size = current.carrier_size
        step: tuple[int, ...] | None = None
        target_tables = tuple(set(table) for table in current.relation_tables)
        for candidate in product(range(size), repeat=size):
            scanned += 1
            if scanned % 4_096 == 0:
                request_checkpoint("during core endomorphism enumeration")
            check = _check_homomorphism_admitted(
                current, current, candidate, target_tables
            )
            if (
                check.status is HomomorphismStatus.HOMOMORPHISM
                and len(set(candidate)) < size
            ):
                step = tuple(candidate)
                break
        if step is None:
            break
        image = tuple(sorted(set(step)))
        relabel = {label: position for position, label in enumerate(image)}
        retraction = tuple(relabel[step[label]] for label in retraction)
        inclusion = tuple(inclusion[label] for label in image)
        current = _induced_substructure(current, image)
    composed = tuple(inclusion[label] for label in retraction)
    final = check_homomorphism(source, source, composed)
    if final.status is not HomomorphismStatus.HOMOMORPHISM:
        raise RuntimeError("composed core retraction is not an endomorphism")
    if tuple(sorted(set(composed))) != inclusion:
        raise RuntimeError("composed core retraction has the wrong image")
    retraction = _idempotent_retraction(composed, inclusion, source.carrier_size)
    if not all(
        retraction[inclusion[core_label]] == core_label
        for core_label in range(len(inclusion))
    ):
        raise RuntimeError("core retraction does not split its inclusion")
    return HomomorphismCoreResult._from_kernel(
        source=source,
        core=current,
        inclusion=inclusion,
        retraction=retraction,
    )


__all__ = [
    "check_homomorphism",
    "check_polymorphism",
    "compute_core",
    "count_homomorphisms",
    "csp_instance_to_source_structure",
    "induced_substructure",
    "quotient_structure",
    "reduct_structure",
    "search_embedding",
    "search_homomorphism",
]
