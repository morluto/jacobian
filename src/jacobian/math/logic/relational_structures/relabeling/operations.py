"""Bounded exact kernels for finite relational carrier relabeling."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._models import (
    MAX_CSP_CONSTRAINTS,
    MAX_CSP_SCOPE_ENTRIES,
    FiniteCspInstance,
)
from jacobian.math.logic.relational_structures.relabeling._models import (
    CspTemplateCarrierRelabeling,
    RelationalCarrierRelabeling,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
)

MAX_RELABELING_SORT_WORK = 1_000_000
MAX_RELABELING_SCOPE_ENTRIES = MAX_CSP_SCOPE_ENTRIES


def _preflight_raw_structure(
    value: FiniteRelationalStructure,
    location: tuple[str, ...],
    *,
    scope_entries: int = 0,
) -> None:
    """Bound raw relation-table work before any recursive ``model_dump``.

    A native caller can bypass field validation with ``model_construct``, so
    ``model_dump`` would otherwise expand and canonicalize unbounded tables
    before the resource envelope is checked. Counting the raw rows and tuple
    sort work directly on the unexpanded attributes keeps admission ahead of
    that expansion. Canonical tables are never larger than their raw source,
    so the bound computed here still holds for the validated value.
    """

    carrier_size = getattr(value, "carrier_size", None)
    if isinstance(carrier_size, bool) or not isinstance(carrier_size, int):
        raise OperationDomainValidationError(
            location=(*location, "carrier_size"),
            code="relational.relabeling.carrier_shape",
            message="the relation carrier size must be an exact integer",
        )
    if carrier_size < 0:
        raise OperationDomainValidationError(
            location=(*location, "carrier_size"),
            code="relational.relabeling.carrier_shape",
            message="the relation carrier size must be nonnegative",
        )
    if carrier_size > MAX_RELATIONAL_CARRIER:
        raise OperationResourceAdmissionError(
            location=(*location, "carrier_size"),
            code="relational.relabeling.carrier_limit",
            message="source carrier exceeds the relabeling envelope",
        )

    signature = getattr(value, "signature", None)
    tables = getattr(value, "relation_tables", None)
    if not isinstance(signature, (list, tuple)) or not isinstance(
        tables, (list, tuple)
    ):
        raise OperationDomainValidationError(
            location=location,
            code="relational.relabeling.structure_shape",
            message="a relation structure must carry an ordered signature and tables",
        )
    if len(signature) > MAX_RELATIONAL_SYMBOLS:
        raise OperationResourceAdmissionError(
            location=(*location, "signature"),
            code="relational.relabeling.symbol_limit",
            message="the relation signature exceeds the admitted symbol envelope",
        )
    if len(tables) != len(signature):
        raise OperationDomainValidationError(
            location=(*location, "relation_tables"),
            code="relational.relabeling.structure_shape",
            message="exactly one complete tuple table is required per relation symbol",
        )

    row_count = 0
    sort_work = 0
    for symbol, table in zip(signature, tables, strict=True):
        if not isinstance(table, (list, tuple)):
            raise OperationDomainValidationError(
                location=(*location, "relation_tables"),
                code="relational.relabeling.structure_shape",
                message="every relation table must be an ordered sequence of tuples",
            )
        if len(table) > MAX_RELATIONAL_TABLE_ROWS:
            raise OperationResourceAdmissionError(
                location=(*location, "relation_tables"),
                code="relational.relabeling.table_limit",
                message="a relation table exceeds the canonical row envelope",
            )
        row_count += len(table)
        if row_count > MAX_RELATIONAL_TRANSPORT_TUPLES:
            raise OperationResourceAdmissionError(
                location=(*location, "relation_tables"),
                code="relational.relabeling.tuple_limit",
                message="aggregate relation tuple transport exceeds the admitted bound",
            )
        # Coordinate substitutions need O(r) work per row; restoring canonical
        # tuple-set order costs O(rows log rows) comparisons in the worst case.
        arity = getattr(symbol, "arity", 0)
        if isinstance(arity, bool) or not isinstance(arity, int):
            arity = 0
        sort_work += len(table) * arity * (1 + len(table).bit_length())
    if sort_work + scope_entries > MAX_RELABELING_SORT_WORK:
        raise OperationResourceAdmissionError(
            location=(*location, "relation_tables"),
            code="relational.relabeling.work_limit",
            message="carrier relabeling exceeds the admitted tuple-sort work",
        )


def _admit_structure(
    value: object, location: tuple[str, ...]
) -> FiniteRelationalStructure:
    if not isinstance(value, FiniteRelationalStructure):
        raise OperationDomainValidationError(
            location=location,
            code="relational.relabeling.structure_type",
            message="source must be a finite relational structure",
        )
    _preflight_raw_structure(value, location)
    try:
        return FiniteRelationalStructure.model_validate(value.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=location,
            code="relational.relabeling.structure_shape",
            message="source must have canonical complete finite relation tables",
        ) from exc


def _preflight_csp_constraints(instance: FiniteCspInstance) -> int:
    """Bound raw constraint occurrences before ``model_dump`` expands them."""

    constraints = getattr(instance, "constraints", None)
    if not isinstance(constraints, (list, tuple)):
        raise OperationDomainValidationError(
            location=("instance", "constraints"),
            code="relational.relabeling.csp_shape",
            message="instance must carry ordered constraint occurrences",
        )
    if len(constraints) > MAX_CSP_CONSTRAINTS:
        raise OperationResourceAdmissionError(
            location=("instance", "constraints"),
            code="relational.relabeling.scope_limit",
            message="constraint occurrence scopes exceed the admitted bound",
        )
    scope_entries = 0
    for constraint in constraints:
        scope = getattr(constraint, "scope", None)
        if not isinstance(scope, (list, tuple)):
            raise OperationDomainValidationError(
                location=("instance", "constraints"),
                code="relational.relabeling.csp_shape",
                message="every constraint occurrence must carry an ordered scope",
            )
        scope_entries += len(scope)
        if scope_entries > MAX_RELABELING_SCOPE_ENTRIES:
            raise OperationResourceAdmissionError(
                location=("instance", "constraints"),
                code="relational.relabeling.scope_limit",
                message="constraint occurrence scopes exceed the admitted bound",
            )
    return scope_entries


def _admit_permutation(value: object, carrier_size: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise OperationDomainValidationError(
            location=("old_to_new",),
            code="relational.relabeling.map_shape",
            message="old_to_new must be a finite integer sequence",
        )
    mapping = tuple(value)
    if len(mapping) != carrier_size:
        raise OperationDomainValidationError(
            location=("old_to_new",),
            code="relational.relabeling.map_axis",
            message="old_to_new must contain one image per source carrier element",
        )
    if any(not isinstance(label, int) or isinstance(label, bool) for label in mapping):
        raise OperationDomainValidationError(
            location=("old_to_new",),
            code="relational.relabeling.map_label",
            message="every carrier image must be an exact integer",
        )
    if any(not 0 <= label < carrier_size for label in mapping):
        raise OperationDomainValidationError(
            location=("old_to_new",),
            code="relational.relabeling.map_range",
            message="every carrier image must belong to the target carrier",
        )
    if len(set(mapping)) != carrier_size:
        raise OperationDomainValidationError(
            location=("old_to_new",),
            code="relational.relabeling.map_bijection",
            message="a carrier relabeling must be bijective",
        )
    return mapping


def _transport_structure(
    source: FiniteRelationalStructure, old_to_new: tuple[int, ...]
) -> FiniteRelationalStructure:
    tables: list[tuple[tuple[int, ...], ...]] = []
    visited = 0
    for table in source.relation_tables:
        rows = []
        for row in table:
            visited += 1
            if visited % 1024 == 0:
                request_checkpoint("during relational carrier relabeling")
            rows.append(tuple(old_to_new[label] for label in row))
        tables.append(tuple(sorted(rows)))
    return FiniteRelationalStructure.model_construct(
        carrier_size=source.carrier_size,
        signature=source.signature,
        relation_tables=tuple(tables),
    )


def _inverse(mapping: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(mapping)
    for old, new in enumerate(mapping):
        inverse[new] = old
    return tuple(inverse)


def relabel_structure_carrier(
    source: FiniteRelationalStructure, old_to_new: Sequence[int]
) -> RelationalCarrierRelabeling:
    """Apply a bijection to every coordinate in all relation tuple tables."""

    source = _admit_structure(source, ("source",))
    mapping = _admit_permutation(old_to_new, source.carrier_size)
    request_checkpoint("before relational carrier relabeling")
    target = _transport_structure(source, mapping)
    return RelationalCarrierRelabeling.model_construct(
        source=source,
        target=target,
        old_to_new=mapping,
        new_to_old=_inverse(mapping),
    )


def relabel_csp_template_carrier(
    instance: FiniteCspInstance, old_to_new: Sequence[int]
) -> CspTemplateCarrierRelabeling:
    """Transport a CSP template while retaining every constraint occurrence."""

    if not isinstance(instance, FiniteCspInstance):
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.relabeling.csp_type",
            message="instance must be a finite CSP instance",
        )
    scope_entries = _preflight_csp_constraints(instance)
    if isinstance(instance.template, FiniteRelationalStructure):
        _preflight_raw_structure(
            instance.template,
            ("instance", "template"),
            scope_entries=scope_entries,
        )
    try:
        instance = FiniteCspInstance.model_validate(instance.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.relabeling.csp_shape",
            message="instance must have valid variables and ordered constraint occurrences",
        ) from exc
    # Revalidating the typed instance above also admits its exact template.
    # Reuse that canonical nested value instead of validating its tables twice.
    source = instance.template
    mapping = _admit_permutation(old_to_new, source.carrier_size)
    request_checkpoint("before CSP template carrier relabeling")
    target_template = _transport_structure(source, mapping)
    target = FiniteCspInstance.model_construct(
        template=target_template,
        variable_count=instance.variable_count,
        constraints=instance.constraints,
    )
    return CspTemplateCarrierRelabeling.model_construct(
        source=instance,
        target=target,
        old_to_new=mapping,
        new_to_old=_inverse(mapping),
    )


__all__ = [
    "MAX_RELABELING_SCOPE_ENTRIES",
    "MAX_RELABELING_SORT_WORK",
    "relabel_csp_template_carrier",
    "relabel_structure_carrier",
]
