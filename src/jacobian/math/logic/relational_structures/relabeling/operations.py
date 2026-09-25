"""Bounded exact kernels for finite relational carrier relabeling."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._models import (
    MAX_CSP_SCOPE_ENTRIES,
    FiniteCspInstance,
)
from jacobian.math.logic.relational_structures.relabeling._models import (
    CspTemplateCarrierRelabeling,
    RelationalCarrierRelabeling,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_TABLE_ROWS,
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
)

MAX_RELABELING_SORT_WORK = 1_000_000
MAX_RELABELING_SCOPE_ENTRIES = MAX_CSP_SCOPE_ENTRIES


def _admit_structure(value: object) -> FiniteRelationalStructure:
    if not isinstance(value, FiniteRelationalStructure):
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.relabeling.structure_type",
            message="source must be a finite relational structure",
        )
    try:
        return FiniteRelationalStructure.model_validate(value.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.relabeling.structure_shape",
            message="source must have canonical complete finite relation tables",
        ) from exc


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


def _admit_work(source: FiniteRelationalStructure, scope_entries: int = 0) -> None:
    if source.carrier_size > MAX_RELATIONAL_CARRIER:
        raise OperationResourceAdmissionError(
            location=("source", "carrier_size"),
            code="relational.relabeling.carrier_limit",
            message="source carrier exceeds the relabeling envelope",
        )
    row_count = 0
    sort_work = 0
    for symbol, table in zip(source.signature, source.relation_tables, strict=True):
        if len(table) > MAX_RELATIONAL_TABLE_ROWS:
            raise OperationResourceAdmissionError(
                location=("source", "relation_tables"),
                code="relational.relabeling.table_limit",
                message="a relation table exceeds the canonical row envelope",
            )
        row_count += len(table)
        # Coordinate substitutions need O(r) work per row; restoring canonical
        # tuple-set order costs O(rows log rows) comparisons in the worst case.
        sort_work += len(table) * symbol.arity * (1 + len(table).bit_length())
    if row_count > MAX_RELATIONAL_TRANSPORT_TUPLES:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.relabeling.tuple_limit",
            message="aggregate relation tuple transport exceeds the admitted bound",
        )
    if scope_entries > MAX_RELABELING_SCOPE_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("instance", "constraints"),
            code="relational.relabeling.scope_limit",
            message="constraint occurrence scopes exceed the admitted bound",
        )
    if sort_work + scope_entries > MAX_RELABELING_SORT_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "relation_tables"),
            code="relational.relabeling.work_limit",
            message="carrier relabeling exceeds the admitted tuple-sort work",
        )


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

    source = _admit_structure(source)
    mapping = _admit_permutation(old_to_new, source.carrier_size)
    _admit_work(source)
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
    scope_entries = sum(len(constraint.scope) for constraint in instance.constraints)
    _admit_work(source, scope_entries)
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
