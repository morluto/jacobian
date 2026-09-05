"""Cyclic prefix-sum residue profile kernel."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian._exact import CanonicalInteger
from jacobian.canonical import (
    CanonicalizationError,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    MAX_MODULUS_DIGITS,
    MAX_SEQUENCE_LENGTH,
    MAX_SEQUENCING_FORBIDDEN_VALUES,
    MAX_SEQUENCING_PERMUTATION_NODES,
    MAX_SEQUENCING_SOURCE_ITEMS,
    CyclicPrefixSumResidueProfileResult,
    FiniteAbelianSequencingSource,
    ForbiddenPrefixSequencingResult,
    PrefixSumResidueRow,
    SequencingPrefixSum,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._sequencing_kernel import (
    SequencingKernelResult,
    search_forbidden_prefix_sequencing,
)
from jacobian.math.combinatorics.additive.values import IndexedIntegerSequence

__all__ = [
    "compute_cyclic_prefix_sum_residue_profile",
    "search_forbidden_prefix_cyclic_ordering",
]

MAX_WORK_UNITS = 50_000_000


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    sequence: tuple[int, ...]
    modulus: int


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _admit(
    sequence: IndexedIntegerSequence,
    modulus: CanonicalInteger,
) -> _AdmissionPlan:
    """Validate the complete native and MCP execution envelope once."""
    if not isinstance(sequence, IndexedIntegerSequence):
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.sequence_type",
            "sequence must be an indexed integer sequence",
        )
    if type(modulus) is not str:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_type",
            "modulus must be a canonical integer string",
        )
    modulus_digits = len(modulus.lstrip("-"))
    if modulus_digits > MAX_MODULUS_DIGITS:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_digits",
            f"modulus may contain at most {MAX_MODULUS_DIGITS} digits",
        )
    try:
        modulus_value = parse_canonical_integer(modulus)
    except CanonicalizationError:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_format",
            "modulus must be a canonical integer string",
        )
    if modulus_value <= 0:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_domain",
            "modulus must be positive",
        )

    item_count = len(sequence.items)
    if item_count > MAX_SEQUENCE_LENGTH:
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.sequence_length",
            f"sequence may contain at most {MAX_SEQUENCE_LENGTH:,} items",
        )
    maximum_item_digits = max(
        (len(item.lstrip("-")) for item in sequence.items),
        default=1,
    )
    work = item_count * max(maximum_item_digits, modulus_digits)
    if work > MAX_WORK_UNITS:
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.work_bound",
            "prefix-sum modular arithmetic exceeds the admitted work bound",
        )

    return _AdmissionPlan(sequence=sequence.as_int_tuple(), modulus=modulus_value)


def compute_cyclic_prefix_sum_residue_profile(
    sequence: IndexedIntegerSequence,
    modulus: CanonicalInteger,
) -> CyclicPrefixSumResidueProfileResult:
    """Return the complete partition of prefix positions by residue.

    For each prefix position k (1-indexed), compute the prefix sum
    S_k = a_1 + ... + a_k mod m and group positions by their residue.
    """
    plan = _admit(sequence, modulus)
    residue_to_positions: dict[int, list[int]] = {}
    running = 0
    for k, value in enumerate(plan.sequence, start=1):
        running = (running + value) % plan.modulus
        if running not in residue_to_positions:
            residue_to_positions[running] = []
        residue_to_positions[running].append(k)

    rows = [
        PrefixSumResidueRow(
            residue=format_canonical_integer(res), positions=tuple(positions)
        )
        for res, positions in sorted(residue_to_positions.items())
    ]
    return CyclicPrefixSumResidueProfileResult(
        modulus=format_canonical_integer(plan.modulus),
        rows=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class _SequencingAdmissionPlan:
    """Canonical reduced values and the complete search envelope."""

    elements: tuple[tuple[int, ...], ...]
    forbidden_values: tuple[tuple[int, ...], ...]
    first_index: int | None
    search_node_limit: int


def _sequencing_tree_bound(element_count: int, first_index: int | None) -> int:
    if element_count == 0:
        return 1
    if first_index is None:
        return sum(
            (element_count - depth) * _falling_factorial(element_count, depth)
            for depth in range(element_count + 1)
        )
    return (
        sum(
            _falling_factorial(element_count - 1, depth)
            for depth in range(element_count)
        )
        + 1
    )


def _falling_factorial(value: int, depth: int) -> int:
    result = 1
    for step in range(depth):
        result *= value - step
    return result


def _admit_forbidden_prefix_sequencing(
    source: FiniteAbelianSequencingSource,
    first_element: tuple[int, ...] | None,
    forbidden_values: tuple[tuple[int, ...], ...],
    search_node_limit: int,
) -> _SequencingAdmissionPlan:
    """Validate the complete sequencing envelope once before search."""

    if not isinstance(source, FiniteAbelianSequencingSource):
        _reject(
            ("source",),
            "forbidden_prefix_sequencing.source_domain",
            "source must be a FiniteAbelianSequencingSource",
        )
    if type(search_node_limit) is not int or not (
        1 <= search_node_limit <= MAX_SEQUENCING_PERMUTATION_NODES
    ):
        _reject(
            ("search_node_limit",),
            "forbidden_prefix_sequencing.node_limit",
            "search_node_limit must be positive and within the admitted bound",
        )

    if len(source.elements) > MAX_SEQUENCING_SOURCE_ITEMS:
        _reject(
            ("source", "elements"),
            "forbidden_prefix_sequencing.source_cardinality",
            "source exceeds the "
            f"{MAX_SEQUENCING_SOURCE_ITEMS}-element exhaustive-search bound",
        )
    if len(forbidden_values) > MAX_SEQUENCING_FORBIDDEN_VALUES:
        _reject(
            ("forbidden_values",),
            "forbidden_prefix_sequencing.forbidden_cardinality",
            "forbidden values exceed the retained request envelope",
        )

    elements = tuple(
        tuple(
            coordinate % modulus
            for coordinate, modulus in zip(element, source.group.moduli, strict=True)
        )
        for element in source.elements
    )
    first_index = None
    if first_element is not None:
        if (
            len(first_element) != len(source.group.moduli)
            or any(
                type(coordinate) is not int
                or len(str(abs(coordinate))) > MAX_MODULUS_DIGITS
                for coordinate in first_element
            )
        ):
            _reject(
                ("first_element",),
                "forbidden_prefix_sequencing.first_element_shape",
                "first_element must contain bounded integer coordinates of the source rank",
            )
        canonical_first = tuple(
            coordinate % modulus
            for coordinate, modulus in zip(
                first_element, source.group.moduli, strict=True
            )
        )
        try:
            first_index = elements.index(canonical_first)
        except ValueError:
            _reject(
                ("first_element",),
                "forbidden_prefix_sequencing.first_element_membership",
                "first_element must reduce to a source element",
            )

    tree_bound = _sequencing_tree_bound(len(elements), first_index)
    if tree_bound > MAX_SEQUENCING_PERMUTATION_NODES:
        _reject(
            ("source", "elements"),
            "forbidden_prefix_sequencing.search_work",
            "complete sequencing search exceeds the admitted global node bound",
        )

    if any(len(value) != len(source.group.moduli) for value in forbidden_values):
        _reject(
            ("forbidden_values",),
            "forbidden_prefix_sequencing.forbidden_rank",
            "every forbidden value must match the source group rank",
        )
    if any(
        type(coordinate) is not int or len(str(abs(coordinate))) > MAX_MODULUS_DIGITS
        for value in forbidden_values
        for coordinate in value
    ):
        _reject(
            ("forbidden_values",),
            "forbidden_prefix_sequencing.forbidden_coordinate",
            "forbidden coordinates must be bounded integers",
        )
    canonical_forbidden = tuple(sorted(
        tuple(
            coordinate % modulus
            for coordinate, modulus in zip(value, source.group.moduli, strict=True)
        )
        for value in forbidden_values
    ))
    if len(set(canonical_forbidden)) != len(canonical_forbidden):
        _reject(
            ("forbidden_values",),
            "forbidden_prefix_sequencing.forbidden_duplicates",
            "forbidden values must be distinct after reduction",
        )
    return _SequencingAdmissionPlan(
        elements=elements,
        forbidden_values=canonical_forbidden,
        first_index=first_index,
        search_node_limit=search_node_limit,
    )


def _build_sequencing_result(
    source: FiniteAbelianSequencingSource,
    first_element: tuple[int, ...] | None,
    forbidden_values: tuple[tuple[int, ...], ...],
    search_node_limit: int,
    search: SequencingKernelResult,
) -> ForbiddenPrefixSequencingResult:
    """Build the trusted public result without replaying search semantics."""

    if search.status != "FOUND":
        return ForbiddenPrefixSequencingResult.model_construct(
            source=source,
            first_element=first_element,
            forbidden_values=forbidden_values,
            search_node_limit=search_node_limit,
            status=search.status,
            ordering=None,
            states_explored=search.states_explored,
        )

    running = tuple(0 for _ in source.group.moduli)
    rows: list[SequencingPrefixSum] = []
    for index in search.ordering_indices:
        element = source.elements[index]
        running = tuple(
            (coordinate + offset) % modulus
            for coordinate, offset, modulus in zip(
                element, running, source.group.moduli, strict=True
            )
        )
        rows.append(
            SequencingPrefixSum(
                source_index=index,
                element=element,
                prefix_sum=running,
            )
        )
    return ForbiddenPrefixSequencingResult.model_construct(
        source=source,
        first_element=first_element,
        forbidden_values=forbidden_values,
        search_node_limit=search_node_limit,
        status="FOUND",
        ordering=tuple(rows),
        states_explored=search.states_explored,
    )


def search_forbidden_prefix_cyclic_ordering(
    source: FiniteAbelianSequencingSource,
    first_element: tuple[int, ...] | None = None,
    forbidden_values: tuple[tuple[int, ...], ...] = (),
    *,
    search_node_limit: int = MAX_SEQUENCING_PERMUTATION_NODES,
) -> ForbiddenPrefixSequencingResult:
    """Find one admitted cyclic sequencing or prove exact nonexistence.

    The native boundary accepts the canonical source value plus reduced group
    elements. Request parsing normalizes caller input; this admission pass is
    the one semantic validation executed before the bounded search.
    """

    plan = _admit_forbidden_prefix_sequencing(
        source,
        first_element,
        forbidden_values,
        search_node_limit,
    )
    search = search_forbidden_prefix_sequencing(
        plan.elements,
        source.group.moduli,
        plan.forbidden_values,
        plan.first_index,
        plan.search_node_limit,
    )
    return _build_sequencing_result(
        source,
        source.elements[plan.first_index] if plan.first_index is not None else None,
        plan.forbidden_values,
        plan.search_node_limit,
        search,
    )
