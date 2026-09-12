"""Complete maximal-chain enumeration from a finite poset's Hasse diagram."""

from __future__ import annotations

from itertools import pairwise

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
    OrderedPair,
)

MAX_MAXIMAL_CHAINS = 100_000
# A complete result allocates one row for every chain and one element slot for
# every repeated chain label.  Bound those unavoidable mathematical output
# units directly; labels and source values retain their own typed limits.
MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS = 1_000_000
# Source validation, dynamic-programming profile construction, and complete
# path materialization all share one deterministic work budget.  The profile
# computes the exact path length and label totals before any path tuple is
# allocated.
MAX_MAXIMAL_CHAIN_WORK = 20_000_000
MAX_MAXIMAL_CHAIN_PROFILE_CELLS = 4 * MAX_POSET_ELEMENTS**2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"poset.maximal_chains.{reason}", message)


class MaximalChainRow(StrictModel):
    """One canonical chain row with structural endpoint and length metadata."""

    elements: tuple[ElementLabel, ...] = Field(
        max_length=MAX_POSET_ELEMENTS,
        description="Chain elements in increasing source-poset order.",
    )
    cover_relations: tuple[OrderedPair, ...] = Field(
        max_length=MAX_POSET_ELEMENTS - 1,
        description="Adjacent Hasse-cover steps, in the same order as elements.",
    )
    lower_endpoint: ElementLabel | None = None
    upper_endpoint: ElementLabel | None = None
    length: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)

    @model_validator(mode="after")
    def require_structural_row(self) -> MaximalChainRow:
        if self.length != len(self.elements):
            raise _validation_error(
                "row_length", "chain length must equal the number of elements"
            )
        if len(self.elements) != len(set(self.elements)):
            raise _validation_error(
                "row_elements_unique", "a chain cannot repeat an element"
            )
        expected_covers = tuple(
            OrderedPair(lower=lower, upper=upper)
            for lower, upper in pairwise(self.elements)
        )
        if self.cover_relations != expected_covers:
            raise _validation_error(
                "row_cover_relations",
                "chain cover relations must match adjacent elements",
            )
        if not self.elements:
            if self.lower_endpoint is not None or self.upper_endpoint is not None:
                raise _validation_error(
                    "empty_row_endpoints",
                    "the empty chain must not have endpoints",
                )
        elif (
            self.lower_endpoint != self.elements[0]
            or self.upper_endpoint != self.elements[-1]
        ):
            raise _validation_error(
                "row_endpoints",
                "chain endpoints must match its first and last elements",
            )
        return self


class ChainLengthCount(StrictModel):
    length: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)
    count: StrictInt = Field(ge=1, le=MAX_MAXIMAL_CHAINS)


class MaximalChainEnumerationResult(StrictModel):
    """Complete inclusion-maximal chain profile, ordered by chain elements."""

    poset: FinitePoset
    chains: tuple[MaximalChainRow, ...] = Field(
        max_length=MAX_MAXIMAL_CHAINS,
        description="Unique maximal chains in lexicographic source-element order.",
    )
    length_histogram: tuple[ChainLengthCount, ...] = Field(
        max_length=MAX_POSET_ELEMENTS + 1,
        description="Exact count of returned chains grouped by cardinality.",
    )

    @model_validator(mode="after")
    def require_structural_profile(self) -> MaximalChainEnumerationResult:
        carrier = set(self.poset.elements)
        covers = {(pair.lower, pair.upper) for pair in self.poset.cover_relations}
        minimal = set(self.poset.minimal_elements)
        maximal = set(self.poset.maximal_elements)
        row_keys = tuple(row.elements for row in self.chains)
        if any(left >= right for left, right in pairwise(row_keys)):
            raise _validation_error(
                "chains_canonical",
                "chains must be unique and lexicographically ordered",
            )
        if any(
            element not in carrier for row in self.chains for element in row.elements
        ):
            raise _validation_error(
                "chain_elements_in_carrier",
                "chain elements must belong to the source poset",
            )
        if any(
            (pair.lower, pair.upper) not in covers
            for row in self.chains
            for pair in row.cover_relations
        ):
            raise _validation_error(
                "chain_covers_source",
                "chain cover relations must belong to the source Hasse diagram",
            )
        if any(
            row.elements
            and (row.lower_endpoint not in minimal or row.upper_endpoint not in maximal)
            for row in self.chains
        ):
            raise _validation_error(
                "chain_endpoints_source",
                "nonempty maximal chains must start and end at source extrema",
            )
        if self.poset.elements and any(not row.elements for row in self.chains):
            raise _validation_error(
                "nonempty_source_empty_chain",
                "a nonempty poset cannot contain an empty chain",
            )
        if not self.chains:
            raise _validation_error(
                "profile_nonempty",
                "a finite poset profile must contain a maximal chain",
            )
        if not self.poset.elements and row_keys != ((),):
            raise _validation_error(
                "empty_source_chain", "the empty poset has exactly one empty chain"
            )
        histogram_counts: dict[int, int] = {}
        for row in self.chains:
            histogram_counts[row.length] = histogram_counts.get(row.length, 0) + 1
        histogram = tuple(sorted(histogram_counts.items()))
        actual_histogram = tuple(
            (item.length, item.count) for item in self.length_histogram
        )
        if actual_histogram != histogram:
            raise _validation_error(
                "histogram_matches_chains",
                "length histogram must exactly count the returned chains",
            )
        return self


def _chain_profile_stats(
    poset: FinitePoset, outgoing: dict[str, list[str]]
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...]]:
    """Admit all paths with a length-indexed DAG dynamic program.

    The profile stores counts by chain cardinality for each reachable source
    element.  It does not enumerate or allocate any complete chain; the caller
    uses the resulting count and element-slot total for semantic admission.
    """

    # Each map is indexed by chain cardinality.  A finite poset's cover graph
    # is a DAG, and every element is reachable from a minimal element.
    path_counts: dict[str, dict[int, int]] = {}
    profile_work = 0

    def profile_from(element: str) -> dict[int, int]:
        nonlocal profile_work
        if element in path_counts:
            return path_counts[element]
        request_checkpoint("during maximal-chain profile admission")
        profile_work += 1
        targets = outgoing[element]
        if not targets:
            leaf_counts = {1: 1}
            path_counts[element] = leaf_counts
        else:
            merged_counts: dict[int, int] = {}
            for target in targets:
                for length, count in profile_from(target).items():
                    profile_work += 1
                    new_length = length + 1
                    merged_counts[new_length] = merged_counts.get(new_length, 0) + count
            path_counts[element] = merged_counts
        return path_counts[element]

    for minimal in poset.minimal_elements:
        profile_from(minimal)
    total_by_length: dict[int, int] = {}
    total_elements = 0
    for minimal in poset.minimal_elements:
        for length, count in path_counts[minimal].items():
            total_by_length[length] = total_by_length.get(length, 0) + count
            total_elements += length * count
    total = sum(total_by_length.values())
    profile_cells = sum(len(counts) for counts in path_counts.values())
    return (
        total,
        total_elements,
        profile_work,
        profile_cells,
        tuple(sorted(total_by_length.items())),
    )


def _admit_typed_source(poset: FinitePoset) -> int:
    from jacobian.math.combinatorics.posets.core.operations import (
        verify_finite_poset,
    )

    request_checkpoint("before maximal-chain source admission")
    if not isinstance(poset, FinitePoset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="poset.maximal_chains.request_type",
            message="maximal-chain enumeration requires a typed finite poset",
        )
    try:
        element_count = len(poset.elements)
        strict_pair_count = len(poset.strict_order_pairs)
        cover_count = len(poset.cover_relations)
    except (AttributeError, TypeError):
        raise OperationDomainValidationError(
            location=("poset",),
            code="poset.maximal_chains.request_structure",
            message="finite-poset fields are incomplete or malformed",
        ) from None
    source_validation_work = (element_count + 1) ** 3 + (
        strict_pair_count + cover_count
    ) * (element_count + 1)
    if source_validation_work > MAX_MAXIMAL_CHAIN_WORK:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.source_work_bound",
            message=(
                "canonical finite-poset validation requires "
                f"{source_validation_work} work units; maximum is "
                f"{MAX_MAXIMAL_CHAIN_WORK}"
            ),
        )
    # The operation consumes cover_relations as its Hasse DAG.  A structurally
    # valid serialized value can retain untrusted derived claims, so establish
    # the complete canonical source profile before using those edges.
    try:
        canonical_source = verify_finite_poset(poset)
    except (AttributeError, TypeError, ValueError):
        canonical_source = False
    if not canonical_source:
        raise OperationDomainValidationError(
            location=("poset",),
            code="poset.maximal_chains.source_claims",
            message="the finite-poset order and Hasse claims are not canonical",
        )
    return source_validation_work


def _profile_and_admit_output(
    poset: FinitePoset,
    outgoing: dict[str, list[str]],
    source_validation_work: int,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if poset.elements:
        (
            total,
            total_elements,
            profile_work,
            profile_cells,
            histogram,
        ) = _chain_profile_stats(poset, outgoing)
    else:
        total = 1
        total_elements = 0
        profile_work = 1
        profile_cells = 1
        histogram = ((0, 1),)
    if profile_cells > MAX_MAXIMAL_CHAIN_PROFILE_CELLS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.intermediate_bound",
            message=(
                "the complete chain profile requires "
                f"{profile_cells} intermediate cells; maximum is "
                f"{MAX_MAXIMAL_CHAIN_PROFILE_CELLS}"
            ),
        )
    if total > MAX_MAXIMAL_CHAINS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.output_bound",
            message=f"the Hasse diagram has {total} maximal chains; maximum is {MAX_MAXIMAL_CHAINS}",
        )
    if total_elements > MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.element_slots_bound",
            message=(
                "the complete maximal-chain result contains "
                f"{total_elements} repeated element slots; maximum is "
                f"{MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS}"
            ),
        )
    estimated_work = (
        source_validation_work
        + profile_work
        + total
        # Materialization, row validation, and result-profile validation each
        # inspect the repeated chain slots; charge all four passes explicitly.
        + 4 * total_elements
        + len(histogram)
    )
    if estimated_work > MAX_MAXIMAL_CHAIN_WORK:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.work_bound",
            message=(
                "complete maximal-chain search and materialization require "
                f"{estimated_work} work units; maximum is "
                f"{MAX_MAXIMAL_CHAIN_WORK}"
            ),
        )
    return total, total_elements, histogram


def enumerate_maximal_chains(poset: FinitePoset) -> MaximalChainEnumerationResult:
    """Enumerate every maximal chain of one canonical finite-poset value."""

    source_validation_work = _admit_typed_source(poset)
    outgoing: dict[str, list[str]] = {element: [] for element in poset.elements}
    for relation in poset.cover_relations:
        outgoing[relation.lower].append(relation.upper)
    for targets in outgoing.values():
        targets.sort()
    _, _, admitted_histogram = _profile_and_admit_output(
        poset, outgoing, source_validation_work
    )
    if not poset.elements:
        request_checkpoint("before maximal-chain result construction")
        return MaximalChainEnumerationResult(
            poset=poset,
            chains=(MaximalChainRow(elements=(), cover_relations=(), length=0),),
            length_histogram=tuple(
                ChainLengthCount(length=length, count=count)
                for length, count in admitted_histogram
            ),
        )
    chains: list[tuple[str, ...]] = []

    def visit(path: list[str]) -> None:
        request_checkpoint("during maximal-chain materialization")
        targets = outgoing[path[-1]]
        if not targets:
            chains.append(tuple(path))
            return
        for target in targets:
            path.append(target)
            visit(path)
            path.pop()

    for minimal in poset.minimal_elements:
        visit([minimal])
    rows = tuple(
        MaximalChainRow(
            elements=chain,
            cover_relations=tuple(
                OrderedPair(lower=lower, upper=upper)
                for lower, upper in pairwise(chain)
            ),
            lower_endpoint=chain[0],
            upper_endpoint=chain[-1],
            length=len(chain),
        )
        for chain in chains
    )
    request_checkpoint("before maximal-chain result construction")
    return MaximalChainEnumerationResult(
        poset=poset,
        chains=rows,
        length_histogram=tuple(
            ChainLengthCount(length=length, count=count)
            for length, count in admitted_histogram
        ),
    )


__all__ = [
    "MAX_MAXIMAL_CHAINS",
    "MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS",
    "MAX_MAXIMAL_CHAIN_PROFILE_CELLS",
    "MAX_MAXIMAL_CHAIN_WORK",
    "ChainLengthCount",
    "MaximalChainEnumerationResult",
    "MaximalChainRow",
    "enumerate_maximal_chains",
]
