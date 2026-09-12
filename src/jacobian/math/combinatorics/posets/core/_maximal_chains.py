"""Complete maximal-chain enumeration from a finite poset's Hasse diagram."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, strict_json_object_size
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
    PosetRequest,
)

MAX_MAXIMAL_CHAINS = 100_000
# The ordinary canonical transport envelope is 10 MiB.  Keep the complete
# profile below that envelope by calculating the exact canonical JSON size
# from the already admitted source and the chain profile.
MAX_MAXIMAL_CHAIN_RESULT_BYTES = 10 * 1024 * 1024
# Source validation, dynamic-programming profile construction, and complete
# path materialization all share one deterministic work budget.  The profile
# computes the exact path length and label totals before any path tuple is
# allocated.
MAX_MAXIMAL_CHAIN_WORK = 20_000_000
MAX_MAXIMAL_CHAIN_PROFILE_CELLS = 4 * MAX_POSET_ELEMENTS**2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"poset.maximal_chains.{reason}", message)


def _encoded_array_size(item_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _chain_row_fixed_overhead() -> int:
    """Return one row's object syntax and field-name bytes."""

    return strict_json_object_size(
        (
            ("elements", 0),
            ("lower_endpoint", 0),
            ("upper_endpoint", 0),
            ("length", 0),
        )
    )


class MaximalChainRow(StrictModel):
    """One canonical chain row with structural endpoint and length metadata."""

    elements: tuple[ElementLabel, ...] = Field(max_length=MAX_POSET_ELEMENTS)
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
    chains: tuple[MaximalChainRow, ...] = Field(max_length=MAX_MAXIMAL_CHAINS)
    length_histogram: tuple[ChainLengthCount, ...] = Field(
        max_length=MAX_POSET_ELEMENTS + 1
    )

    @model_validator(mode="after")
    def require_structural_profile(self) -> MaximalChainEnumerationResult:
        carrier = set(self.poset.elements)
        row_keys = tuple(row.elements for row in self.chains)
        if row_keys != tuple(sorted(set(row_keys))):
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
        histogram = tuple(
            (length, count)
            for length, count in sorted(
                {
                    row.length: sum(item.length == row.length for item in self.chains)
                    for row in self.chains
                }.items()
            )
        )
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
) -> tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]:
    """Admit and size all paths without materializing any chain."""

    # Each map is indexed by chain cardinality.  Keeping aggregate label
    # bytes by length is enough to calculate each row's exact canonical JSON
    # size; no reachability or path enumeration is replayed here.
    path_counts: dict[str, dict[int, int]] = {}
    path_element_label_bytes: dict[str, dict[int, int]] = {}
    path_upper_endpoint_bytes: dict[str, dict[int, int]] = {}
    path_row_bytes: dict[str, dict[int, int]] = {}
    label_bytes = {
        element: len(encode_strict_json(element)) for element in poset.elements
    }
    row_fixed_overhead = _chain_row_fixed_overhead()
    profile_work = 0

    def profile_from(element: str) -> dict[int, int]:
        nonlocal profile_work
        if element in path_counts:
            return path_counts[element]
        profile_work += 1
        targets = outgoing[element]
        if not targets:
            leaf_counts = {1: 1}
            path_counts[element] = leaf_counts
            path_element_label_bytes[element] = {1: label_bytes[element]}
            path_upper_endpoint_bytes[element] = {1: label_bytes[element]}
        else:
            merged_counts: dict[int, int] = {}
            element_totals: dict[int, int] = {}
            endpoint_totals: dict[int, int] = {}
            for target in targets:
                for length, count in profile_from(target).items():
                    profile_work += 1
                    new_length = length + 1
                    merged_counts[new_length] = merged_counts.get(new_length, 0) + count
                    element_totals[new_length] = element_totals.get(new_length, 0) + (
                        count * label_bytes[element]
                        + path_element_label_bytes[target][length]
                    )
                    endpoint_totals[new_length] = (
                        endpoint_totals.get(new_length, 0)
                        + path_upper_endpoint_bytes[target][length]
                    )
            path_counts[element] = merged_counts
            path_element_label_bytes[element] = element_totals
            path_upper_endpoint_bytes[element] = endpoint_totals
        row_bytes: dict[int, int] = {}
        for length, count in path_counts[element].items():
            element_array_bytes = (
                count * (length + 1) + path_element_label_bytes[element][length]
            )
            row_bytes[length] = (
                count * row_fixed_overhead
                + element_array_bytes
                + count * label_bytes[element]
                + path_upper_endpoint_bytes[element][length]
                + count * len(encode_strict_json(length))
            )
        path_row_bytes[element] = row_bytes
        return path_counts[element]

    for minimal in poset.minimal_elements:
        profile_from(minimal)
    total_by_length: dict[int, int] = {}
    total_row_bytes = 0
    total_elements = 0
    for minimal in poset.minimal_elements:
        for length, count in path_counts[minimal].items():
            total_by_length[length] = total_by_length.get(length, 0) + count
            total_row_bytes += path_row_bytes[minimal][length]
            total_elements += length * count
    total = sum(total_by_length.values())
    profile_cells = sum(len(counts) for counts in path_counts.values()) * 4
    return (
        total,
        total_elements,
        total_row_bytes,
        profile_work,
        profile_cells,
        tuple(sorted(total_by_length.items())),
    )


def _admit_typed_source(request: PosetRequest) -> tuple[FinitePoset, int]:
    from jacobian.math.combinatorics.posets.core.operations import (
        verify_finite_poset,
    )

    candidate = getattr(request, "poset", None)
    if not isinstance(request, PosetRequest) or not isinstance(candidate, FinitePoset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="poset.maximal_chains.request_type",
            message="maximal-chain enumeration requires a typed finite-poset request",
        )
    poset = candidate
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
    return poset, source_validation_work


def _profile_and_admit_output(
    poset: FinitePoset,
    outgoing: dict[str, list[str]],
    source_validation_work: int,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if poset.elements:
        (
            total,
            total_elements,
            total_row_bytes,
            profile_work,
            profile_cells,
            histogram,
        ) = _chain_profile_stats(poset, outgoing)
    else:
        total = 1
        total_elements = 0
        total_row_bytes = len(
            encode_strict_json(
                {
                    "elements": [],
                    "lower_endpoint": None,
                    "upper_endpoint": None,
                    "length": 0,
                }
            )
        )
        profile_work = 1
        profile_cells = 4
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
    source_bytes = len(encode_strict_json(poset.model_dump(mode="json")))
    chains_bytes = 2 + max(total - 1, 0) + total_row_bytes
    histogram_bytes = _encoded_array_size(
        tuple(
            strict_json_object_size(
                (
                    ("length", len(encode_strict_json(length))),
                    ("count", len(encode_strict_json(count))),
                )
            )
            for length, count in histogram
        )
    )
    predicted_result_bytes = strict_json_object_size(
        (
            ("poset", source_bytes),
            ("chains", chains_bytes),
            ("length_histogram", histogram_bytes),
        )
    )
    if predicted_result_bytes > MAX_MAXIMAL_CHAIN_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.result_bytes_bound",
            message=(
                "the complete maximal-chain result is predicted to occupy "
                f"{predicted_result_bytes} bytes; maximum is "
                f"{MAX_MAXIMAL_CHAIN_RESULT_BYTES}"
            ),
        )
    estimated_work = (
        source_validation_work
        + profile_work
        + total
        + 2 * total_elements
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


def enumerate_maximal_chains(request: PosetRequest) -> MaximalChainEnumerationResult:
    poset, source_validation_work = _admit_typed_source(request)
    outgoing: dict[str, list[str]] = {element: [] for element in poset.elements}
    for relation in poset.cover_relations:
        outgoing[relation.lower].append(relation.upper)
    for targets in outgoing.values():
        targets.sort()
    _profile_and_admit_output(poset, outgoing, source_validation_work)
    if not poset.elements:
        return MaximalChainEnumerationResult(
            poset=poset,
            chains=(MaximalChainRow(elements=(), length=0),),
            length_histogram=(ChainLengthCount(length=0, count=1),),
        )
    chains: list[tuple[str, ...]] = []

    def visit(path: tuple[str, ...]) -> None:
        targets = outgoing[path[-1]]
        if not targets:
            chains.append(path)
            return
        for target in targets:
            visit((*path, target))

    for minimal in poset.minimal_elements:
        visit((minimal,))
    rows = tuple(
        MaximalChainRow(
            elements=chain,
            lower_endpoint=chain[0],
            upper_endpoint=chain[-1],
            length=len(chain),
        )
        for chain in chains
    )
    histogram_counts: dict[int, int] = {}
    for row in rows:
        histogram_counts[row.length] = histogram_counts.get(row.length, 0) + 1
    return MaximalChainEnumerationResult(
        poset=poset,
        chains=rows,
        length_histogram=tuple(
            ChainLengthCount(length=length, count=count)
            for length, count in sorted(histogram_counts.items())
        ),
    )
