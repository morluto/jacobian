"""Complete maximal-chain enumeration from a finite poset's Hasse diagram."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
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
# profile below that envelope by charging the source, every repeated chain
# label, and a conservative per-row structural allowance before expansion.
MAX_MAXIMAL_CHAIN_RESULT_BYTES = 10 * 1024 * 1024
_MAX_CHAIN_ROW_OVERHEAD_BYTES = 128


class MaximalChainRow(StrictModel):
    elements: tuple[ElementLabel, ...] = Field(max_length=MAX_POSET_ELEMENTS)
    lower_endpoint: ElementLabel | None = None
    upper_endpoint: ElementLabel | None = None
    length: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)


class ChainLengthCount(StrictModel):
    length: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)
    count: StrictInt = Field(ge=1, le=MAX_MAXIMAL_CHAINS)


class MaximalChainEnumerationResult(StrictModel):
    poset: FinitePoset
    chains: tuple[MaximalChainRow, ...] = Field(max_length=MAX_MAXIMAL_CHAINS)
    length_histogram: tuple[ChainLengthCount, ...] = Field(
        max_length=MAX_POSET_ELEMENTS + 1
    )


def _chain_profile_stats(
    poset: FinitePoset, outgoing: dict[str, list[str]]
) -> tuple[int, int, int]:
    """Count paths and repeated labels without materializing any chain."""

    path_counts: dict[str, int] = {}
    path_element_label_bytes: dict[str, int] = {}
    path_upper_endpoint_bytes: dict[str, int] = {}

    def count_from(element: str) -> int:
        if element in path_counts:
            return path_counts[element]
        targets = outgoing[element]
        element_bytes = len(encode_strict_json(element))
        if not targets:
            count = 1
            path_element_label_bytes[element] = element_bytes
            path_upper_endpoint_bytes[element] = element_bytes
        else:
            count = sum(count_from(target) for target in targets)
            path_element_label_bytes[element] = count * element_bytes + sum(
                path_element_label_bytes[target] for target in targets
            )
            path_upper_endpoint_bytes[element] = sum(
                path_upper_endpoint_bytes[target] for target in targets
            )
        path_counts[element] = count
        return count

    for minimal in poset.minimal_elements:
        count_from(minimal)
    total = sum(path_counts[element] for element in poset.minimal_elements)
    total_element_label_bytes = sum(
        path_element_label_bytes[element] for element in poset.minimal_elements
    )
    total_endpoint_label_bytes = sum(
        path_counts[element] * len(encode_strict_json(element))
        + path_upper_endpoint_bytes[element]
        for element in poset.minimal_elements
    )
    return total, total_element_label_bytes, total_endpoint_label_bytes


def enumerate_maximal_chains(request: PosetRequest) -> MaximalChainEnumerationResult:
    from jacobian.math.combinatorics.posets.core.operations import (
        verify_finite_poset,
    )

    poset = request.poset
    # The operation consumes cover_relations as its Hasse DAG.  A structurally
    # valid serialized value can retain untrusted derived claims, so establish
    # the complete canonical source profile before using those edges.
    if not verify_finite_poset(poset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="poset.maximal_chains.source_claims",
            message="the finite-poset order and Hasse claims are not canonical",
        )
    if not poset.elements:
        return MaximalChainEnumerationResult(
            poset=poset,
            chains=(MaximalChainRow(elements=(), length=0),),
            length_histogram=(ChainLengthCount(length=0, count=1),),
        )
    outgoing: dict[str, list[str]] = {element: [] for element in poset.elements}
    for relation in poset.cover_relations:
        outgoing[relation.lower].append(relation.upper)
    for targets in outgoing.values():
        targets.sort()
    total, total_element_label_bytes, total_endpoint_label_bytes = _chain_profile_stats(
        poset, outgoing
    )
    if total > MAX_MAXIMAL_CHAINS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.output_bound",
            message=f"the Hasse diagram has {total} maximal chains; maximum is {MAX_MAXIMAL_CHAINS}",
        )
    # The source is retained in the exact result.  Encode only that already
    # admitted source now; chains themselves are represented by a dynamic
    # label-byte recurrence and are not materialized until all admission
    # checks pass.
    source_bytes = len(encode_strict_json(poset.model_dump(mode="json")))
    predicted_result_bytes = (
        source_bytes
        + total_element_label_bytes
        + total_endpoint_label_bytes
        + total * _MAX_CHAIN_ROW_OVERHEAD_BYTES
        + (len(poset.elements) + 1) * 64
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
    histogram: dict[int, int] = {}
    for row in rows:
        histogram[row.length] = histogram.get(row.length, 0) + 1
    return MaximalChainEnumerationResult(
        poset=poset,
        chains=rows,
        length_histogram=tuple(
            ChainLengthCount(length=length, count=count)
            for length, count in sorted(histogram.items())
        ),
    )
