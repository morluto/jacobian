"""Complete maximal-chain enumeration from a finite poset's Hasse diagram."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
    PosetRequest,
)

MAX_MAXIMAL_CHAINS = 100_000


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


def enumerate_maximal_chains(request: PosetRequest) -> MaximalChainEnumerationResult:
    from jacobian.math.combinatorics.posets.core.operations import (
        _admit_canonical_poset,
    )

    poset = request.poset
    _admit_canonical_poset(poset)
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
    path_counts: dict[str, int] = {}

    def count_from(element: str) -> int:
        if element in path_counts:
            return path_counts[element]
        targets = outgoing[element]
        count = 1 if not targets else sum(count_from(target) for target in targets)
        path_counts[element] = count
        return count

    total = sum(count_from(element) for element in poset.minimal_elements)
    if total > MAX_MAXIMAL_CHAINS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="poset.maximal_chains.output_bound",
            message=f"the Hasse diagram has {total} maximal chains; maximum is {MAX_MAXIMAL_CHAINS}",
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
