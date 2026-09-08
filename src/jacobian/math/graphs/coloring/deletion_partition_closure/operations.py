"""Exact bitset union of forbidden pairs in deletion partition blocks."""

from itertools import combinations

from jacobian.math.graphs.coloring.deletion_partition_closure._models import (
    DeletionPartitionBlocker,
    DeletionPartitionClosure,
)
from jacobian.math.graphs.coloring.deletion_partition_closure.values import (
    DeletionPartitionTemplate,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def construct(template: DeletionPartitionTemplate) -> DeletionPartitionClosure:
    """Construct xy iff every deletion row other than x,y separates x and y.

    The entire canonical envelope is admitted: n <= 256, at most n(n-1)
    memberships, n(n-1)/2 output pairs and that many blockers. There are at
    most n(n-1) bitset updates on n-bit integers; a forbidden pair is expanded
    only once. Labels have at most 64 UTF-8 bytes, so even sixfold JSON escaping
    bounds source plus complete output below 128 MiB (including 32,768 digits
    for the retained block bound). No exponential coloring backend is needed.
    """
    vertices = template.vertices
    positions = {vertex: index for index, vertex in enumerate(vertices)}
    blocked = [0] * len(vertices)
    witnesses: dict[tuple[int, int], tuple[int, int]] = {}
    for row_index, row in enumerate(template.rows):
        for block_index, block in enumerate(row.blocks):
            indices = tuple(positions[vertex] for vertex in block)
            mask = sum(1 << index for index in indices)
            for left in indices:
                new = mask & ~((1 << (left + 1)) - 1) & ~blocked[left]
                while new:
                    bit = new & -new
                    right = bit.bit_length() - 1
                    witnesses[left, right] = (row_index, block_index)
                    new ^= bit
                blocked[left] |= mask
    axis: list[tuple[str, str]] = []
    edges: list[tuple[str, str]] = []
    blockers: list[DeletionPartitionBlocker] = []
    for left, right in combinations(range(len(vertices)), 2):
        pair = (
            min(vertices[left], vertices[right]),
            max(vertices[left], vertices[right]),
        )
        pair_index = len(axis)
        axis.append(pair)
        witness = witnesses.get((left, right))
        if witness is None:
            edges.append(pair)
        else:
            blockers.append(
                DeletionPartitionBlocker(
                    pair_index=pair_index, row_index=witness[0], block_index=witness[1]
                )
            )
    return DeletionPartitionClosure(
        template=template,
        source_pair_axis=tuple(axis),
        graph=SimpleUndirectedGraph(vertices=vertices, edges=tuple(edges)),
        nonedge_blockers=tuple(blockers),
    )
