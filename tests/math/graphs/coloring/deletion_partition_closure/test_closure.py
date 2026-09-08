"""Independent pair-row oracle, normalization and full-envelope examples."""

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring.deletion_partition_closure import construct
from jacobian.math.graphs.coloring.deletion_partition_closure.values import (
    DeletionPartitionRow,
    DeletionPartitionTemplate,
)


def partitions(items: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    result = []
    for partition in partitions(rest):
        result.append(((first,), *partition))
        for index in range(len(partition)):
            result.append(
                (
                    *partition[:index],
                    (first, *partition[index]),
                    *partition[index + 1 :],
                )
            )
    return result


def check_oracle(template: DeletionPartitionTemplate) -> None:
    result = construct(template)
    expected_edges = set()
    expected_blockers = []
    axis = tuple(tuple(sorted(pair)) for pair in combinations(template.vertices, 2))
    for pair_index, (left, right) in enumerate(axis):
        blocking = [
            (row_index, block_index)
            for row_index, row in enumerate(template.rows)
            if row.deleted_vertex not in (left, right)
            for block_index, block in enumerate(row.blocks)
            if left in block and right in block
        ]
        if blocking:
            expected_blockers.append((pair_index, *min(blocking)))
        else:
            expected_edges.add((left, right))
    assert result.template == template
    assert result.source_pair_axis == axis
    assert set(result.graph.edges) == expected_edges
    assert [
        (b.pair_index, b.row_index, b.block_index) for b in result.nonedge_blockers
    ] == expected_blockers
    # Replay every deletion coloring and each maximality obstruction.
    for row in template.rows:
        colors = {
            vertex: index for index, block in enumerate(row.blocks) for vertex in block
        }
        assert all(
            colors[left] != colors[right]
            for left, right in result.graph.edges
            if row.deleted_vertex not in (left, right)
        )
    for blocker in result.nonedge_blockers:
        pair = axis[blocker.pair_index]
        block = template.rows[blocker.row_index].blocks[blocker.block_index]
        assert set(pair) <= set(block)


@pytest.mark.parametrize("size", range(5))
def test_all_partition_templates_through_four_vertices(size: int) -> None:
    vertices = tuple("dcba"[:size])
    row_options = [
        partitions(tuple(v for v in vertices if v != deleted)) for deleted in vertices
    ]
    for rows in product(*row_options):
        template = DeletionPartitionTemplate(
            vertices=vertices,
            block_bound=max(1, size - 1),
            rows=tuple(
                DeletionPartitionRow(deleted_vertex=v, blocks=blocks)
                for v, blocks in zip(vertices, rows, strict=True)
            ),
        )
        check_oracle(template)


def cyclic_template(s: int) -> DeletionPartitionTemplate:
    size = 3 * s + 1
    vertices = tuple(f"v{index}" for index in range(size))
    return DeletionPartitionTemplate(
        vertices=vertices,
        block_bound=3,
        rows=tuple(
            DeletionPartitionRow(
                deleted_vertex=vertices[v],
                blocks=tuple(
                    tuple(vertices[(v + 1 + b * s + j) % size] for j in range(s))
                    for b in range(3)
                ),
            )
            for v in range(size)
        ),
    )


def test_cyclic_interval_template() -> None:
    check_oracle(cyclic_template(3))


def test_full_256_vertex_cyclic_template() -> None:
    template = cyclic_template(85)
    result = construct(template)
    assert len(result.source_pair_axis) == 32640
    assert len(result.graph.edges) + len(result.nonedge_blockers) == 32640
    # A pair survives precisely at cyclic distance at least s; all shorter
    # distances share a length-s interval in some translated deletion row.
    expected = {
        tuple(sorted((template.vertices[i], template.vertices[j])))
        for i, j in combinations(range(256), 2)
        if min(j - i, 256 - (j - i)) >= 85
    }
    assert set(result.graph.edges) == expected


def test_block_member_row_permutations_and_relabeling() -> None:
    template = cyclic_template(2)
    reordered = DeletionPartitionTemplate(
        vertices=template.vertices,
        block_bound=template.block_bound,
        rows=tuple(
            DeletionPartitionRow(
                deleted_vertex=row.deleted_vertex,
                blocks=tuple(tuple(reversed(block)) for block in reversed(row.blocks)),
            )
            for row in reversed(template.rows)
        ),
    )
    assert reordered == template
    assert construct(reordered) == construct(template)
    mapping = {
        v: f"x{len(template.vertices) - i}" for i, v in enumerate(template.vertices)
    }
    renamed = DeletionPartitionTemplate(
        vertices=tuple(mapping[v] for v in template.vertices),
        block_bound=3,
        rows=tuple(
            DeletionPartitionRow(
                deleted_vertex=mapping[row.deleted_vertex],
                blocks=tuple(tuple(mapping[v] for v in block) for block in row.blocks),
            )
            for row in template.rows
        ),
    )
    original, transported = construct(template), construct(renamed)
    assert transported.nonedge_blockers == original.nonedge_blockers
    assert transported.graph.edges == tuple(
        tuple(sorted((mapping[left], mapping[right])))
        for left, right in original.graph.edges
    )


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (("a", (("b",),)), ("a", (("b",),))),
        (("a", (("a",),)), ("b", (("a",),))),
        (("a", (("b",), ("b",))), ("b", (("a",),))),
        (("a", ()), ("b", (("a",),))),
        (("a", (("x",),)), ("b", (("a",),))),
    ],
)
def test_malformed_rows_rejected(
    rows: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...],
) -> None:
    with pytest.raises(ValidationError):
        DeletionPartitionTemplate(
            vertices=("a", "b"),
            block_bound=2,
            rows=tuple(
                DeletionPartitionRow(deleted_vertex=v, blocks=blocks)
                for v, blocks in rows
            ),
        )


def test_bound_and_empty_blocks_rejected() -> None:
    with pytest.raises(ValidationError):
        DeletionPartitionRow(deleted_vertex="a", blocks=((),))
    template = cyclic_template(1)
    with pytest.raises(ValidationError):
        DeletionPartitionTemplate(
            vertices=template.vertices, block_bound=2, rows=template.rows
        )
    with pytest.raises(ValidationError):
        DeletionPartitionTemplate(vertices=(), block_bound=0, rows=())


def test_large_bound_and_graph_compatible_labels() -> None:
    vertices = ("", "\n", "é")
    template = DeletionPartitionTemplate(
        vertices=vertices,
        block_bound=10**1000,
        rows=tuple(
            DeletionPartitionRow(
                deleted_vertex=v, blocks=tuple((w,) for w in vertices if w != v)
            )
            for v in vertices
        ),
    )
    check_oracle(template)
    assert (
        DeletionPartitionTemplate.model_validate_json(template.model_dump_json())
        == template
    )


@pytest.mark.parametrize(
    "vertices",
    [
        ("a", "a"),
        ("e\u0301",),
        ("é" * 33,),
        ("\ud800",),
        tuple(str(i) for i in range(257)),
    ],
)
def test_invalid_carriers(vertices: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        DeletionPartitionTemplate(vertices=vertices, block_bound=1, rows=())
