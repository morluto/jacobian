"""Public admission boundary for sparse fixed-length quiver paths."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_math_run_admits_sparse_cycle_at_motivating_scale() -> None:
    vertex_count = 64
    arrows = [[vertex, (vertex + 1) % vertex_count] for vertex in range(vertex_count)]
    result = invoke_operation(
        "quiver.paths.fixed_length.compute",
        {
            "quiver": {"vertex_count": vertex_count, "arrows": arrows},
            "length": 32,
        },
        Catalog.open(),
    )

    output = result.output
    assert output["length"] == 32
    assert output["total_paths"] == "64"
    assert output["quiver"]["vertex_count"] == vertex_count
    assert output["path_matrix"]["entries"] == [
        [
            str(int(column == (row + 32) % vertex_count))
            for column in range(vertex_count)
        ]
        for row in range(vertex_count)
    ]
