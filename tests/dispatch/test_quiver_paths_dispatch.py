"""Public path-count dispatch for sparse quivers."""

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_sparse_cycle_path_counts_through_public_boundary() -> None:
    size = 64
    output = invoke_operation(
        "quiver.paths.fixed_length.compute",
        {
            "quiver": {
                "vertex_count": size,
                "arrows": [[source, (source + 1) % size] for source in range(size)],
            },
            "length": 32,
        },
        Catalog.open(),
    ).output
    assert output["total_paths"] == "64"
    assert output["path_matrix"]["entries"] == [
        [str(int(target == (source + 32) % size)) for target in range(size)]
        for source in range(size)
    ]
    assert encode_strict_json(output)
