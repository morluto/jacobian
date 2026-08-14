from __future__ import annotations

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.operation_declarations import InlineOperation
from jacobian.package_index import (
    collect_inline_index_entries,
    generate_package_index,
    load_package_index,
)


def test_generated_index_matches_live_inline_declarations() -> None:
    live_ids = {
        operation.operation_id
        for _module, operations in load_builtin_operation_modules()
        for operation in operations
        if isinstance(operation, InlineOperation)
    }
    entries = collect_inline_index_entries()
    indexed_ids = {entry.operation_id for entry in entries}

    assert indexed_ids == live_ids
    assert "matrix.determinant.compute" in indexed_ids
    assert all(entry.module and entry.symbol for entry in entries)


def test_package_index_loads_matrix_determinant_symbol() -> None:
    index = generate_package_index()
    operation = index.load("matrix.determinant.compute")

    assert isinstance(operation, InlineOperation)
    assert operation.operation_id == "matrix.determinant.compute"
    descriptor = index.get("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.symbol == "MATRIX_DETERMINANT_COMPUTE"


def test_load_package_index_matches_live_declarations() -> None:
    generated = generate_package_index()
    loaded = load_package_index()

    assert set(loaded.entries) == set(generated.entries)
    assert loaded.get("matrix.determinant.compute") is not None
