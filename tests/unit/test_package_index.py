from __future__ import annotations

import pytest

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.operations import OperationInputKind
from jacobian.family_catalog import FAMILY_INDEX_SPECS
from jacobian.graphs.operation_resources import SELECTED_GRAPH_OPERATION_IDS
from jacobian.lean_frontend.selected import SELECTED_LEAN_OPERATION_IDS
from jacobian.operation_catalog import OperationCatalogError
from jacobian.operation_declarations import InlineOperation
from jacobian.package_index import (
    collect_family_index_entries,
    collect_inline_index_entries,
    generate_package_index,
    load_package_index,
)
from jacobian.polynomials.selected import SELECTED_POLYNOMIAL_OPERATION_IDS
from jacobian.sat_smt.selected import SELECTED_SAT_SMT_OPERATION_IDS
from jacobian.schema_registry import model_schema_uri
from jacobian.selected_operations import SELECTED_CORE_OPERATION_IDS


def test_generated_index_matches_live_inline_declarations() -> None:
    live_ids = {
        operation.operation_id
        for _module, operations, _checkers in load_builtin_operation_modules()
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
    explicit = loaded.get("graph.construct.explicit")
    assert explicit is not None
    assert explicit.family == "graph"


def test_package_index_does_not_load_family_operations_as_inline_symbols() -> None:
    index = generate_package_index()

    with pytest.raises(
        OperationCatalogError, match="family operation requires overlay catalog state"
    ):
        index.load("graph.construct.explicit")


def test_family_descriptor_retains_live_routing_metadata() -> None:
    index = generate_package_index()
    graph_schema_uri = model_schema_uri(
        name="jacobian.simple-undirected-graph",
        version="1",
        model=SimpleUndirectedGraph,
    )

    neighborhood = index.get("graph.compute.neighborhood_independence")
    assert neighborhood is not None
    descriptor = neighborhood.descriptor()
    assert descriptor.accepted_input_kinds == (OperationInputKind.TYPED_ARTIFACT,)
    assert descriptor.accepted_artifact_types == (graph_schema_uri,)
    assert descriptor.read_only is False

    propose = index.get("lean.statement.propose")
    assert propose is not None
    assert propose.descriptor().accepted_input_kinds == (
        OperationInputKind.STRUCTURED_REQUEST,
        OperationInputKind.FORMAL_PROPOSITION,
    )

    inspect = index.get("lean.proof_state.inspect")
    assert inspect is not None
    assert inspect.descriptor().read_only is True


def test_family_index_covers_selected_family_ids() -> None:
    selected = (
        SELECTED_GRAPH_OPERATION_IDS
        | SELECTED_POLYNOMIAL_OPERATION_IDS
        | SELECTED_LEAN_OPERATION_IDS
        | SELECTED_SAT_SMT_OPERATION_IDS
        | SELECTED_CORE_OPERATION_IDS
    )
    indexed = {spec.operation_id for spec in FAMILY_INDEX_SPECS}
    entries = collect_family_index_entries()

    assert indexed == selected
    assert {entry.operation_id for entry in entries} == selected
    assert all(entry.family for entry in entries)
    examples = {entry.operation_id: entry.examples for entry in entries}
    assert examples["polynomial.expression.normalize"][0].name == "combine_like_terms"
    assert examples["sat.cnf.materialize"][0].name == "finite-coloring-cnf"


def test_family_graph_properties_schema_lists_supported_invariants() -> None:
    from jacobian.family_catalog import family_index_payloads
    from jacobian.graphs.invariants import PROPERTY_NAMES

    payload = next(
        item
        for item in family_index_payloads()
        if item["operation_id"] == "graph.compute.properties"
    )
    assert payload["input_schema"]["x-supported-invariants"] == list(PROPERTY_NAMES)
