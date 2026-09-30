"""Fail-closed checks for owner-local public tool manifests."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from jacobian.catalog.builtins import (
    _BUILTIN_TOOL_MODULES,
    BUILTIN_TOOLS,
    _load_tools,
)
from jacobian.catalog.catalog import Catalog
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring._models import (
    NonmonochromaticColoringResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring.operations import (
    verify_nonmonochromatic_coloring,
)


def test_tool_manifest_discovery_is_deterministic_and_owner_local() -> None:
    assert tuple(sorted(_BUILTIN_TOOL_MODULES)) == _BUILTIN_TOOL_MODULES
    assert len(_BUILTIN_TOOL_MODULES) == len(set(_BUILTIN_TOOL_MODULES))
    assert all(
        module_name.startswith("jacobian.math.") and module_name.endswith("._tools")
        for module_name in _BUILTIN_TOOL_MODULES
    )


def test_public_catalog_is_sorted_and_unique() -> None:
    operation_ids = tuple(tool.operation_id for tool in BUILTIN_TOOLS)
    assert operation_ids == tuple(sorted(set(operation_ids)))
    catalog = Catalog.open()
    assert all(catalog.operation(tool.operation_id) is tool for tool in BUILTIN_TOOLS)


def test_hypergraph_coloring_has_one_bounded_decision_operation() -> None:
    catalog = Catalog.open()

    assert (
        catalog.operation("hypergraph.nonmonochromatic_vertex_coloring.q_decide")
        is not None
    )
    assert catalog.operation("hypergraph.coloring.non_monochromatic.decide") is None


def test_retired_hypergraph_coloring_identifier_has_a_documented_migration() -> None:
    """A deliberately removed catalog ID must point callers to its replacement."""

    migration = (
        Path(__file__).resolve().parents[2] / "docs" / "reference" / "tools.md"
    ).read_text()
    assert "hypergraph.coloring.non_monochromatic.decide" in migration
    assert "hypergraph.nonmonochromatic_vertex_coloring.q_decide" in migration
    assert "COLORABLE" in migration
    assert "witness.assignments" in migration


@pytest.mark.parametrize("colorable", [False, True])
def test_documented_coloring_migration_produces_a_valid_published_result(
    colorable: bool,
) -> None:
    operation = Catalog.open().operation(
        "hypergraph.nonmonochromatic_vertex_coloring.q_decide"
    )
    assert operation is not None
    vertices = ("a", "b")
    old_coloring = (1, 0)
    payload = {
        "hypergraph": {"vertices": vertices, "edges": (("edge", vertices),)},
        "palette_size": 2 if colorable else 1,
        "outcome": "COLORABLE" if colorable else "NOT_COLORABLE",
        **(
            {
                "witness": {
                    "assignments": tuple(zip(vertices, old_coloring, strict=True))
                }
            }
            if colorable
            else {}
        ),
    }

    result = operation.result_type.model_validate_json(json.dumps(payload))

    assert isinstance(result, NonmonochromaticColoringResult)
    assert verify_nonmonochromatic_coloring(result)
    if colorable:
        assert result.witness is not None
        assert result.witness.assignments == (("a", 1), ("b", 0))
    else:
        assert result.witness is None


def test_cyclotomic_inclusion_has_no_composition_operation() -> None:
    """Composition is a call-order helper, not a published postcondition.

    ``matrix.cyclic.cyclotomic_inclusion.compute`` on the composed pair
    already publishes the canonical inclusion, so a composition entry would
    describe a call order rather than a distinct mathematical relation.
    """

    catalog = Catalog.open()

    assert catalog.operation("matrix.cyclic.cyclotomic_inclusion.compute") is not None
    assert catalog.operation("matrix.cyclic.cyclotomic_inclusion.compose") is None


def test_friable_count_manifest_has_one_executable_example() -> None:
    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "number_theory.friable.count.compute"
    )

    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    assert operation.run(request).count == 34


def test_tool_loading_rejects_a_malformed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.catalog.builtins.import_module",
        lambda _module_name: types.SimpleNamespace(TOOLS=[]),
    )

    with pytest.raises(TypeError, match="must export a tuple of MathTool values"):
        _load_tools(("jacobian.math.example._tools",))


def test_tool_loading_rejects_duplicate_operation_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = BUILTIN_TOOLS[0]
    monkeypatch.setattr(
        "jacobian.catalog.builtins.import_module",
        lambda _module_name: types.SimpleNamespace(TOOLS=(tool, tool)),
    )

    with pytest.raises(ValueError, match="built-in operation IDs must be unique"):
        _load_tools(("jacobian.math.example._tools",))
