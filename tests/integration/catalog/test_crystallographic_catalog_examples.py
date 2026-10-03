"""Advertised crystallographic-extension example through the catalog."""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_pairing_operation_is_published_with_square_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "crystallographic.extension.polytope_facet_pairings.compute"
    )

    assert tool.examples[0].name == "unit_square_translation_pairings"
    # Route through dispatch so operation-ID lookup, strict wire parsing, and
    # the serialized output envelope all have to accept the advertised example.
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    validated = tool.result_type.model_validate(result.output)
    assert validated.facet_profile.dimension == 2
