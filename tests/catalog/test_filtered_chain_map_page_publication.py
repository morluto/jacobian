"""Publication check for the filtered chain-map page operation.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapPageResult,
)
from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS


def test_page_map_tool_uses_the_native_operation_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.page.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapPageResult
    example = tool.examples[0]
    result = invoke_operation(
        tool.operation_id,
        example.input,
        Catalog.open(),
    )
    assert result.output["maps"] == [[[["1"]]]]
