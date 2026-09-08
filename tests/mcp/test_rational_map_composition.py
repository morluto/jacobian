"""MCP declaration coverage for rational-map composition."""

from jacobian.math.polynomials.rational_functions.composition._tools import TOOLS


def test_rational_map_composition_tool_is_declared() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert tool.operation_id == "rational_function_map.compose.compute"
    assert tool.request_type.__name__ == "RationalMapCompositionRequest"
    assert tool.result_type.__name__ == "RationalFunctionMapComposition"
