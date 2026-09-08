"""MCP declaration coverage for rational-map composition."""

from jacobian.math.polynomials.rational_functions.composition._tools import TOOLS


def test_rational_map_composition_tool_is_declared() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert tool.operation_id == "rational_function_map.compose.compute"
    assert tool.request_type.__name__ == "RationalMapCompositionRequest"
    assert tool.result_type.__name__ == "RationalFunctionMapComposition"
    schema = tool.request_type.model_json_schema()
    description = schema.get("description", "")
    assert "inner.target_coordinates" in description
    assert "outer.source_variables" in description
    properties = schema["properties"]
    assert "inner.target_coordinates" in properties["inner"]["description"]
    assert "outer.source_variables" in properties["outer"]["description"]
