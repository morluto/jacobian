"""Published function-field declarations belong in the catalog lane."""

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS


def _tool(operation_id: str):
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)


def _run_example(operation_id: str):
    tool = _tool(operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    return tool.run(request)


def test_affine_hyperelliptic_valuation_example_is_published_and_runs() -> None:
    result = _run_example("function_field.hyperelliptic_affine_place.valuation.compute")

    assert result.valuation.value == 1


def test_hyperelliptic_infinity_valuation_example_is_published_and_runs() -> None:
    result = _run_example(
        "function_field.hyperelliptic_infinity_place.valuation.compute"
    )

    assert result.valuation.value == -3


def test_rational_riemann_roch_is_the_published_operation() -> None:
    operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    operation_id = "function_field.riemann_roch_space.compute"

    assert operation_id in operation_ids
    assert "function_field.hyperelliptic_infinity_riemann_roch_space.compute" not in operation_ids

    result = _run_example(operation_id)
    assert result.dimension == 3
