"""Catalog schema check for the restored residue-image bounds."""


def test_both_residue_image_operations_advertise_the_restored_bounds() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    residue_image_tools = [
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id.startswith("modular.polynomial_residue_image.")
    ]
    assert len(residue_image_tools) == 2

    for tool in residue_image_tools:
        exponents_schema = tool.result_type.model_json_schema()["properties"][
            "normalized_terms"
        ]["items"]["properties"]["exponents"]
        assert exponents_schema["maxItems"] == 6
        assert exponents_schema["items"]["minimum"] == 0
        assert exponents_schema["items"]["maximum"] == 32
