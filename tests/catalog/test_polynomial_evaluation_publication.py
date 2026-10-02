"""Canonical QQ polynomial evaluation has one public point/result contract."""

from jacobian.catalog.catalog import Catalog


def test_polynomial_evaluation_has_one_public_declaration_and_explicit_regimes() -> (
    None
):
    catalog = Catalog.open()
    assert catalog.operation("polynomial.rational.compute.evaluate") is None
    operation = catalog.operation("polynomial.map.evaluate")
    assert operation is not None
    schema = operation.request_type.model_json_schema()
    source_description = schema["properties"]["polynomial"]["description"]
    assert "degree at most 127" in source_description
    assert "total degree at most 64" in source_description
    assert "32,768-digit" in source_description
    assert "complete ordered axis" in schema["properties"]["point"]["description"]
    assert set(operation.result_type.model_json_schema()["properties"]) == {"value"}
    assert {example.name for example in operation.examples} == {
        "rational_evaluate_x2_plus_one",
        "renamed_univariate_degree_65",
        "simple_eval",
    }
    for example in operation.examples:
        polynomial = example.input["polynomial"]
        assert example.input["point"]["variables"] == polynomial["variables"]
