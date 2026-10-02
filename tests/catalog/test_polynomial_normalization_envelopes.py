"""Inspection publishes the same output envelope as canonical decoding."""

from jacobian.catalog.catalog import Catalog


def test_factorization_publishes_the_admitted_degree_factor_capacity() -> None:
    operation = Catalog.open().operation("polynomial.factor.compute")
    assert operation is not None
    schema = operation.result_type.model_json_schema(mode="serialization")
    assert schema["properties"]["factors"]["maxItems"] == 500


def test_composition_inspection_explains_identity_and_expansion_envelopes() -> None:
    operation = Catalog.open().operation("polynomial.map.compose")
    assert operation is not None
    assert "Identity substitution on either side" in operation.description
    assert "4096 terms" in operation.description
    assert "degree product at most 128" in operation.description


def test_square_free_inspection_separates_degree_and_multiplicity_envelopes() -> None:
    operation = Catalog.open().operation("polynomial.compute.square_free_decomposition")
    assert operation is not None
    assert "nonzero a and b" in operation.description
    assert "32768" in operation.description
    assert "general sources retain the exponent-64" in operation.description
    request = operation.request_type.model_json_schema()
    description = request["properties"]["polynomial"]["description"]
    assert "degree-one reduction" in description
    assert "256-digit" in description
    result = operation.result_type.model_json_schema(mode="serialization")
    multiplicity = result["$defs"]["PolynomialSquareFreeFactor"]["properties"][
        "multiplicity"
    ]
    assert multiplicity["maximum"] == 64
