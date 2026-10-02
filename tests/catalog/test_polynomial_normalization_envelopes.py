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
