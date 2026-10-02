"""Centralizer discovery distinguishes a basis member from a spanned identity."""

from jacobian.catalog.catalog import Catalog


def test_centralizer_description_promises_identity_in_the_span() -> None:
    descriptor = Catalog.open().inspect("matrix.centralizer.compute")
    assert descriptor is not None
    assert "identity lies in the span" in descriptor.description
    assert "need not be a basis member" in descriptor.description


def test_centralizer_schema_describes_its_own_structured_envelope() -> None:
    descriptor = Catalog.open().inspect("matrix.centralizer.compute")
    assert descriptor is not None
    description = descriptor.input_schema["properties"]["matrix"]["description"]
    assert "sum(m_i^2)" in description
    assert "262,144" in description
    assert "scalar order 17" in description
    assert "distinct diagonal order 32" in description
    assert "General matrices remain bounded to order 16" in description
