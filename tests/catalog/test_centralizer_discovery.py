"""Centralizer discovery distinguishes a basis member from a spanned identity."""

from jacobian.catalog.catalog import Catalog


def test_centralizer_description_promises_identity_in_the_span() -> None:
    descriptor = Catalog.open().inspect("matrix.centralizer.compute")
    assert descriptor is not None
    assert "identity lies in the span" in descriptor.description
    assert "need not be a basis member" in descriptor.description
