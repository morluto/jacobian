"""Catalog-owned discovery for operations kept out of math test collection."""

import json

from jacobian.catalog.catalog import Catalog


def test_record_minima_is_published() -> None:
    """Record extraction is published as its own certified operation (#3725)."""
    assert (
        Catalog.open().operation(
            "number_theory.simultaneous_approximation.record_minima.compute"
        )
        is not None
    )


def test_autocorrelation_catalog_schema_registers_canonical_rational_defs() -> None:
    catalog = Catalog.open()
    descriptor = catalog.inspect("sequence.autocorrelation.aperiodic.compute")
    assert descriptor is not None
    assert "CanonicalRational" in json.dumps(descriptor.input_schema)
