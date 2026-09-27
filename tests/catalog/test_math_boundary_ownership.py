"""Catalog-owned discovery for operations kept out of math test collection."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationResult,
    FiniteIntegerSequence,
)
from jacobian.math.number_theory.sequences.core.operations import (
    aperiodic_autocorrelation,
)


def test_record_minima_is_published() -> None:
    """Record extraction is published as its own certified operation (#3725)."""
    assert (
        Catalog.open().operation(
            "number_theory.simultaneous_approximation.record_minima.compute"
        )
        is not None
    )


def test_catalog_accepts_serialized_integer_sequence_source() -> None:
    source = FiniteIntegerSequence(values=(1, 2, 3))
    payload = json.loads(source.model_dump_json())
    result = invoke_operation(
        "sequence.autocorrelation.aperiodic.compute",
        payload,
        Catalog.open(),
    )
    native = aperiodic_autocorrelation(source)
    assert result.output == native.model_dump(mode="json")
    restored = AutocorrelationResult.model_validate_json(json.dumps(result.output))
    assert isinstance(restored.source, FiniteIntegerSequence)
    assert restored.source.values == (1, 2, 3)


def test_autocorrelation_catalog_schema_registers_canonical_rational_defs() -> None:
    catalog = Catalog.open()
    descriptor = catalog.inspect("sequence.autocorrelation.aperiodic.compute")
    assert descriptor is not None
    assert "CanonicalRational" in json.dumps(descriptor.input_schema)
