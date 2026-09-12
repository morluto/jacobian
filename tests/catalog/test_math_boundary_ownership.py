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
    sequence_order_shape,
)


def test_record_minima_remains_unpublished() -> None:
    """Record extraction composes the public range operation in native code."""
    assert (
        Catalog.open().operation(
            "number_theory.simultaneous_approximation.record_minima.compute"
        )
        is None
    )


def test_catalog_accepts_serialized_integer_sequence_source() -> None:
    source = sequence_order_shape(FiniteIntegerSequence(values=(1, 2, 3))).source
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
    descriptor = next(
        operation
        for operation in catalog.snapshot().operations
        if operation.operation_id == "sequence.autocorrelation.aperiodic.compute"
    )
    assert "CanonicalRational" in json.dumps(descriptor.input_schema)
