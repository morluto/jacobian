from __future__ import annotations

import json

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    FiniteSequence,
)
from jacobian.math.number_theory.sequences.core.operations import (
    aperiodic_autocorrelation,
    sequence_order_shape,
    sequence_sum,
)
from jacobian.math.number_theory.sequences.core.values import IntegerSequence


def test_sequence_core_manifest_publishes_every_recovered_operation() -> None:
    ids = {
        tool.operation_id
        for tool in BUILTIN_TOOLS
        if tool.operation_id.startswith("sequence.")
    }
    assert {
        "sequence.compute.sum",
        "sequence.compute.product",
        "sequence.compute.gcd",
        "sequence.compute.lcm",
        "sequence.compute.minimum",
        "sequence.compute.maximum",
        "sequence.compute.range",
        "sequence.compute.distinct_count",
        "sequence.compute.mean",
        "sequence.compute.median",
        "sequence.compute.prefix_sums",
        "sequence.compute.first_differences",
        "sequence.compute.prefix_products",
        "sequence.compute.prefix_minima",
        "sequence.compute.prefix_maxima",
        "sequence.compute.prefix_gcds",
        "sequence.compute.prefix_lcms",
        "sequence.compute.second_differences",
        "sequence.transform.sorted_unique",
        "sequence.transform.sort",
        "sequence.transform.reverse",
        "sequence.transform.parities",
        "sequence.transform.signs",
        "sequence.decide.arithmetic",
        "sequence.decide.geometric",
        "sequence.decide.nondecreasing",
        "sequence.decide.strictly_increasing",
        "sequence.compute.frequencies",
        "sequence.compute.zero_indices",
        "sequence.autocorrelation.aperiodic.compute",
        "sequence.autocorrelation.cyclic.compute",
    } <= ids


def test_dispatch_matches_native_sum_value() -> None:
    native = sequence_sum(IntegerSequence(values=(1, 2, 3)))
    dispatched = invoke_operation(
        "sequence.compute.sum",
        {"values": ["1", "2", "3"]},
        Catalog.open(),
    )

    assert dispatched.output == native.model_dump(mode="json")


def test_dispatch_accepts_exact_rational_autocorrelation_input() -> None:
    dispatched = invoke_operation(
        "sequence.autocorrelation.aperiodic.compute",
        {"values": [{"num": "1", "den": "2"}, {"num": "1", "den": "3"}]},
        Catalog.open(),
    )

    assert dispatched.output["convention"] == "aperiodic"
    assert dispatched.output["cells"] == [
        {"lag": -1, "value": {"num": "1", "den": "6"}},
        {"lag": 0, "value": {"num": "13", "den": "36"}},
        {"lag": 1, "value": {"num": "1", "den": "6"}},
    ]


def _rational_sequence(values: tuple[int, ...]) -> FiniteRationalSequence:
    return FiniteRationalSequence(
        values=tuple(CanonicalRational(num=value, den=1) for value in values)
    )


def test_order_shape_native_and_catalog_paths_share_rational_carrier() -> None:
    source = _rational_sequence((1, 3, 3, 2))
    native = sequence_order_shape(source)
    dispatched = invoke_operation(
        "sequence.order_shape.profile.compute",
        {"values": ["1", "3", "3", "2"]},
        Catalog.open(),
    )
    assert dispatched.output == native.model_dump(mode="json")
    assert isinstance(native.source, FiniteRationalSequence)
    assert all(
        isinstance(row.square, CanonicalRational)
        and isinstance(row.neighbor_product, CanonicalRational)
        for row in native.log_concavity_rows
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


def test_autocorrelation_value_schema_registers_canonical_rational_defs() -> None:
    schema = FiniteSequence.model_json_schema()
    assert "CanonicalRational" in json.dumps(schema)
    catalog = Catalog.open()
    descriptor = next(
        operation
        for operation in catalog.snapshot().operations
        if operation.operation_id == "sequence.autocorrelation.aperiodic.compute"
    )
    assert "CanonicalRational" in json.dumps(descriptor.input_schema)
