from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.sequences.core.operations import sequence_sum
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
    } <= ids


def test_dispatch_matches_native_sum_value() -> None:
    native = sequence_sum(IntegerSequence(values=("1", "2", "3")))
    dispatched = invoke_operation(
        "sequence.compute.sum",
        {"values": ["1", "2", "3"]},
        Catalog.open(),
    )

    assert dispatched.output == native.model_dump(mode="json")
