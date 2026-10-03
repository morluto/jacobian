"""Public producer-to-consumer composition for periodic congruence unions.

``congruence.periodic_union.measure.compute`` publishes a canonical ``source``
that ``congruence.periodic_union.profile.compute`` consumes. The
advertised-example runner invokes each operation with its own authored input,
and the conformance sweep derives both from the catalog, so neither composes
the actual public pair. This is the only check that a serialized measure
output is still accepted by the profile operation unchanged.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
)


def test_published_profile_consumes_serialized_measure_source_unchanged() -> None:
    catalog = Catalog.open()
    measure_output = invoke_operation(
        "congruence.periodic_union.measure.compute",
        {"subsets": [{"modulus": "5", "residues": ["0"]}], "complement": False},
        catalog,
    ).output

    operation = catalog.operation("congruence.periodic_union.profile.compute")
    assert operation is not None
    assert operation.request_type is PeriodicCongruenceUnionProfileRequest
    result = invoke_operation(
        operation.operation_id,
        measure_output["source"],
        catalog,
    )
    validated = PeriodicCongruenceUnionProfileResult.model_validate_json(
        json.dumps(result.output)
    )

    assert validated.source.model_dump(mode="json") == measure_output["source"]
    assert validated.common_period == 5
    assert validated.occupied_count == 1
    assert validated.density.as_fraction() == Fraction(1, 5)
    assert validated.occupied_residues == (0,)
