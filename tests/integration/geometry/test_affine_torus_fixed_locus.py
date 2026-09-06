"""Integration checks for affine-torus dispatch and catalog projection."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.affine_tori import (
    AffineTorusFixedLocusResult,
    RationalAffineTorusMap,
)
from jacobian.math.geometry.affine_tori._models import AffineTorusFixedLocusRequest


def _source(
    linear_part: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> RationalAffineTorusMap:
    dimension = len(linear_part)
    payload: dict[str, Any] = {
        "torus": {"dimension": dimension},
        "linear_part": {
            "row_count": dimension,
            "column_count": dimension,
            "entries": linear_part,
        },
        "translation": {
            "torus": {"dimension": dimension},
            "coordinates": [
                {"num": value.numerator, "den": value.denominator}
                for value in translation
            ],
        },
    }
    return RationalAffineTorusMap.model_validate(payload)


def test_empty_outcome_round_trips_through_public_dispatch() -> None:
    source = _source(((1,),), (Fraction(1, 3),))
    payload = {"affine_map": source.model_dump(mode="json")}
    dispatched = invoke_operation(
        "affine_torus.fixed_locus.compute",
        payload,
        Catalog.open(),
    )
    restored = AffineTorusFixedLocusResult.model_validate_json(
        json.dumps(dispatched.output)
    )

    assert restored.outcome.status == "EMPTY"
    assert restored.outcome.obstruction.coefficients == (1,)
    assert restored.outcome.obstruction_pairing.as_fraction() == Fraction(1, 3)
    assert restored.model_dump(mode="json") == dispatched.output


def test_public_tool_validates_and_executes_its_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_torus.fixed_locus.compute"
    )
    request = AffineTorusFixedLocusRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    schema = tool.request_type.model_json_schema()
    linear_schema = schema["$defs"]["RationalAffineTorusMap"]["properties"][
        "linear_part"
    ]

    result = tool.run(request)

    assert "linear part is square" in tool.examples[0].description
    assert "same standard torus" in tool.examples[0].description
    assert linear_schema["properties"]["row_count"]["maximum"] == 32
    assert linear_schema["properties"]["column_count"]["maximum"] == 32
    assert linear_schema["properties"]["entries"]["maxItems"] == 32
    assert linear_schema["properties"]["entries"]["items"]["maxItems"] == 32
    assert isinstance(result, AffineTorusFixedLocusResult)
    assert result.outcome.status == "NONEMPTY"
    assert result.outcome.fixed_locus.finite_components.component_count == 2
