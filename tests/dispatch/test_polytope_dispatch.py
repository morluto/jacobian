"""Dispatch boundary for the exact facet-incidence profile operation."""

import math
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.polytope._models import MAX_COMPUTED_FACETS


def _cyclic_facet_count(vertices: int, dimension: int) -> int:
    """Upper-bound-theorem facet count attained by a moment-curve hull."""
    half = [
        math.comb(vertices - dimension - 1 + index, index)
        for index in range(dimension // 2 + 1)
    ]
    total = 2 * sum(half)
    if dimension % 2 == 0:
        total -= half[-1]
    return total


def _moment_curve_size_beyond_facet_cap(dimension: int) -> int:
    """Smallest moment-curve vertex count whose facet count exceeds the cap."""
    count = dimension + 1
    while _cyclic_facet_count(count, dimension) <= MAX_COMPUTED_FACETS:
        count += 1
    return count


def _moment_curve_payload(count: int, dimension: int) -> dict[str, Any]:
    return {
        "vertices": [
            {
                "coordinates": [
                    {"num": str(t**k), "den": "1"} for k in range(1, dimension + 1)
                ]
            }
            for t in range(1, count + 1)
        ]
    }


def test_dispatch_computes_the_unit_square_facet_profile() -> None:
    result = invoke_operation(
        "polytope.facets.compute",
        {
            "vertices": [
                {"coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}]},
                {"coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}]},
                {"coordinates": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}]},
                {"coordinates": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}]},
            ]
        },
        Catalog.open(),
    )

    assert result.operation_id == "polytope.facets.compute"
    assert result.output["dimension"] == 2
    assert len(result.output["facets"]) == 4


def test_dispatch_rejects_a_profile_beyond_the_facet_cap_as_invalid_request() -> None:
    """A moment-curve hull is a cyclic polytope, so its facet count is the
    upper-bound-theorem maximum for its vertex count; the payload size is
    derived so that count exceeds MAX_COMPUTED_FACETS. Admission
    materializes that enumeration during request validation, so math.run
    must answer with the typed invalid-request error -- never leak an
    execution-time ValueError as a host exception."""
    payload = _moment_curve_payload(
        count=_moment_curve_size_beyond_facet_cap(dimension=7), dimension=7
    )

    with pytest.raises(OperationRequestValidationError) as exc_info:
        invoke_operation("polytope.facets.compute", payload, Catalog.open())

    assert (
        f"{MAX_COMPUTED_FACETS}-facet result bound" in exc_info.value.errors()[0]["msg"]
    )


def test_dispatch_admits_the_seven_simplex_with_interior_rows_at_the_cap_budget() -> (
    None
):
    """A seven-simplex plus four distinct strict interior points has only
    eight facets; admission must accept it even though no row-count upper
    bound could have proven that."""
    simplex = [
        {"coordinates": [{"num": "0", "den": "1"}] * 7},
        *[
            {
                "coordinates": [
                    {"num": "1" if index == axis else "0", "den": "1"}
                    for axis in range(7)
                ]
            }
            for index in range(7)
        ],
    ]
    payload = {
        "vertices": simplex
        + [{"coordinates": [{"num": "1", "den": str(k)}] * 7} for k in range(9, 13)]
    }

    result = invoke_operation("polytope.facets.compute", payload, Catalog.open())

    assert len(result.output["facets"]) == 8
