"""Real profiles demonstrate the scopes advertised by discovery."""

import asyncio
import json
from fractions import Fraction
from itertools import combinations
from typing import Any

from mcp.types import TextContent

from jacobian.math.combinatorics.additive._models import OrderedDifferenceProfileResult
from jacobian.math.geometry._models import CircumradiusProfileResult
from jacobian.mcp.server import create_server
from mcp import Client


def _assert_differences(
    output: dict[str, Any], coordinates: tuple[tuple[int, ...], ...]
) -> None:
    decoded = OrderedDifferenceProfileResult.model_validate_json(json.dumps(output))
    expected = {
        (tuple(a - b for a, b in zip(left, right, strict=True)), i, j)
        for i, left in enumerate(coordinates)
        for j, right in enumerate(coordinates)
        if i != j
    }
    actual = [
        (entry.difference.coordinates, pair.left_index, pair.right_index)
        for entry in decoded.entries
        for pair in entry.pairs
    ]
    assert (
        len(actual)
        == decoded.total_ordered_pairs
        == len(coordinates) * (len(coordinates) - 1)
    )
    assert set(actual) == expected
    assert all(entry.multiplicity == len(entry.pairs) for entry in decoded.entries)


def test_difference_profiles_allow_mixed_norms_and_retain_source_bounds() -> None:
    async def scenario() -> None:
        operation_id = "additive.ordered_difference_profile.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert not inspected.is_error
            card = inspected.structured_content["operation"]
            assert "share one dimension" in card["description"]
            example = card["examples"][0]["input"]
            coordinates = tuple(
                tuple(int(value) for value in vector["coordinates"])
                for vector in example["vectors"]["vectors"]
            )
            assert (
                len({sum(value * value for value in vector) for vector in coordinates})
                > 1
            )
            result = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": example}
            )
            assert not result.is_error
            _assert_differences(result.structured_content["output"], coordinates)
            for vectors in (((0, 0), (1, 0), (0, 1), (1, 1)), ((-999999,), (999999,))):
                payload = {
                    "vectors": {
                        "vectors": [
                            {"coordinates": [str(value) for value in vector]}
                            for vector in vectors
                        ]
                    }
                }
                result = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert not result.is_error
                _assert_differences(result.structured_content["output"], vectors)
            for invalid_vectors in (((0, 0), (1,)), ((0,), (1000000,))):
                payload = {
                    "vectors": {
                        "vectors": [
                            {"coordinates": [str(value) for value in vector]}
                            for vector in invalid_vectors
                        ]
                    }
                }
                rejected = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert rejected.is_error
                content = rejected.content[0]
                assert isinstance(content, TextContent)
                failure = json.loads(
                    content.text.removeprefix("Error executing tool math.run: ")
                )
                assert failure["code"] == "INVALID_REQUEST"

    asyncio.run(scenario())


def test_circumradius_profiles_mark_collinear_triples_and_allow_concyclic_quadruples() -> (
    None
):
    async def scenario() -> None:
        operation_id = "geometry.points.circumradius_profile.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert not inspected.is_error
            assert (
                "general position is not checked"
                in inspected.structured_content["operation"]["description"]
            )
            for coordinates in (
                ((0, 0), (1, 0), (2, 0)),
                ((0, 0), (1, 0), (0, 1), (1, 1)),
            ):
                payload = {
                    "points": [
                        {
                            "x": {"num": str(x), "den": "1"},
                            "y": {"num": str(y), "den": "1"},
                        }
                        for x, y in coordinates
                    ]
                }
                result = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert not result.is_error
                decoded = CircumradiusProfileResult.model_validate_json(
                    json.dumps(result.structured_content["output"])
                )
                assert {entry.indices for entry in decoded.entries} == set(
                    combinations(range(len(coordinates)), 3)
                )
                assert tuple(
                    (point.x.as_fraction(), point.y.as_fraction())
                    for point in decoded.points
                ) == tuple((Fraction(x), Fraction(y)) for x, y in coordinates)
                if len(coordinates) == 3:
                    assert decoded.entries[0].is_degenerate
                    assert decoded.entries[0].radius_squared is None
                else:
                    # All four points lie on the circle centered at (1/2,1/2).
                    assert len(decoded.entries) == 4
                    for entry in decoded.entries:
                        assert not entry.is_degenerate
                        assert entry.radius_squared is not None
                        assert entry.radius_squared.as_fraction() == Fraction(1, 2)

    asyncio.run(scenario())
