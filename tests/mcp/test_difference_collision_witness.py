"""Live discovery and execution expose a complete source-checkable collision."""

import asyncio
import json
from collections import Counter

from jsonschema import validate

from jacobian.math.combinatorics.additive._models import OrderedDifferenceProfileResult
from jacobian.math.combinatorics.additive.operations import (
    verify_ordered_difference_profile,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_mcp_collision_schema_execution_and_round_trip() -> None:
    async def scenario() -> None:
        operation_id = "additive.ordered_difference_profile.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            found = await client.call_tool("math.find", {"operation_id": operation_id})
            assert not found.is_error
            assert found.structured_content is not None
            schema = found.structured_content["operation"]["output_schema"]
            collision = schema["$defs"]["OrderedDifferenceCollision"]
            assert set(collision["required"]) == {"difference", "pairs"}
            assert collision["properties"]["pairs"]["minItems"] == 2
            assert collision["properties"]["pairs"]["maxItems"] == 2

            for source in (
                ((0, 0), (1, 0), (0, 1), (1, 1)),
                ((0,), (1,), (2,)),
                ((0,), (1,), (3,)),
            ):
                payload = {
                    "vectors": {
                        "vectors": [
                            {"coordinates": [str(c) for c in v]} for v in source
                        ]
                    }
                }
                executed = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert not executed.is_error
                assert executed.structured_content is not None
                output = executed.structured_content["output"]
                validate(output, schema)
                decoded = OrderedDifferenceProfileResult.model_validate_json(
                    json.dumps(output)
                )
                assert decoded.model_dump(mode="json") == output
                assert verify_ordered_difference_profile(decoded)
                counts = Counter(
                    tuple(a - b for a, b in zip(left, right, strict=True))
                    for i, left in enumerate(source)
                    for j, right in enumerate(source)
                    if i != j
                )
                repeated = sorted(d for d, count in counts.items() if count > 1)
                witness = output["first_collision"]
                if not repeated:
                    assert witness is None
                    continue
                assert witness is not None
                difference = tuple(int(c) for c in witness["difference"]["coordinates"])
                assert difference == repeated[0]
                pairs = [(p["left_index"], p["right_index"]) for p in witness["pairs"]]
                expected_pairs = [
                    (i, j)
                    for i, left in enumerate(source)
                    for j, right in enumerate(source)
                    if i != j
                    and tuple(a - b for a, b in zip(left, right, strict=True))
                    == difference
                ]
                assert pairs == expected_pairs[:2]
                assert len(set(pairs)) == 2

    asyncio.run(scenario())
