"""The discovered best-response example agrees with its exact row minima."""

import asyncio
import json
from fractions import Fraction

from jacobian.math.logic.games.finite._models import BestResponseResult
from jacobian.mcp.server import create_server
from mcp import Client


def test_discovered_best_response_example_explains_tie() -> None:
    async def scenario() -> None:
        operation_id = "game_theory.best_response.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            found = await client.call_tool("math.find", {"operation_id": operation_id})
            assert not found.is_error
            assert found.structured_content is not None
            example = found.structured_content["operation"]["examples"][0]
            matrix = example["input"]["payoff_matrix"]
            entries = [
                Fraction(int(value["num"]), int(value["den"]))
                for value in matrix["entries"]
            ]
            rows = [
                entries[start : start + matrix["n_cols"]]
                for start in range(0, len(entries), matrix["n_cols"])
            ]
            minima = [min(row) for row in rows]
            assert minima == [Fraction(0), Fraction(0)]
            assert example["description"] == (
                "Both rows have worst-case payoff 0; the tie selects the first row (index 0)."
            )
            executed = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": example["input"]}
            )
            assert not executed.is_error
            assert executed.structured_content is not None
            output = executed.structured_content["output"]
            restored = BestResponseResult.model_validate_json(json.dumps(output))
            assert restored.value.as_fraction() == max(minima) == Fraction(0)
            assert restored.best_row == minima.index(max(minima)) == 0
            assert restored.model_dump(mode="json") == output

    asyncio.run(scenario())
