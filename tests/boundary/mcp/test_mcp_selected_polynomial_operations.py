from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server
from jacobian.registry import CheckerRegistry


def test_polynomial_positivity_loads_only_the_selected_path(
    mcp_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_installation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "selected polynomial operations must not install portfolio"
        )

    monkeypatch.setattr(CheckerRegistry, "authorize", reject_installation)

    polynomial = {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variable": "x",
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "2", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
    interval = {
        "interval_schema_version": "1",
        "lo": {"num": "0", "den": "1"},
        "hi": {"num": "1", "den": "1"},
    }

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(mcp_state), raise_exceptions=True) as client:
            decided = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.interval.positivity.decide",
                    "payload": {"polynomial": polynomial, "interval": interval},
                },
            )
            assert isinstance(decided.structured_content, dict)
            decision = decided.structured_content["output"]
            verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.interval.positivity.verify",
                    "payload": {
                        "polynomial": polynomial,
                        "interval": interval,
                        "claimed_positive": decision["positive"],
                        "claimed_sign_changes_at_lo": decision["sign_changes_at_lo"],
                        "claimed_sign_changes_at_hi": decision["sign_changes_at_hi"],
                        "claimed_roots_in_open_interval": decision[
                            "roots_in_open_interval"
                        ],
                        "claimed_endpoint_root": decision["endpoint_root"],
                    },
                },
            )
            assert isinstance(verified.structured_content, dict)
            assert verified.structured_content["output"]["conclusion"] == "TRUE"

            enclosed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.interval.enclose",
                    "payload": {"polynomial": polynomial, "interval": interval},
                },
            )
            assert isinstance(enclosed.structured_content, dict)
            enclosure = enclosed.structured_content["output"]
            enclosure_verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.interval.enclosure.verify",
                    "payload": {
                        "polynomial": polynomial,
                        "interval": interval,
                        "claimed_bernstein_coefficients": enclosure[
                            "bernstein_coefficients"
                        ],
                        "claimed_lo": enclosure["lo"],
                        "claimed_hi": enclosure["hi"],
                    },
                },
            )
            assert isinstance(enclosure_verified.structured_content, dict)
            assert (
                enclosure_verified.structured_content["output"]["conclusion"] == "TRUE"
            )

            for operation_id in (
                "polynomial.map.evaluate",
                "polynomial.map.compute_jacobian",
                "polynomial.map.keller_condition.verify",
                "polynomial.map.collision_witness",
                "polynomial.map.collision.search",
                "polynomial.map.collision.verify",
                "polynomial.map.collision_evidence.verify",
                "polynomial.map.inverse.refute_by_collision",
                "polynomial.identity.verify",
                "polynomial.rational_function.identity.verify",
                "polynomial.map.inverse.candidate_synthesize",
                "polynomial.map.inverse.verify",
                "polynomial.system.solution.verify",
                "polynomial.system.rational_solution.search",
                "polynomial.jacobian_degree_slice.system.materialize",
                "polynomial.nullstellensatz.infeasibility_certificate.verify",
            ):
                invalid = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {}},
                )
                assert isinstance(invalid.structured_content, dict)
                assert invalid.structured_content["operation_id"] == operation_id

    asyncio.run(scenario())
