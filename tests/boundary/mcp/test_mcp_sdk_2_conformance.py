"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from pathlib import Path

import pytest
from mcp_types.methods import serialize_server_result

import jacobian.adapters.mcp.server as server_module
from jacobian.adapters.mcp.server import create_server
from jacobian.adapters.mcp.tools import math_run
from jacobian.contracts.operations import OperationResult
from jacobian.registry import CheckerRegistry


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.0.0"
    assert importlib.metadata.version("mcp-types") == "2.0.0"
    assert not inspect.iscoroutinefunction(math_run)


def test_mcp_v2_static_validation_context_errors_and_structured_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(server_module, "Context", raising=False)

    def reject_portfolio_assembly(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selected operation must not assemble the portfolio")

    monkeypatch.setattr(CheckerRegistry, "authorize", reject_portfolio_assembly)

    async def scenario() -> None:
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        server = create_server(
            tmp_path,
        )
        assert not hasattr(server_module, "Context")
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in listed.tools
            )
            invoke = next(tool for tool in listed.tools if tool.name == "math.run")
            assert set(invoke.input_schema["properties"]) == {
                "operation_id",
                "inputs",
                "payload",
            }
            assert invoke.output_schema == OperationResult.model_json_schema()
            find = next(tool for tool in listed.tools if tool.name == "math.find")
            assert set(find.input_schema["properties"]) == {"request"}
            assert find.input_schema["properties"]["request"]["discriminator"] == {
                "mapping": {
                    "inspect": "#/$defs/OperationInspectRequest",
                    "search": "#/$defs/OperationSearchRequest",
                },
                "propertyName": "op",
            }
            assert find.output_schema["type"] == "object"
            assert find.output_schema["discriminator"] == {
                "mapping": {
                    "operation": "#/$defs/OperationInspectionResult",
                    "discovery": "#/$defs/OperationSearchResult",
                    "error": "#/$defs/OperationDiscoveryError",
                },
                "propertyName": "kind",
            }
            assert len(find.output_schema["oneOf"]) == 3
            assert set(
                find.output_schema["$defs"]["OperationSearchResult"]["required"]
            ) >= {"kind", "matches", "total_matches", "truncated"}
            assert set(
                find.output_schema["$defs"]["OperationInspectionResult"]["required"]
            ) >= {"kind", "operation"}
            assert (
                find.output_schema["$defs"]["OperationDiscoveryCard"][
                    "additionalProperties"
                ]
                is False
            )
            assert (
                find.output_schema["$defs"]["OperationDiscoveryErrorDetail"][
                    "additionalProperties"
                ]
                is False
            )
            serialized_tools = serialize_server_result(
                "tools/list",
                "2026-07-28",
                listed.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
            assert serialized_tools["tools"][0]["outputSchema"]["type"] == "object"

            with pytest.raises(MCPError) as unknown:
                await client.call_tool("math.find", {"unknown_key": "rejected"})
            assert '"code": "INVALID_INPUT"' in str(unknown.value)

            mixed_find_request = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "matrix rank",
                        "operation_id": "matrix.rank.compute",
                    }
                },
            )
            assert mixed_find_request.is_error is True

            with pytest.raises(MCPError) as retired_reasoning_input:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "polynomial.expression.normalize",
                        "payload": {},
                        "reasoning_run_id": "retired",
                    },
                )
            assert '"code": "INVALID_INPUT"' in str(retired_reasoning_input.value)

            for tool_name, arguments in (
                (
                    "math.find",
                    {"request": ('{"op":"search","query":"matrix rank"}')},
                ),
                (
                    "math.run",
                    {
                        "operation_id": "polynomial.expression.normalize",
                        "payload": "{}",
                    },
                ),
            ):
                with pytest.raises(MCPError) as stringified_object:
                    await client.call_tool(tool_name, arguments)
                assert '"code": "INVALID_INPUT"' in str(stringified_object.value)

            contract_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "polynomial.expression.normalize",
                    }
                },
            )
            assert isinstance(contract_result.structured_content, dict)
            contract = contract_result.structured_content
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.expression.normalize",
                    "payload": contract["operation"]["examples"][0]["input"],
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content == OperationResult.model_validate(
                result.structured_content
            ).model_dump(mode="json")
            verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.expression_normalization.verify",
                    "payload": {
                        "normalization_uri": result.structured_content["output"][
                            "normalization_uri"
                        ]
                    },
                },
            )
            assert isinstance(verified.structured_content, dict)
            assert verified.structured_content["output"]["conclusion"] == "TRUE"

            missing_polytope_inputs = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polytope.separate",
                    "payload": {
                        "point_uri": "artifact://sha256/" + "a" * 64,
                        "generator_set_uri": "artifact://sha256/" + "b" * 64,
                    },
                },
            )
            assert isinstance(missing_polytope_inputs.structured_content, dict)
            assert (
                missing_polytope_inputs.structured_content["operation_id"]
                == "polytope.separate"
            )

            finite_coverage = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite.coverage.verify",
                    "payload": {
                        "canonicalizer_id": "finite.string.nfc@1",
                        "scope_items": ["alpha", "beta", "gamma"],
                        "pages": [
                            {"items": ["alpha"]},
                            {"items": ["beta", "gamma"]},
                        ],
                    },
                },
            )
            assert isinstance(finite_coverage.structured_content, dict)
            assert finite_coverage.structured_content["output"]["conclusion"] == "TRUE"

            magma_tables = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_magma.table.enumerate",
                    "payload": {"order": 1},
                },
            )
            assert isinstance(magma_tables.structured_content, dict)
            assert magma_tables.structured_content["output"]["enumerated_count"] == 1

            invalid_countermodel = await client.call_tool(
                "math.run",
                {
                    "operation_id": "universal_algebra.search.countermodel",
                    "payload": {},
                },
            )
            assert isinstance(invalid_countermodel.structured_content, dict)
            assert (
                invalid_countermodel.structured_content["operation_id"]
                == "universal_algebra.search.countermodel"
            )

            law_evaluation = await client.call_tool(
                "math.run",
                {
                    "operation_id": "universal_algebra.evaluate_laws",
                    "payload": {
                        "problem": {
                            "structure": {"order": 1, "table": [[0]]},
                            "laws": [
                                {
                                    "law_id": "idempotence",
                                    "variables": ["x"],
                                    "left": {
                                        "kind": "PRODUCT",
                                        "left": {
                                            "kind": "VARIABLE",
                                            "variable": "x",
                                        },
                                        "right": {
                                            "kind": "VARIABLE",
                                            "variable": "x",
                                        },
                                    },
                                    "right": {
                                        "kind": "VARIABLE",
                                        "variable": "x",
                                    },
                                }
                            ],
                        }
                    },
                },
            )
            assert isinstance(law_evaluation.structured_content, dict)
            law_certificate = law_evaluation.structured_content["output"][
                "certificate_uri"
            ]
            law_verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "universal_algebra.law_evaluation.verify",
                    "payload": {"certificate_uri": law_certificate},
                },
            )
            assert isinstance(law_verified.structured_content, dict)
            assert law_verified.structured_content["output"]["conclusion"] == "TRUE"

            with pytest.raises(MCPError) as missing_resource:
                await client.read_resource("artifact://sha256/" + "f" * 64)
            assert missing_resource.value.code == -32602
            assert "requested Jacobian resource does not exist" in str(
                missing_resource.value
            )

    asyncio.run(scenario())
