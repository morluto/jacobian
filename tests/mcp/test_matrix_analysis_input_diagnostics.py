"""Matrix analysis reports caller paths through actual SDK error/recovery calls."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.types import TextContent

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.math.matrices.analysis._tools import TOOLS
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    EmbeddedRealSimpleNumberFieldMatrix,
)
from jacobian.math.number_theory.number_fields import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    embeddings,
)
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client

_INERTIA = "matrix.inertia.compute"
_FARKAS = "matrix.farkas_certificate.check"


def _ratio(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _inputs() -> list[tuple[str, dict[str, Any], tuple[str | int, ...]]]:
    farkas = {
        "rhs_vector": [_ratio(0), _ratio(-1)],
        "multipliers": [_ratio(1), _ratio(1)],
    }
    presentation = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -2))
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    positive = SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=1, den=1),
        ),
    )
    embedded = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding, entries=((positive,),)
    )
    rows = MAX_RATIONAL_MATRIX_ORDER + 1
    large_farkas = {
        "constraint_matrix": {
            "entries": [[_ratio(1)], [_ratio(-1)]] + [[_ratio(0)]] * (rows - 2)
        },
        "rhs_vector": [_ratio(0), _ratio(-1)] + [_ratio(0)] * (rows - 2),
        "multipliers": [_ratio(1), _ratio(1)] + [_ratio(0)] * (rows - 2),
    }
    return [
        (
            _FARKAS,
            {
                **farkas,
                "constraint_matrix": {"entries": [[_ratio(1)], [_ratio(-1)]]},
            },
            ("constraint_matrix", "entries", 0, 0, "den"),
        ),
        (
            _FARKAS,
            {
                **farkas,
                "constraint_matrix": {
                    "row_count": 2,
                    "column_count": 1,
                    "entries": [
                        {"row": 0, "column": 0, "value": _ratio(1)},
                        {"row": 1, "column": 0, "value": _ratio(-1)},
                    ],
                },
            },
            ("constraint_matrix", "entries", 0, "value", "den"),
        ),
        (
            _FARKAS,
            large_farkas,
            ("constraint_matrix", "entries", 0, 0, "den"),
        ),
        (
            _INERTIA,
            {"matrix": {"entries": [[_ratio(1)]]}},
            ("matrix", "entries", 0, 0, "den"),
        ),
        (
            _INERTIA,
            {"matrix": embedded.model_dump(mode="json")},
            ("matrix", "entries", 0, 0, "coefficients_ascending", 0, "den"),
        ),
        (
            _INERTIA,
            {"matrix": embedded.model_dump(mode="json", exclude={"domain"})},
            ("matrix", "entries", 0, 0, "coefficients_ascending", 0, "den"),
        ),
    ]


@pytest.mark.parametrize("direct", [False, True])
def test_analysis_matrix_paths_counts_and_recovery(direct: bool) -> None:
    async def scenario() -> None:
        catalog = Catalog(
            tuple(tool for tool in TOOLS if tool.operation_id in {_INERTIA, _FARKAS})
        )
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=direct_operation_tools(catalog) if direct else (),
        )

        async with Client(server, raise_exceptions=False) as client:

            async def run(operation: str, payload: dict[str, Any]) -> Any:
                return await client.call_tool(
                    operation if direct else "math.run",
                    payload
                    if direct
                    else {"operation_id": operation, "payload": payload},
                )

            def diagnostic(operation: str, reply: Any) -> dict[str, Any]:
                assert reply.is_error
                assert reply.structured_content is None
                assert len(reply.content) == 1
                content = reply.content[0]
                assert isinstance(content, TextContent)
                tool = operation if direct else "math.run"
                data = json.loads(
                    content.text.removeprefix(f"Error executing tool {tool}: ")
                )
                assert isinstance(data, dict)
                assert data["code"] == "INVALID_REQUEST"
                assert data["operation_id"] == operation
                for issue in data["errors"]:
                    assert all(
                        not isinstance(part, str)
                        or ("function-" not in part and "json-or-python[" not in part)
                        for part in issue["location"]
                    )
                return data

            for operation, valid, location in _inputs():
                inspected = await client.call_tool(
                    "math.find", {"operation_id": operation}
                )
                assert inspected.structured_content is not None
                descriptor = inspected.structured_content["operation"]
                Draft202012Validator(descriptor["input_schema"]).validate(valid)
                invalid = copy.deepcopy(valid)
                target: Any = invalid
                for key in location[:-1]:
                    target = target[key]
                target[location[-1]] = "bad"
                error = diagnostic(operation, await run(operation, invalid))
                assert len(error["errors"]) == 1
                assert error["errors"][0]["location"] == list(location)
                assert error["omitted_error_count"] == 0

                recovered = await run(operation, valid)
                assert not recovered.is_error
                assert recovered.structured_content is not None
                output = (
                    recovered.structured_content
                    if direct
                    else recovered.structured_content["output"]
                )
                Draft202012Validator(descriptor["output_schema"]).validate(output)
                if operation == _FARKAS:
                    assert output["valid"] is True
                    assert output["y_t_a"] == [_ratio(0)]
                    assert output["y_t_b"] == _ratio(-1)
                else:
                    assert (
                        output["n_positive"],
                        output["n_negative"],
                        output["n_zero"],
                    ) == (1, 0, 0)

            for operation, field in [
                (_FARKAS, "constraint_matrix"),
                (_INERTIA, "matrix"),
            ]:
                payload: dict[str, Any] = {
                    field: {
                        "entries": [[{"num": "1", "den": "bad"}] * 9 for _ in range(9)]
                    },
                }
                if operation == _FARKAS:
                    payload.update(
                        rhs_vector=[_ratio(0)] * 9, multipliers=[_ratio(0)] * 9
                    )
                error = diagnostic(operation, await run(operation, payload))
                assert len(error["errors"]) == 64
                assert error["omitted_error_count"] == 17

            for operation, prefixes in [
                (_FARKAS, ("Dense matrix: ", "Sparse matrix: ")),
                (_INERTIA, ("Rational matrix: ", "Embedded real matrix: ")),
            ]:
                field = "constraint_matrix" if operation == _FARKAS else "matrix"
                ambiguous: dict[str, Any] = {field: {"entries": [{}]}}
                if operation == _FARKAS:
                    ambiguous.update(rhs_vector=[_ratio(0)], multipliers=[_ratio(0)])
                error = diagnostic(operation, await run(operation, ambiguous))
                assert all(
                    item["message"].startswith(prefixes) for item in error["errors"]
                )
                assert all(
                    any(item["message"].startswith(prefix) for item in error["errors"])
                    for prefix in prefixes
                )
                assert all(
                    part not in {"dense", "sparse", "rational", "algebraic"}
                    for item in error["errors"]
                    for part in item["location"]
                )

    asyncio.run(scenario())
