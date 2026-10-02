"""Real MCP producer values feed unchanged into exact differentiation."""

import asyncio
import json
from fractions import Fraction

from mcp.types import TextContent

from jacobian.math.polynomials import rational_polynomial_integral
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.mcp.server import create_server
from mcp import Client

_OPERATION = "polynomial.rational.compute.derivative"


def _source(numerator: int, denominator: int, exponent: int) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": numerator, "den": denominator},
                        "exponents": [exponent],
                    }
                ]
            },
        }
    )


def test_live_derivative_output_composes_without_reconstruction() -> None:
    async def scenario() -> None:
        coefficient = 10**256 - 1
        async with Client(create_server(), raise_exceptions=False) as client:
            first = await client.call_tool(
                "math.run",
                {
                    "operation_id": _OPERATION,
                    "payload": {
                        "polynomial": _source(coefficient, 1, 127).model_dump(
                            mode="json"
                        )
                    },
                },
            )
            assert not first.is_error
            assert first.structured_content is not None
            second = await client.call_tool(
                "math.run",
                {
                    "operation_id": _OPERATION,
                    "payload": {
                        "polynomial": first.structured_content["output"]["derivative"]
                    },
                },
            )
            assert not second.is_error
            assert second.structured_content is not None
            value = RationalPolynomial.model_validate_json(
                json.dumps(second.structured_content["output"]["derivative"])
            )
            assert value == _source(16002 * coefficient, 1, 125)

    asyncio.run(scenario())


def test_live_derivative_consumes_native_antiderivatives_and_reports_overflow() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            for source in (_source(1, 1, 127), _source(1, 10**256 - 1, 126)):
                primitive = rational_polynomial_integral(source).antiderivative
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": _OPERATION,
                        "payload": {"polynomial": primitive.model_dump(mode="json")},
                    },
                )
                assert not result.is_error
                assert result.structured_content is not None
                value = RationalPolynomial.model_validate_json(
                    json.dumps(result.structured_content["output"]["derivative"])
                )
                assert value == source
                term = value.polynomial.terms[0]
                assert term.coefficient.as_fraction() == Fraction(
                    1, source.polynomial.terms[0].coefficient.den
                )
            overflowing = _source(10**32768 - 1, 1, 2)
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": _OPERATION,
                    "payload": {"polynomial": overflowing.model_dump(mode="json")},
                },
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            text = refused.content[0].text
            diagnostic = json.loads(text[text.index("{") :])
            assert diagnostic["stage"] == "resource_admission"
            assert "polynomial.derivative_output_bound" in text

    asyncio.run(scenario())
