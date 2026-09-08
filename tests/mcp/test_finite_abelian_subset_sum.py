"""Product-group profiles preserve their canonical values over MCP."""

import asyncio
import json

from tests.mcp.test_direct_operations import _server

from jacobian.math.combinatorics.additive._subset_sum_residue import (
    SubsetSumResidueProfileResult,
    subset_sum_residue_profile,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)
from mcp import Client


def test_product_group_native_mcp_parity() -> None:
    async def scenario() -> None:
        operation = "additive.subset_sum.residue_profile.compute"
        async with Client(_server(operation), raise_exceptions=True) as client:
            for moduli, coordinates, include_empty in (
                ((2, 2), ((1, 0), (0, 1)), False),
                ((), ((), ()), True),
                ((1024,), ((0,),) * 4095, True),
            ):
                group = FiniteAbelianProductGroup(moduli=moduli)
                sequence = tuple(
                    FiniteAbelianGroupElement(group=group, coordinates=c)
                    for c in coordinates
                )
                expected = subset_sum_residue_profile(
                    None, None, include_empty, group=group, sequence=sequence
                )
                response = await client.call_tool(
                    operation,
                    {
                        "group": group.model_dump(mode="json"),
                        "sequence": [v.model_dump(mode="json") for v in sequence],
                        "include_empty_subset": include_empty,
                    },
                )
                assert response.structured_content is not None
                actual = SubsetSumResidueProfileResult.model_validate_json(
                    json.dumps(response.structured_content)
                )
                assert actual == expected
                assert actual.group_rows is not None
                # A returned parent-bound element composes directly as the next
                # indexed source without changing its group presentation.
                next_result = subset_sum_residue_profile(
                    None,
                    None,
                    True,
                    group=group,
                    sequence=(actual.group_rows[0].element,),
                )
                assert next_result.group_rows is not None
                assert next_result.group_rows[0].multiplicity == 2

    asyncio.run(scenario())
