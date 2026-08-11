from __future__ import annotations

import pytest

from jacobian.contracts.matrix_operations import LatticeReductionRequest
from jacobian.domains.matrix_lattice.lll_protocol import parse_lll_worker_response


def test_lll_response_is_closed_and_bound_to_source_dimensions() -> None:
    request = LatticeReductionRequest.model_validate(
        {"basis": {"entries": [["1", "0"], ["0", "1"]]}}
    )
    response = {
        "protocol": "jacobian.flint-lll-worker/v1",
        "result": {
            "reduced_basis": {"entries": [["1", "0"], ["0", "1"]]},
            "transformation": {"entries": [["1", "0"], ["0", "1"]]},
            "rank": 2,
        },
    }
    parsed = parse_lll_worker_response(response, request=request)
    assert parsed.result.rank == 2

    with pytest.raises(ValueError, match="dimensions do not match"):
        parse_lll_worker_response(
            {
                **response,
                "result": {
                    **response["result"],
                    "reduced_basis": {"entries": [["1"], ["0"]]},
                },
            },
            request=request,
        )

    with pytest.raises(ValueError, match="invalid LLL"):
        parse_lll_worker_response(
            {**response, "unexpected": True},
            request=request,
        )
