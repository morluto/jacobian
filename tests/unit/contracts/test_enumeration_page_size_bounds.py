"""Focused tests for the bounded enumeration page_size cap (PR3 atomic fix).

``GraphEnumerationRequest`` and ``MatrixEnumerationRequest`` cap ``page_size``
at 1..256 so a runaway page request cannot exhaust the bounded enumeration
backend.  The lower bound (1) and upper bound (256) are both enforced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.plugin_graphs import GraphEnumerationRequest
from jacobian.contracts.plugin_matrices import MatrixEnumerationRequest


def _graph_payload(page_size: int) -> dict:
    return {"bounds": {"vertices": 4}, "page_size": page_size}


def _matrix_payload(page_size: int) -> dict:
    return {
        "bounds": {"rows": 2, "cols": 2, "entries": (1, 2)},
        "page_size": page_size,
    }


@pytest.mark.parametrize(
    "model,payload",
    [
        (GraphEnumerationRequest, _graph_payload),
        (MatrixEnumerationRequest, _matrix_payload),
    ],
)
def test_page_size_accepts_one_through_256(model, payload) -> None:
    model.model_validate(payload(1))
    model.model_validate(payload(256))


@pytest.mark.parametrize(
    "model,payload",
    [
        (GraphEnumerationRequest, _graph_payload),
        (MatrixEnumerationRequest, _matrix_payload),
    ],
)
def test_page_size_rejects_above_256(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload(257))


@pytest.mark.parametrize(
    "model,payload",
    [
        (GraphEnumerationRequest, _graph_payload),
        (MatrixEnumerationRequest, _matrix_payload),
    ],
)
def test_page_size_rejects_zero(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload(0))
