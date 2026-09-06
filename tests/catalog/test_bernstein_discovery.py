"""The basis transform has a distinct discovery intent from interval enclosure."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


@pytest.mark.parametrize(
    "query",
    [
        "Convert a rational polynomial on a rational box to its exact tensor-product Bernstein basis coefficients at a specified multidegree.",
        "Exact Bernstein coefficient tensor for a bivariate polynomial on a rational rectangle, retaining its axes and degree elevation.",
    ],
)
def test_bernstein_coordinates_are_discoverable(query: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=5)).matches
    assert "polynomial.bernstein.coefficients.compute" in {
        match.operation_id for match in matches
    }
