"""Dispatch boundaries for high-order bounded QQ splitting fields."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation

_S6 = {
    "polynomial": {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [6]},
                {"coefficient": {"num": "-1", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "-1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
}


def test_catalog_dispatch_rejects_degree_six_splitting_field() -> None:
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            "number_field.polynomial.splitting_field.compute", _S6, Catalog.open()
        )
