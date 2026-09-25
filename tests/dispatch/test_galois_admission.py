"""Dispatch boundaries for high-order bounded QQ splitting fields."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

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


def test_catalog_dispatch_accepts_the_degree_six_full_symmetric_group() -> None:
    result = invoke_operation(
        "number_field.polynomial.splitting_field.compute", _S6, Catalog.open()
    )

    assert result.output["field"]["degree"] == 720
    assert len(result.output["field"]["basis_labels"]) == 720
