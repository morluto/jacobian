"""Catalog and dispatch integration for Poisson-binomial distributions."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_catalog_dispatch_reuses_request_admission_for_exact_result() -> None:
    result = invoke_operation(
        "probability.poisson_binomial.distribution.compute",
        {
            "probabilities": [
                {"num": "1", "den": "2"},
                {"num": "1", "den": "3"},
            ]
        },
        Catalog.open(),
    )

    assert result.output == {
        "probabilities": [
            {"num": "1", "den": "2"},
            {"num": "1", "den": "3"},
        ],
        "count_distribution": {
            "atoms": [
                {
                    "value": {"num": "0", "den": "1"},
                    "probability": {"num": "1", "den": "3"},
                },
                {
                    "value": {"num": "1", "den": "1"},
                    "probability": {"num": "1", "den": "2"},
                },
                {
                    "value": {"num": "2", "den": "1"},
                    "probability": {"num": "1", "den": "6"},
                },
            ]
        },
    }
