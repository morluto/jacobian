"""Public factorization output retains the native canonical factor order."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._models import PolynomialFactorizationResult
from jacobian.math.polynomials.operations import (
    polynomial_factorization,
    verify_polynomial_factorization,
)


def test_factorization_dispatch_output_round_trips_without_reordering() -> None:
    # x^2 + 12*x + 20 = (x + 2)*(x + 10); numeric 2 precedes 10.
    source = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]},
                {"coefficient": {"num": "12", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "20", "den": "1"}, "exponents": [0]},
            ]
        },
    }
    output = invoke_operation(
        "polynomial.factor.compute", {"polynomial": source}, Catalog.open()
    ).output

    decoded = PolynomialFactorizationResult.model_validate_json(json.dumps(output))
    assert decoded == polynomial_factorization(decoded.polynomial)
    assert verify_polynomial_factorization(decoded)
    assert output["polynomial"] == output["reconstructed"] == source
    assert output["coefficient"] == {"num": "1", "den": "1"}
    assert output["factors"] == [
        {
            "factor": {
                "domain": "QQ",
                "variables": ["x"],
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [1],
                        },
                        {
                            "coefficient": {"num": constant, "den": "1"},
                            "exponents": [0],
                        },
                    ]
                },
            },
            "multiplicity": 1,
        }
        for constant in ("2", "10")
    ]
