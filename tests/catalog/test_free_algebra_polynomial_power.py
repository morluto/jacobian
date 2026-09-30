"""Published power admission reports its public request field."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation


def test_overlong_power_reports_the_polynomial_input() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            "free_algebra.polynomial.power.compute",
            {
                "polynomial": {
                    "alphabet": ["a"],
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "word": ["a"] * 33}
                    ],
                },
                "exponent": 2,
            },
            Catalog.open(),
        )
    assert error.value.errors()[0]["type"].endswith("power_result_word_length_budget")
    assert error.value.errors()[0]["loc"] == ("polynomial",)
