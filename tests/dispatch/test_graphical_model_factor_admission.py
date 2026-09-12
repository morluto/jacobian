"""Public-dispatch admission for dense exact graphical factors."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation


def _large_integer_factor_payload() -> dict[str, object]:
    value = {"num": "9" * 256, "den": "1"}
    domain_sizes = [2] * 12
    return {
        "left": {
            "variables": list(range(6)),
            "domain_sizes": domain_sizes,
            "table": [value] * 64,
        },
        "right": {
            "variables": list(range(6, 12)),
            "domain_sizes": domain_sizes,
            "table": [value] * 64,
        },
    }


def _large_denominator_marginal_payload() -> dict[str, object]:
    domain_sizes = [32, 8, 16]
    denominators = [10**255 + 2 * index + 1 for index in range(32)]
    table = [
        {"num": "1", "den": str(denominators[index // 128 % 32])}
        for index in range(4_096)
    ]
    return {
        "factor": {
            "variables": [0, 1, 2],
            "domain_sizes": domain_sizes,
            "table": table,
        },
        "variable": 0,
    }


def test_public_multiply_admits_boundary_product_with_identity_factor() -> None:
    value = {"num": "9" * 256, "den": "1"}
    identity = {"num": "1", "den": "1"}
    payload = {
        "left": {
            "variables": [0],
            "domain_sizes": [2],
            "table": [value, value],
        },
        "right": {
            "variables": [0],
            "domain_sizes": [2],
            "table": [identity, identity],
        },
    }

    result = invoke_operation(
        "graphical_model.factor.multiply", payload, Catalog.open()
    )

    assert result.output["factor"]["table"] == [value, value]


def test_public_multiply_rejects_growth_before_output_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_value: object) -> CanonicalRational:
        raise AssertionError("multiply output was expanded before admission")

    monkeypatch.setattr(CanonicalRational, "from_fraction", fail)

    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            "graphical_model.factor.multiply",
            _large_integer_factor_payload(),
            Catalog.open(),
        )

    assert error.value.errors()[0]["type"] == (
        "graphical_model.factor_multiply_rational_bound"
    )


def test_public_marginalize_rejects_growth_before_output_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_value: object) -> CanonicalRational:
        raise AssertionError("marginal output was expanded before admission")

    monkeypatch.setattr(CanonicalRational, "from_fraction", fail)

    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            "graphical_model.factor.marginalize",
            _large_denominator_marginal_payload(),
            Catalog.open(),
        )

    assert error.value.errors()[0]["type"] == (
        "graphical_model.factor_marginalize_rational_bound"
    )
