"""Dispatch boundaries for principal Dirichlet characters."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def test_dispatch_rejects_overbound_character_values_before_result_construction() -> (
    None
):
    payload = {
        "character": {
            "group": {
                "modulus": 3,
                "unit_residues": [1, 2],
                "character_count": 2,
                "invariant_factors": [2],
                "generators": [2],
                "generator_orders": [2],
                "unit_coordinates": [[0], [1]],
                "exponent": 2,
            },
            "coordinates": [0],
        },
        "integer": "1" * 257,
    }
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation("dirichlet_character.value.compute", payload, Catalog.open())
    assert error.value.errors()[0]["type"] == "dirichlet_character.integer_digit_bound"


def test_dispatch_projects_forged_character_as_domain_validation_error() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "dirichlet_character.principal.value.compute",
            {
                "character": {
                    "modulus": 4,
                    "unit_residues": [1],
                    "values": [0, 1, 0, 0],
                },
                "integer": "1",
            },
            Catalog.open(),
        )

    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.unit_residues_mismatch"
    )
