"""Dispatch boundaries for principal Dirichlet characters."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


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
