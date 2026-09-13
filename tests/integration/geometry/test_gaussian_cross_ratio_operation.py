"""Public-boundary evidence for the Gaussian projective cross-ratio."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation

OPERATION_ID = "geometry.projective_line.cross_ratio.gaussian_rational.compute"


def _z(real: int, imaginary: int = 0) -> dict[str, dict[str, str]]:
    return {
        "real": {"num": str(real), "den": "1"},
        "imaginary": {"num": str(imaginary), "den": "1"},
    }


def test_cross_ratio_survives_strict_public_dispatch() -> None:
    result = invoke_operation(
        OPERATION_ID,
        {
            "first": {"coordinates": [_z(0), _z(1)]},
            "second": {"coordinates": [_z(1), _z(1)]},
            "third": {"coordinates": [_z(2), _z(1)]},
            "fourth": {"coordinates": [_z(1), _z(0)]},
        },
        Catalog.open(),
    )
    assert result.operation_id == OPERATION_ID
    assert result.output == {
        "real": {"num": "2", "den": "1"},
        "imaginary": {"num": "0", "den": "1"},
    }


def test_coincident_points_reach_operation_admission_on_dispatch() -> None:
    payload = {
        "first": {
            "coordinates": [
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "second": {
            "coordinates": [
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "third": {
            "coordinates": [
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "fourth": {
            "coordinates": [
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
    }
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "geometry.projective_line.cross_ratio.gaussian_rational.compute",
            payload,
            Catalog.open(),
        )
    assert not isinstance(error.value, OperationRequestValidationError)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.points_not_distinct"
    )
