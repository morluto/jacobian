"""Public-boundary evidence for the Gaussian projective cross-ratio."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

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
