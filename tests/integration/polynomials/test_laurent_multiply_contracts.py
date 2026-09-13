"""Native/catalog parity for rational Laurent multiplication (#3615)."""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


def test_catalog_discovers_laurent_multiplication_vocabulary() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(need="rational Laurent polynomial multiplication")
    )

    assert (
        result.matches[0].operation_id == "polynomial.laurent.rational.multiply.compute"
    )
