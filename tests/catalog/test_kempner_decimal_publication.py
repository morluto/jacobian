"""Publication check for the published dense Kempner decimal enclosure.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.kempner._models import (
    KempnerDecimalEnclosure,
    KempnerDecimalEnclosureRequest,
    KempnerDigitSet,
)
from jacobian.math.number_theory.kempner.operations import (
    enclose_kempner_series_decimal,
)


def test_fixed_point_dense_operation_is_published_and_native_exported() -> None:
    digit_set = KempnerDigitSet(base=4, allowed_digits=(0, 1, 2))
    request = KempnerDecimalEnclosureRequest(digit_set=digit_set, cutoff=3, precision=8)
    native = enclose_kempner_series_decimal(digit_set, 3, 8)
    tool = Catalog.open().operation("number_theory.kempner_series.enclose_decimal")
    assert tool is not None

    published = tool.run(request)

    assert isinstance(published, KempnerDecimalEnclosure)
    assert published == native
    assert (
        KempnerDecimalEnclosure.model_validate_json(published.model_dump_json())
        == published
    )
