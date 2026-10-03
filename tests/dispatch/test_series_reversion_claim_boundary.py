"""Public reversion outputs retain the native authored-relation contract."""

import json

from jacobian._exact import CanonicalRational as Q
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import reversion, verify_reversion
from jacobian.math.polynomials.series._models import (
    SeriesReversionResult,
    TruncatedSeries,
)


def test_public_reversion_matches_native_and_checks_authored_relation() -> None:
    source = TruncatedSeries(
        variable="q",
        truncation_order=8,
        coefficients=tuple(
            Q.from_integer_ratio(v, 1) for v in [0, 1, 1, 0, 0, 0, 0, 0]
        ),
    )
    native = reversion(source)
    response = invoke_operation(
        "formal_series.rational.reversion.compute",
        source.model_dump(mode="json"),
        Catalog.open(),
    )
    claim = SeriesReversionResult.model_validate_json(json.dumps(response.output))
    assert claim == native
    assert verify_reversion(claim)
    assert not verify_reversion(claim.model_copy(update={"left_residual": ()}))
    coefficients = list(claim.result.coefficients)
    coefficients[2] = Q.from_integer_ratio(0, 1)
    forged = claim.model_copy(
        update={
            "result": claim.result.model_copy(
                update={"coefficients": tuple(coefficients)}
            )
        }
    )
    decoded = SeriesReversionResult.model_validate_json(forged.model_dump_json())
    assert not verify_reversion(decoded)
