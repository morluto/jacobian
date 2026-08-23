"""Dispatch execution tests for certified factoring (moved from math to satisfy ownership)."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_math_run_executes_certified_compute() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "integer.factor.certified_compute", {"value": "360"}, catalog
    )
    assert result.output["status"] == "COMPLETE"
    primes = [int(f["prime"]) for f in result.output["factors"]]
    assert primes == [2, 3, 5]


def test_math_run_executes_pratt_certificate() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "integer.primality.certificate.compute", {"value": "101"}, catalog
    )
    assert result.output["status"] == "CERTIFIED"
    assert result.output["certificate"]["prime"] == "101"
