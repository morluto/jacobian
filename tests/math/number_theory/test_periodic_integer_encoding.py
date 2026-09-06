"""Periodic scalar schemas preserve signs and the distinct period/count envelopes."""

from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceUnionSource,
    PeriodicNonnegativeInteger,
    PeriodicPositiveInteger,
)
from jacobian.math.number_theory.periodic_interval_count._models import (
    MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS,
)
from jacobian.math.number_theory.periodic_interval_count.operations import (
    compute_periodic_interval_count,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
)


@pytest.mark.parametrize(
    ("annotation", "minimum"),
    [(PeriodicPositiveInteger, 1), (PeriodicNonnegativeInteger, 0)],
)
def test_periodic_sign_constraints_match_native_json_and_schema(
    annotation: Any, minimum: int
) -> None:
    adapter = TypeAdapter(annotation)
    assert adapter.validate_python(minimum) == minimum
    assert adapter.validate_json(encode_strict_json(str(minimum))) == minimum
    with pytest.raises(ValidationError):
        adapter.validate_python(minimum - 1)
    with pytest.raises(ValidationError):
        adapter.validate_json(encode_strict_json(str(minimum - 1)))
    for mode in ("validation", "serialization"):
        schema = Draft202012Validator(adapter.json_schema(mode=mode))
        assert schema.is_valid(str(minimum))
        assert not schema.is_valid(str(minimum - 1))
        assert not schema.is_valid(minimum)


def test_large_prefix_count_uses_cutoff_not_period_envelope() -> None:
    source = PeriodicCongruenceUnionSource(
        subsets=(PeriodicCongruenceSubset(modulus=1, residues=(0,)),),
        complement=False,
    )
    result = compute_periodic_union_prefix_count(source, 10**500)
    assert result.count == result.cutoff == 10**500
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_interval_preserves_full_signed_endpoint_envelope() -> None:
    source = PeriodicCongruenceUnionSource(subsets=(), complement=True)
    endpoint = 10**MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS - 1
    result = compute_periodic_interval_count(source, -endpoint, endpoint)
    assert result.count == 2 * endpoint + 1
    assert type(result).model_validate_json(result.model_dump_json()) == result
