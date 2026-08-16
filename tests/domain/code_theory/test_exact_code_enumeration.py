from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.code_theory import LinearCodeRequest
from jacobian.domains.code_theory.operations import (
    compute_min_distance,
    compute_weight_dist,
)


def test_prime_field_code_enumeration_uses_the_declared_matrix() -> None:
    request = LinearCodeRequest(field_order=2, generator_matrix=((1, 1),))

    assert compute_min_distance(request).minimum_distance == 2
    assert compute_weight_dist(request).weights == ((0, 1), (2, 1))


def test_code_contract_rejects_nonprime_fields_and_unbounded_enumeration() -> None:
    with pytest.raises(ValidationError, match="prime"):
        LinearCodeRequest(field_order=4, generator_matrix=((1,),))
    with pytest.raises(ValidationError, match="enumeration"):
        LinearCodeRequest(field_order=251, generator_matrix=((1,), (1,), (1,)))
