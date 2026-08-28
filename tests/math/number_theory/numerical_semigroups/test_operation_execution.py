"""Execution-path tests for numerical-semigroup operations."""

from typing import Any

import pytest

from jacobian.math.number_theory.numerical_semigroups import (
    _element_invariant_operations as element_operations,
)
from jacobian.math.number_theory.numerical_semigroups import (
    _global_invariant_operations as global_operations,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementDeltaSetRequest,
    ElementElasticityRequest,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_operations import (
    compute_element_catenary_degree,
    compute_element_delta_set,
    compute_element_elasticity,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_models import (
    BettiElementsRequest,
    DeltaSetRequest,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_operations import (
    compute_betti_elements,
    compute_delta_set,
)


@pytest.mark.parametrize(
    ("operation", "operation_request", "module", "kernel_name"),
    (
        (
            compute_element_delta_set,
            ElementDeltaSetRequest(generators=("3", "5"), value="15"),
            element_operations,
            "factorization_lengths",
        ),
        (
            compute_element_elasticity,
            ElementElasticityRequest(generators=("3", "5"), value="15"),
            element_operations,
            "factorization_length_extrema",
        ),
        (
            compute_element_catenary_degree,
            ElementCatenaryDegreeRequest(generators=("3", "5"), value="15"),
            element_operations,
            "factorizations",
        ),
        (
            compute_betti_elements,
            BettiElementsRequest(generators=("3", "5")),
            global_operations,
            "betti_data",
        ),
        (
            compute_delta_set,
            DeltaSetRequest(generators=("3", "5")),
            global_operations,
            "delta_periodicity_bound",
        ),
    ),
)
def test_trusted_semigroup_producers_run_each_expensive_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    operation_request: Any,
    module: Any,
    kernel_name: str,
) -> None:
    original = getattr(module, kernel_name)
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, kernel_name, counted)

    operation(operation_request)

    assert calls == 1
