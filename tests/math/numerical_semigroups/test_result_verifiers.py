"""Independent replay verification for numerical-semigroup result claims."""

from typing import Any

import pytest

from jacobian.math.numerical_semigroups import (
    _element_invariant_operations as element_operations,
)
from jacobian.math.numerical_semigroups import (
    _global_invariant_operations as global_operations,
)
from jacobian.math.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)
from jacobian.math.numerical_semigroups._element_invariant_operations import (
    _verify_element_catenary_degree_result,
    _verify_element_delta_set_result,
    _verify_element_elasticity_result,
    compute_element_catenary_degree,
    compute_element_delta_set,
    compute_element_elasticity,
)
from jacobian.math.numerical_semigroups._factorization_models import (
    FactorizationComputeRequest,
    FactorizationComputeResult,
    FactorizationGraphComputeRequest,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeRequest,
    FactorizationLengthsComputeResult,
)
from jacobian.math.numerical_semigroups._factorization_operations import (
    _verify_factorization_compute_result,
    _verify_factorization_graph_compute_result,
    _verify_factorization_lengths_compute_result,
    compute_factorization_graph,
    compute_factorization_lengths,
    compute_factorizations,
)
from jacobian.math.numerical_semigroups._global_invariant_models import (
    BettiElementsRequest,
    BettiElementsResult,
    CatenaryDegreeRequest,
    CatenaryDegreeResult,
    DeltaSetRequest,
    DeltaSetResult,
)
from jacobian.math.numerical_semigroups._global_invariant_operations import (
    compute_betti_elements,
    compute_catenary_degree,
    compute_delta_set,
    verify_betti_elements_result,
    verify_catenary_degree_result,
    verify_delta_set_result,
)
from jacobian.math.numerical_semigroups._presentation_models import (
    MinimalPresentationRequest,
    MinimalPresentationResult,
)
from jacobian.math.numerical_semigroups._presentation_operations import (
    compute_minimal_presentation,
    verify_minimal_presentation_result,
)


def _forged(result: Any, **updates: object) -> dict[str, Any]:
    return {**result.model_dump(), **updates}


def test_factorization_verifier_rejects_forged_complete_family() -> None:
    result = compute_factorizations(
        FactorizationComputeRequest(generators=("3", "5"), value="15")
    )
    forged = FactorizationComputeResult.model_validate(
        _forged(result, factorizations=((5, 0),))
    )

    assert not _verify_factorization_compute_result(forged)


@pytest.mark.parametrize(
    ("result_type", "payload"),
    (
        (
            FactorizationComputeResult,
            {
                "value": "100000000",
                "minimal_generators": ("2", "3"),
                "in_semigroup": False,
                "factorizations": (),
            },
        ),
        (
            FactorizationLengthsComputeResult,
            {
                "value": "100000000",
                "minimal_generators": ("2", "3"),
                "in_semigroup": False,
                "lengths": (),
            },
        ),
        (
            FactorizationGraphComputeResult,
            {
                "value": "100000000",
                "minimal_generators": ("2", "3"),
                "in_semigroup": False,
                "factorizations": (),
                "edges": (),
                "connected_components": (),
                "is_connected": True,
            },
        ),
    ),
)
def test_factorization_replay_results_reapply_request_value_envelopes(
    result_type: type[Any], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="value must be at most"):
        result_type.model_validate(payload)


@pytest.mark.parametrize(
    ("result_type", "payload"),
    (
        (
            ElementDeltaSetResult,
            {
                "value": "100000000",
                "minimal_generators": ("2", "3"),
                "factorization_lengths": (),
                "delta_set": (),
            },
        ),
        (
            ElementElasticityResult,
            {
                "value": "100000000",
                "minimal_generators": ("2", "3"),
                "minimum_length": 1,
                "maximum_length": 1,
                "elasticity": "1",
            },
        ),
    ),
)
def test_element_invariant_replay_results_reapply_request_value_envelopes(
    result_type: type[Any], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="value must be at most"):
        result_type.model_validate(payload)


@pytest.mark.parametrize(
    ("result_type", "payload", "verify"),
    (
        (
            FactorizationComputeResult,
            {
                "value": "9990",
                "minimal_generators": ("6", "10", "14", "15"),
                "in_semigroup": False,
                "factorizations": (),
            },
            _verify_factorization_compute_result,
        ),
        (
            FactorizationGraphComputeResult,
            {
                "value": "9990",
                "minimal_generators": ("6", "10", "14", "15"),
                "in_semigroup": False,
                "factorizations": (),
                "edges": (),
                "connected_components": (),
                "is_connected": True,
            },
            _verify_factorization_graph_compute_result,
        ),
        (
            ElementCatenaryDegreeResult,
            {
                "value": "9990",
                "minimal_generators": ("6", "10", "14", "15"),
                "factorization_count": 13_307_204,
                "catenary_degree": 6,
            },
            _verify_element_catenary_degree_result,
        ),
    ),
)
def test_materialization_results_parse_structurally_but_verification_reapplies_admission(
    result_type: type[Any], payload: dict[str, Any], verify: Any
) -> None:
    """Parsing a claim never counts or enumerates its factorization family."""

    result = result_type.model_validate(payload)

    assert not verify(result)


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


def test_factorization_length_verifier_rejects_forged_length_set() -> None:
    result = compute_factorization_lengths(
        FactorizationLengthsComputeRequest(generators=("3", "5"), value="15")
    )
    forged = FactorizationLengthsComputeResult.model_validate(
        _forged(result, lengths=(3,))
    )

    assert not _verify_factorization_lengths_compute_result(forged)


def test_element_invariant_verifiers_reject_forged_claims() -> None:
    delta = compute_element_delta_set(
        ElementDeltaSetRequest(generators=("3", "5"), value="15")
    )
    forged_delta = ElementDeltaSetResult.model_validate(
        _forged(delta, factorization_lengths=(3,))
    )
    elasticity = compute_element_elasticity(
        ElementElasticityRequest(generators=("3", "5"), value="15")
    )
    forged_elasticity = ElementElasticityResult.model_validate(
        _forged(
            elasticity,
            minimum_length=4,
            maximum_length=5,
            elasticity="5/4",
        )
    )
    catenary = compute_element_catenary_degree(
        ElementCatenaryDegreeRequest(generators=("3", "5"), value="15")
    )
    forged_catenary = ElementCatenaryDegreeResult.model_validate(
        _forged(catenary, catenary_degree=0)
    )

    assert not _verify_element_delta_set_result(forged_delta)
    assert not _verify_element_elasticity_result(forged_elasticity)
    assert _verify_element_catenary_degree_result(catenary)
    assert not _verify_element_catenary_degree_result(forged_catenary)


def test_factorization_graph_verifier_rejects_forged_edge_set() -> None:
    result = compute_factorization_graph(
        FactorizationGraphComputeRequest(generators=("3", "5"), value="15")
    )
    forged = FactorizationGraphComputeResult.model_validate(
        _forged(result, edges=((0, 1),))
    )

    assert not _verify_factorization_graph_compute_result(forged)


def test_global_invariant_verifiers_reject_forged_claims() -> None:
    betti = compute_betti_elements(BettiElementsRequest(generators=("3", "5")))
    forged_betti = BettiElementsResult.model_validate(
        _forged(betti, candidate_count=betti.candidate_count + 1)
    )
    delta = compute_delta_set(DeltaSetRequest(generators=("3", "5")))
    forged_delta = DeltaSetResult.model_validate(_forged(delta, periodicity_bound=1))
    catenary = compute_catenary_degree(CatenaryDegreeRequest(generators=("3", "5")))
    forged_catenary = CatenaryDegreeResult.model_validate(
        _forged(
            catenary,
            catenary_degree=0,
            betti_degrees=({"betti_element": "15", "catenary_degree": 0},),
            witness_betti_elements=(),
        )
    )

    assert not verify_betti_elements_result(forged_betti)
    assert not verify_delta_set_result(forged_delta)
    assert not verify_catenary_degree_result(forged_catenary)


def test_minimal_presentation_verifier_rejects_forged_relation() -> None:
    result = compute_minimal_presentation(
        MinimalPresentationRequest(generators=("3", "5"))
    )
    forged = MinimalPresentationResult.model_validate(
        _forged(result, relations=({"first": (5, 0), "second": (0, 3)},) * 2)
    )

    assert not verify_minimal_presentation_result(forged)
