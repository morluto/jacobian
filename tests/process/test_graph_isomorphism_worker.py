"""Process-boundary behavior for graph-isomorphism workers."""

import pytest

import jacobian.process as process
from jacobian.math.graphs.isomorphism._models import GraphIsomorphismRequest
from jacobian.math.graphs.isomorphism._operations import decide_graph_isomorphism
from jacobian.math.number_field._models import NumberFieldRequest
from jacobian.math.number_field._operations import compute_nf_discriminant
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._direct_factorization_models import (
    FactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    enumerate_divisors,
    factorize_certified,
    factorize_primes,
)
from jacobian.process import BoundedProcessResult


def test_timed_out_vf2_worker_is_an_unknown_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = decide_graph_isomorphism(
        GraphIsomorphismRequest.model_validate(
            {
                "graph_a": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
                "graph_b": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
            }
        )
    )

    assert result.status == "UNKNOWN"
    assert result.vertex_mapping == ()


def test_timed_out_certified_factorization_is_an_unknown_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = factorize_certified(CertifiedFactorizationRequest(value="10403"))

    assert result.status == "UNKNOWN"
    assert result.factors == ()


def test_timed_out_direct_factorization_worker_is_an_unknown_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    request = FactorizationRequest(value="12")
    divisors = enumerate_divisors(request)
    factors = factorize_primes(request)

    assert divisors.status == "UNKNOWN"
    assert divisors.divisors == ()
    assert factors.status == "UNKNOWN"
    assert factors.factors == ()


def test_timed_out_number_field_worker_is_an_unknown_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = compute_nf_discriminant(
        NumberFieldRequest(coefficients_descending=("1", "0", "-2"), variable="x")
    )

    assert result.status == "UNKNOWN"
    assert result.discriminant is None
