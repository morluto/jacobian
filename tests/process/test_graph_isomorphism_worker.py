"""Process-boundary behavior for graph-isomorphism workers."""

import pytest

import jacobian.process as process
from jacobian.math.graphs.isomorphism import _vf2_process as isomorphism_operations
from jacobian.math.graphs.isomorphism._models import GraphIsomorphismRequest
from jacobian.math.graphs.isomorphism._vf2_process import decide_graph_isomorphism
from jacobian.math.graphs.isomorphism._vf2_worker import _first_isomorphism_mapping
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._direct_factorization_models import (
    FactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
    _FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    enumerate_divisors,
    factorize_certified,
    factorize_primes,
)
from jacobian.math.number_theory.number_fields import (
    _discriminant_process as number_field_operations,
)
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def test_vf2_worker_obtains_a_positive_witness_in_one_search_traversal() -> None:
    class Matcher:
        def __init__(self) -> None:
            self.traversals = 0

        def isomorphisms_iter(self):  # type: ignore[no-untyped-def]
            self.traversals += 1
            yield {1: 0, 0: 1}

    matcher = Matcher()

    assert _first_isomorphism_mapping(matcher) == [(0, 1), (1, 0)]
    assert matcher.traversals == 1


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


def test_vf2_worker_has_private_cwd_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def complete_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        return BoundedProcessResult(
            returncode=0,
            stdout=b'{"ok":true,"mapping":[[0,0],[1,1]]}',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", complete_worker)

    result = decide_graph_isomorphism(
        GraphIsomorphismRequest.model_validate(
            {
                "graph_a": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
                "graph_b": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
            }
        )
    )

    assert result.status == "ISOMORPHIC"
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=isomorphism_operations._VF2_ADDRESS_SPACE_BYTES,
        file_size_bytes=isomorphism_operations._VF2_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-vf2-")


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


def test_factorization_workers_have_private_cwds_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, object]] = []

    def timed_out_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.append(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(process, "run_bounded_process", timed_out_worker)

    certified = factorize_certified(CertifiedFactorizationRequest(value="10403"))
    direct = factorize_primes(FactorizationRequest(value="12"))

    assert certified.status == "UNKNOWN"
    assert direct.status == "UNKNOWN"
    assert len(recorded) == 2
    for invocation, prefix in zip(
        recorded,
        ("jacobian-certified-factor-", "jacobian-direct-factor-"),
        strict=True,
    ):
        assert invocation["resource_limits"] == ProcessResourceLimits(
            cpu_seconds=60,
            address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
            file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
        )
        assert str(invocation["cwd"]).split("/")[-1].startswith(prefix)


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


def test_number_field_worker_has_private_cwd_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def complete_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        return BoundedProcessResult(
            returncode=0,
            stdout=b'{"kind":"complete","discriminant":"8"}',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", complete_worker)

    result = compute_nf_discriminant(
        NumberFieldRequest(coefficients_descending=("1", "0", "-2"), variable="x")
    )

    assert result.discriminant == "8"
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=number_field_operations._WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=number_field_operations._WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-number-field-")


def test_number_field_worker_start_failure_is_an_unknown_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        raise OSError("worker unavailable")

    monkeypatch.setattr(process, "run_bounded_process", unavailable)

    result = compute_nf_discriminant(
        NumberFieldRequest(coefficients_descending=("1", "0", "-2"), variable="x")
    )

    assert result.status == "UNKNOWN"
    assert result.detail == "the bounded number-field worker could not be started"
