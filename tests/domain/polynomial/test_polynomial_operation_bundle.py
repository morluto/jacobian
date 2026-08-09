from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial import build_polynomial_bundle
from jacobian.process_policy import ProcessResult, ProcessTermination


@pytest.fixture
def polynomial_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install only the polynomial operations exercised by this module."""

    with open_domain_services(
        tmp_path / "state", build_polynomial_bundle()
    ) as services:
        yield services


def _polynomial(
    variables: list[str],
    terms: list[tuple[int | tuple[int, ...], int, int]],
) -> dict[str, object]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": variables,
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": (
                        [exponents] if isinstance(exponents, int) else list(exponents)
                    ),
                }
                for exponents, numerator, denominator in terms
            ]
        },
    }


def test_polynomial_bundle_installs_and_computes_exact_invariants(
    polynomial_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = polynomial_services

    installed_ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    assert {
        "polynomial.compute.gcd",
        "polynomial.compute.resultant",
        "polynomial.compute.discriminant",
        "polynomial.compute.square_free_decomposition",
        "polynomial.groebner_basis.compute",
    } <= installed_ids

    gcd_result = _invoke(
        runtime,
        "polynomial.compute.gcd",
        {
            "left": _polynomial(["x"], [(2, 1, 1), (0, -1, 1)]),
            "right": _polynomial(
                ["x"],
                [(2, 1, 1), (1, -3, 1), (0, 2, 1)],
            ),
        },
    )
    assert gcd_result.execution.status is ExecutionStatus.COMPLETED
    assert gcd_result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert gcd_result.output["result"] == {
        "gcd": _polynomial(["x"], [(1, 1, 1), (0, -1, 1)]),
        "bezout": {
            "left_multiplier": _polynomial(["x"], [(0, 1, 3)]),
            "right_multiplier": _polynomial(["x"], [(0, -1, 3)]),
        },
        "normalization": "MONIC",
    }

    resultant_result = _invoke(
        runtime,
        "polynomial.compute.resultant",
        {
            "left": _polynomial(
                ["x", "y"],
                [((1, 1), 1, 1), ((0, 0), 1, 1)],
            ),
            "right": _polynomial(
                ["x", "y"],
                [((1, 0), 1, 1), ((0, 1), 1, 1)],
            ),
            "elimination_variable": "x",
        },
    )
    assert resultant_result.output["result"] == {
        "elimination_variable": "x",
        "resultant": {
            "kind": "POLYNOMIAL",
            "value": _polynomial(
                ["y"],
                [(2, 1, 1), (0, -1, 1)],
            ),
        },
        "convention": "SYLVESTER_DETERMINANT",
    }

    discriminant_result = _invoke(
        runtime,
        "polynomial.compute.discriminant",
        {
            "polynomial": _polynomial(
                ["x", "y"],
                [((2, 0), 1, 1), ((0, 1), -1, 1)],
            ),
            "variable": "x",
        },
    )
    assert discriminant_result.output["result"]["discriminant"] == {
        "kind": "POLYNOMIAL",
        "value": _polynomial(["y"], [(1, 4, 1)]),
    }

    square_free_result = _invoke(
        runtime,
        "polynomial.compute.square_free_decomposition",
        {
            "polynomial": _polynomial(
                ["x"],
                [
                    (5, 1, 1),
                    (4, 1, 1),
                    (3, -5, 1),
                    (2, -1, 1),
                    (1, 8, 1),
                    (0, -4, 1),
                ],
            )
        },
    )
    square_free = square_free_result.output["result"]
    assert square_free["coefficient"] == {"num": "1", "den": "1"}
    assert square_free["factors"] == [
        {
            "factor": _polynomial(["x"], [(1, 1, 1), (0, 2, 1)]),
            "multiplicity": 2,
        },
        {
            "factor": _polynomial(["x"], [(1, 1, 1), (0, -1, 1)]),
            "multiplicity": 3,
        },
    ]
    assert square_free["reconstructed"] == _polynomial(
        ["x"],
        [
            (5, 1, 1),
            (4, 1, 1),
            (3, -5, 1),
            (2, -1, 1),
            (1, 8, 1),
            (0, -4, 1),
        ],
    )

    groebner_result = _invoke(
        runtime,
        "polynomial.groebner_basis.compute",
        {
            "generators": [
                _polynomial(
                    ["x", "y"],
                    [((1, 1), 1, 1), ((0, 0), -1, 1)],
                ),
                _polynomial(
                    ["x", "y"],
                    [((1, 0), -1, 1), ((0, 2), 1, 1)],
                ),
            ],
            "monomial_order": "lex",
            "resource_budget": {
                "wall_seconds": 10,
                "maximum_basis_polynomials": 64,
                "maximum_output_terms": 1024,
            },
        },
    )
    assert groebner_result.execution.status is ExecutionStatus.COMPLETED
    assert groebner_result.output == {
        "variables": ["x", "y"],
        "monomial_order": "lex",
        "basis": [
            _polynomial(
                ["x", "y"],
                [((1, 0), 1, 1), ((0, 2), -1, 1)],
            ),
            _polynomial(
                ["x", "y"],
                [((0, 3), 1, 1), ((0, 0), -1, 1)],
            ),
        ],
        "completion": "COMPLETE",
        "normalization": "REDUCED_MONIC",
    }
    assert len(groebner_result.artifact_uris) == 3
    obligation = runtime.core.store.get(groebner_result.artifact_uris[2])
    assert obligation.payload["verification_status"] == "UNVERIFIED"
    assert obligation.manifest.parents == tuple(
        sorted(
            (
                groebner_result.artifact_uris[0],
                groebner_result.artifact_uris[1],
            )
        )
    )

    for result in (
        gcd_result,
        resultant_result,
        discriminant_result,
        square_free_result,
    ):
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert result.artifact_uris == ()
        assert result.relationships == ()

    invalid_result = _invoke(
        runtime,
        "polynomial.compute.gcd",
        {
            "left": _polynomial(["x"], [(1, 1, 1)]),
            "right": _polynomial(["y"], [(1, 1, 1)]),
        },
    )

    assert invalid_result.execution.status is ExecutionStatus.ERROR
    assert invalid_result.diagnostics[0].code == "INVALID_POLYNOMIAL_REQUEST"
    assert invalid_result.artifact_uris == ()

    zero_multiplier_result = _invoke(
        runtime,
        "polynomial.compute.gcd",
        {
            "left": _polynomial(["x"], [(1, 1, 1), (0, -1, 1)]),
            "right": _polynomial(["x"], [(1, 1, 1), (0, -1, 1)]),
        },
    )
    assert zero_multiplier_result.execution.status is ExecutionStatus.COMPLETED
    multipliers = zero_multiplier_result.output["result"]["bezout"]
    assert (
        multipliers["left_multiplier"]["polynomial"]["terms"] == []
        or multipliers["right_multiplier"]["polynomial"]["terms"] == []
    )

    monkeypatch.setattr(
        "jacobian.domains.polynomial.groebner.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    timeout_result = _invoke(
        runtime,
        "polynomial.groebner_basis.compute",
        {
            "generators": [_polynomial(["x"], [(2, 1, 1), (0, 1, 1)])],
            "resource_budget": {
                "wall_seconds": 1,
                "maximum_basis_polynomials": 64,
                "maximum_output_terms": 1024,
            },
        },
    )

    assert timeout_result.execution.status is ExecutionStatus.TIMEOUT
    assert timeout_result.diagnostics[0].code == "POLYNOMIAL_GROEBNER_TIMEOUT"
    assert timeout_result.artifact_uris == ()


def test_groebner_result_preserves_advertised_input_variable_bound(
    attached_complete_runtime,
) -> None:
    variables = ["a", "b", "c", "d", "e"]
    result = _invoke(
        attached_complete_runtime,
        "polynomial.groebner_basis.compute",
        {
            "generators": [
                _polynomial(variables, [((0, 0, 0, 0, 0), 1, 1)]),
            ],
            "monomial_order": "lex",
            "resource_budget": {
                "wall_seconds": 10,
                "maximum_basis_polynomials": 64,
                "maximum_output_terms": 1024,
            },
        },
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output == {
        "variables": variables,
        "monomial_order": "lex",
        "basis": [_polynomial(variables, [((0, 0, 0, 0, 0), 1, 1)])],
        "completion": "COMPLETE",
        "normalization": "REDUCED_MONIC",
    }


def test_polynomial_output_budget_failure_is_explicit_and_writes_no_artifacts(
    polynomial_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = polynomial_services
    artifact_writes: list[object] = []
    original_put = cast(Any, runtime.core.artifacts.put)

    def recording_put(*args: object, **kwargs: object) -> object:
        artifact_writes.append((args, kwargs))
        return original_put(*args, **kwargs)

    monkeypatch.setattr(runtime.core.artifacts, "put", recording_put)
    monkeypatch.setattr(
        "jacobian.domains.polynomial.operations._MAX_OUTPUT_TERMS",
        0,
    )

    result = _invoke(
        runtime,
        "polynomial.compute.gcd",
        {
            "left": _polynomial(["x"], [(2, 1, 1), (0, -1, 1)]),
            "right": _polynomial(["x"], [(1, 1, 1), (0, -1, 1)]),
        },
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "POLYNOMIAL_OUTPUT_LIMIT_EXCEEDED"
    assert result.diagnostics[0].stage == "polynomial_output_validation"
    assert result.artifact_uris == ()
    assert artifact_writes == []


def test_groebner_result_budget_failure_crosses_worker_protocol(
    polynomial_services,
) -> None:
    runtime = polynomial_services

    result = _invoke(
        runtime,
        "polynomial.groebner_basis.compute",
        {
            "generators": [
                _polynomial(["x"], [(1, 1, 1), (0, 1, 1)]),
            ],
            "resource_budget": {
                "wall_seconds": 10,
                "maximum_basis_polynomials": 64,
                "maximum_output_terms": 1,
            },
        },
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED"
    assert result.artifact_uris == ()
