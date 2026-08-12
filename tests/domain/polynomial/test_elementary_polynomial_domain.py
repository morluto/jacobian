from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial import build_polynomial_bundle
from tests.support.services import DomainTestServices, open_domain_services


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_polynomial_bundle()
    ) as services:
        yield services


def _polynomial(
    terms: list[tuple[int, int, int]],
    variable: str = "x",
) -> dict[str, object]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": [variable],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": [exponent],
                }
                for exponent, numerator, denominator in terms
            ]
        },
    }


def _invoke(
    domain_services: DomainTestServices,
    capability_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    outcome = domain_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    assert outcome.execution.status is ExecutionStatus.COMPLETED
    assert outcome.artifact_uris == ()
    return outcome.output["result"]


def test_integer_polynomial_operations_preserve_ring_semantics(
    domain_services,
) -> None:

    assert _invoke(
        domain_services,
        "polynomial.integer.compute.gcd",
        {
            "left": {"coefficients": ["6", "6", "0"]},
            "right": {"coefficients": ["8", "8", "0"]},
        },
    ) == {
        "gcd": {
            "coefficient_order": "DESCENDING_DEGREE",
            "coefficients": ["2", "2", "0"],
        },
        "left_content": "6",
        "right_content": "8",
        "gcd_content": "2",
        "normalization": "NONNEGATIVE_LEADING_COEFFICIENT",
    }
    assert (
        _invoke(
            domain_services,
            "polynomial.integer.compute.content",
            {"polynomial": {"coefficients": ["-6", "-6", "0"]}},
        )["content"]
        == "6"
    )

    # Content, primitive part, and normalization must retain an exact
    # reconstruction even when the source has a negative leading coefficient.
    primitive = _invoke(
        domain_services,
        "polynomial.integer.compute.primitive_part",
        {"polynomial": {"coefficients": ["-6", "-6", "0"]}},
    )
    assert primitive["content"] == "6"
    assert primitive["primitive_part"]["coefficients"] == ["-1", "-1", "0"]
    assert primitive["reconstruction"]["coefficients"] == ["-6", "-6", "0"]

    assert _invoke(
        domain_services,
        "polynomial.integer.compute.evaluate",
        {"polynomial": {"coefficients": ["2", "-3", "1"]}, "point": "4"},
    ) == {"point": "4", "value": "21"}
    assert _invoke(
        domain_services,
        "polynomial.integer.compute.compose",
        {
            "outer": {"coefficients": ["1", "0", "1"]},
            "inner": {"coefficients": ["1", "1"]},
        },
    )["composition"]["coefficients"] == ["1", "2", "2"]


def test_integer_polynomial_evaluation_formats_large_exact_result(
    domain_services: DomainTestServices,
) -> None:
    point = "9" * 256
    result = _invoke(
        domain_services,
        "polynomial.integer.compute.evaluate",
        {
            "polynomial": {"coefficients": ["1"] + (["0"] * 127)},
            "point": point,
        },
    )

    assert result["point"] == point
    assert len(result["value"]) > 4_300


def test_rational_polynomial_operations_return_typed_intermediates(
    domain_services,
) -> None:

    division = _invoke(
        domain_services,
        "polynomial.rational.compute.quotient_remainder",
        {
            "left": _polynomial([(2, 1, 1), (0, -1, 1)]),
            "right": _polynomial([(1, 1, 1), (0, -1, 1)]),
        },
    )
    assert division["quotient"] == _polynomial([(1, 1, 1), (0, 1, 1)])
    assert division["remainder"] == _polynomial([])
    assert division["reconstruction"] == _polynomial([(2, 1, 1), (0, -1, 1)])

    assert _invoke(
        domain_services,
        "polynomial.rational.compute.evaluate",
        {
            "polynomial": _polynomial([(2, 1, 2), (0, 1, 3)]),
            "point": {"num": "2", "den": "3"},
        },
    )["value"] == {"num": "5", "den": "9"}
    assert _invoke(
        domain_services,
        "polynomial.rational.compute.derivative",
        {"polynomial": _polynomial([(3, 1, 2), (1, -2, 1)])},
    )["derivative"] == _polynomial([(2, 3, 2), (0, -2, 1)])
    integral = _invoke(
        domain_services,
        "polynomial.rational.compute.integral",
        {"polynomial": _polynomial([(2, 3, 1), (0, 2, 1)])},
    )
    assert integral["antiderivative"] == _polynomial([(3, 1, 1), (1, 2, 1)])
    assert integral["integration_constant"] == "ZERO"


def test_partial_fraction_output_is_structured_and_reconstructs(
    domain_services,
) -> None:
    numerator = _polynomial([(2, 1, 1), (1, 2, 1), (0, 3, 1)])
    denominator = _polynomial([(3, 1, 1), (2, 4, 1), (1, 5, 1), (0, 2, 1)])

    result = _invoke(
        domain_services,
        "polynomial.rational.compute.partial_fraction_decomposition",
        {"numerator": numerator, "denominator": denominator},
    )

    assert result["polynomial_part"] == _polynomial([])
    assert result["reconstruction_numerator"] == numerator
    assert result["reconstruction_denominator"] == denominator
    assert result["terms"] == [
        {
            "numerator": _polynomial([(0, -2, 1)]),
            "denominator_factor": _polynomial([(1, 1, 1), (0, 1, 1)]),
            "denominator_exponent": 1,
        },
        {
            "numerator": _polynomial([(0, 2, 1)]),
            "denominator_factor": _polynomial([(1, 1, 1), (0, 1, 1)]),
            "denominator_exponent": 2,
        },
        {
            "numerator": _polynomial([(0, 3, 1)]),
            "denominator_factor": _polynomial([(1, 1, 1), (0, 2, 1)]),
            "denominator_exponent": 1,
        },
    ]


def test_partial_fraction_uses_the_declared_univariate_generator(
    domain_services,
) -> None:
    numerator = _polynomial([(1, 1, 1), (0, 3, 1)], "t")
    denominator = _polynomial([(2, 1, 1), (0, -1, 1)], "t")

    result = _invoke(
        domain_services,
        "polynomial.rational.compute.partial_fraction_decomposition",
        {"numerator": numerator, "denominator": denominator},
    )

    assert result["reconstruction_numerator"] == numerator
    assert result["reconstruction_denominator"] == denominator
    assert result["terms"] == [
        {
            "numerator": _polynomial([(0, 2, 1)], "t"),
            "denominator_factor": _polynomial([(1, 1, 1), (0, -1, 1)], "t"),
            "denominator_exponent": 1,
        },
        {
            "numerator": _polynomial([(0, -1, 1)], "t"),
            "denominator_factor": _polynomial([(1, 1, 1), (0, 1, 1)], "t"),
            "denominator_exponent": 1,
        },
    ]


def test_partial_fraction_normalizes_non_monic_denominators_exactly(
    domain_services,
) -> None:
    numerator = _polynomial([(2, 2, 1), (1, -3, 1), (0, 1, 1)], "t")
    denominator = _polynomial([(2, 6, 1), (1, -3, 1), (0, -3, 1)], "t")

    result = _invoke(
        domain_services,
        "polynomial.rational.compute.partial_fraction_decomposition",
        {"numerator": numerator, "denominator": denominator},
    )

    # The source cancels to (2t - 1)/(6t + 3). Both the structured
    # decomposition and the reconstruction use the same monic factor t + 1/2:
    # 1/3 - (1/3)/(t + 1/2) = (t/3 - 1/6)/(t + 1/2).
    assert result["polynomial_part"] == _polynomial([(0, 1, 3)], "t")
    assert result["terms"] == [
        {
            "numerator": _polynomial([(0, -1, 3)], "t"),
            "denominator_factor": _polynomial(
                [(1, 1, 1), (0, 1, 2)],
                "t",
            ),
            "denominator_exponent": 1,
        }
    ]
    assert result["reconstruction_numerator"] == _polynomial(
        [(1, 1, 3), (0, -1, 6)],
        "t",
    )
    assert result["reconstruction_denominator"] == _polynomial(
        [(1, 1, 1), (0, 1, 2)],
        "t",
    )


def test_elementary_polynomial_requests_fail_closed_before_artifact_writes(
    domain_services,
) -> None:
    outcome = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.integer.compute.compose",
            input={
                "outer": {"coefficients": ["1", *(["0"] * 63), "1"]},
                "inner": {"coefficients": ["1", *(["0"] * 63), "1"]},
            },
        )
    )

    assert outcome.execution.status is ExecutionStatus.ERROR
    assert outcome.diagnostics[0].code == "INVALID_POLYNOMIAL_REQUEST"
    assert outcome.artifact_uris == ()


def test_integer_polynomial_shift_is_exact_computed(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.integer.compute.shift",
            input={
                "polynomial": {
                    "coefficient_order": "DESCENDING_DEGREE",
                    "coefficients": ["1", "0"],
                },
                "shift": 2,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "shift": 2,
        "shifted": {
            "coefficient_order": "DESCENDING_DEGREE",
            "coefficients": ["1", "2"],
        },
        "convention": "SUBSTITUTE_X_PLUS_SHIFT",
    }
