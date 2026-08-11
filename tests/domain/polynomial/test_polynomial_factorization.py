from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.contracts.capabilities import CapabilityAssuranceLevel
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial import build_polynomial_bundle
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.services import DomainTestServices, open_domain_services


@pytest.fixture
def polynomial_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_polynomial_bundle()
    ) as services:
        yield services


def _polynomial(
    terms: list[tuple[int, int, int]],
) -> dict[str, object]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": [degree],
                }
                for degree, numerator, denominator in terms
            ]
        },
    }


def test_factor_compute_preserves_multiplicity_and_reconstructs_exactly(
    polynomial_services: DomainTestServices,
) -> None:
    polynomial = _polynomial([(2, 1, 1), (1, -2, 1), (0, 1, 1)])

    result = _invoke(
        polynomial_services,
        "polynomial.factor.compute",
        {"polynomial": polynomial},
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()
    assert result.output["result"] == {
        "coefficient": {"num": "1", "den": "1"},
        "factors": [
            {
                "factor": _polynomial([(1, 1, 1), (0, -1, 1)]),
                "multiplicity": 2,
            }
        ],
        "reconstructed": polynomial,
        "normalization": "CONTENT_AND_MONIC_IRREDUCIBLES",
        "irreducibility_assurance": "UNVERIFIED",
        "product_reconstruction": "EXACT",
    }


def test_factor_compute_handles_zero_as_a_coefficient_not_a_unit(
    polynomial_services: DomainTestServices,
) -> None:
    result = _invoke(
        polynomial_services,
        "polynomial.factor.compute",
        {"polynomial": _polynomial([])},
    )

    assert result.output["result"] == {
        "coefficient": {"num": "0", "den": "1"},
        "factors": [],
        "reconstructed": _polynomial([]),
        "normalization": "CONTENT_AND_MONIC_IRREDUCIBLES",
        "irreducibility_assurance": "UNVERIFIED",
        "product_reconstruction": "EXACT",
    }


def test_factor_compute_preserves_rational_coefficient_and_irreducible_factor(
    polynomial_services: DomainTestServices,
) -> None:
    polynomial = _polynomial([(2, -3, 2), (0, -3, 2)])

    result = _invoke(
        polynomial_services,
        "polynomial.factor.compute",
        {"polynomial": polynomial},
    )

    assert result.output["result"] == {
        "coefficient": {"num": "-3", "den": "2"},
        "factors": [
            {
                "factor": _polynomial([(2, 1, 1), (0, 1, 1)]),
                "multiplicity": 1,
            }
        ],
        "reconstructed": polynomial,
        "normalization": "CONTENT_AND_MONIC_IRREDUCIBLES",
        "irreducibility_assurance": "UNVERIFIED",
        "product_reconstruction": "EXACT",
    }
