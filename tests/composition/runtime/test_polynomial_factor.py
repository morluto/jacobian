from __future__ import annotations

from tests.support.polynomials import univariate_term as _term

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.runtime.model import JacobianRuntime


def test_factor_compute_preserves_multiplicity_and_reconstructs_exactly(
    attached_complete_runtime: JacobianRuntime,
) -> None:
    runtime = attached_complete_runtime
    polynomial = {
        "terms": [
            _term(1, 2),
            _term(-2, 1),
            _term(1, 0),
        ]
    }

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.factor.compute",
            input={"variable": "x", "polynomial": polynomial},
        )
    )

    assert result.output["coefficient"] == {"num": "1", "den": "1"}
    assert len(result.output["factors"]) == 1
    assert result.output["factors"][0]["multiplicity"] == 2
    assert result.output["reconstructed"] == polynomial
    assert result.output["product_reconstruction"] == "EXACT"
    assert result.output["irreducibility_verification"] == "UNVERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    source = runtime.core.store.get(result.output["source_polynomial_uri"])
    factorization = runtime.core.store.get(result.output["factorization_uri"])
    assert (
        source.manifest.semantics_uri
        == runtime.portfolio.polynomial.polynomial_semantics_uri
    )
    assert (
        factorization.manifest.semantics_uri
        == runtime.portfolio.polynomial.factorization_semantics_uri
    )
    assert source.manifest.semantics_uri != factorization.manifest.semantics_uri


def test_factor_compute_handles_zero_as_a_coefficient_not_a_unit(
    attached_complete_runtime: JacobianRuntime,
) -> None:
    runtime = attached_complete_runtime
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.factor.compute",
            input={"variable": "x", "polynomial": {"terms": []}},
        )
    )

    assert result.output["coefficient"] == {"num": "0", "den": "1"}
    assert "unit" not in result.output
    assert result.output["factors"] == []
    assert result.output["reconstructed"] == {"terms": []}
    assert result.output["irreducibility_verification"] == "UNVERIFIED"


def test_factor_compute_preserves_rational_coefficient_and_irreducible_factor(
    attached_complete_runtime: JacobianRuntime,
) -> None:
    runtime = attached_complete_runtime
    polynomial = {
        "terms": [
            {
                "coefficient": {"num": "-3", "den": "2"},
                "exponents": [2],
            },
            {
                "coefficient": {"num": "-3", "den": "2"},
                "exponents": [0],
            },
        ]
    }

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.factor.compute",
            input={"variable": "x", "polynomial": polynomial},
        )
    )

    assert result.output["coefficient"] == {"num": "-3", "den": "2"}
    assert result.output["factors"] == [
        {
            "factor": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2],
                    },
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [0],
                    },
                ]
            },
            "multiplicity": 1,
        }
    ]
    assert result.output["reconstructed"] == polynomial
