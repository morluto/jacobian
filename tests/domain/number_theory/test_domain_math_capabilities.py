from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts import number_theory as number_theory_contracts
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityDiscoveryRequest,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.number_theory import (
    FiniteFieldPolynomialMapRequest,
    FiniteFieldPolynomialMapResult,
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import OperationInstaller
from jacobian.runtime.config import CheckerAuthorityMode


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_number_theory_bundle(),
    ) as services:
        yield services


def test_finite_field_fiber_histogram_has_contract_level_bound() -> None:
    schema = FiniteFieldPolynomialMapResult.model_json_schema()

    assert schema["properties"]["fiber_histogram"]["maxItems"] == 20_000


def _stored_modular_residue_result(
    domain_services: DomainTestServices,
    result,
) -> dict[str, object]:
    stored = domain_services.core.store.get(result.output["result_uri"])
    assert isinstance(stored.payload, dict)
    return stored.payload


def test_quadratic_residues_are_complete_for_the_modulus(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.enumerate.quadratic_residues",
            input={"modulus": 10},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"residues": ["0", "1", "4", "5", "6", "9"]}


def test_extended_gcd_returns_a_valid_bezout_identity(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.extended_gcd",
            input={"left": "84", "right": "30"},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "gcd": "6",
        "left_coefficient": "-1",
        "right_coefficient": "3",
    }


def test_domain_error_fails_before_artifact_writes(domain_services) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.compute.inverse",
            input={"value": "6", "modulus": 9},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "NUMBER_THEORY_OPERATION_NOT_APPLICABLE"
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.artifact_uris == ()


def test_number_theory_boundary_results(domain_services) -> None:
    empty_cases = (
        ("integer.compute.proper_divisors", {"value": "1"}, {"divisors": []}),
        ("integer.compute.proper_divisors", {"value": "-1"}, {"divisors": []}),
        ("integer.compute.prime_factorization", {"value": "1"}, {"factors": []}),
        ("integer.compute.prime_factorization", {"value": "-1"}, {"factors": []}),
    )
    for capability_id, payload, expected in empty_cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.output["result"] == expected

    for capability_id, payload in (
        ("integer.compute.divisors", {"value": "0"}),
        ("integer.compute.prime_factorization", {"value": "0"}),
        ("integer.compute.previous_prime", {"n": 2}),
    ):
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.ERROR
        assert result.artifact_uris == ()


def _cubic_residue_payload() -> dict[str, object]:
    return {
        "modulus": 7,
        "variables": [
            {"name": "x", "residues": [0, 1, 2, 3, 4, 5, 6]},
        ],
        "terms": [{"coefficient": "4", "exponents": [3]}],
    }


def test_modular_polynomial_residue_image_is_complete_and_materialized(
    domain_services,
) -> None:
    inline = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=_cubic_residue_payload(),
        )
    )
    assert inline.execution.status is ExecutionStatus.COMPLETED
    assert inline.artifact_uris == ()
    assert inline.output["result"]["table"] is None

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.assignments.materialize",
            input=_cubic_residue_payload(),
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.obligations == ()
    stored_result = _stored_modular_residue_result(domain_services, result)
    assert stored_result == {
        "semantics_version": "modular-polynomial-residue-image.v1",
        "modulus": 7,
        "variable_order": ["x"],
        "domains": [[0, 1, 2, 3, 4, 5, 6]],
        "normalized_terms": [{"coefficient": 4, "exponents": [3]}],
        "enumeration_scope": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
        "total_assignments": 7,
        "image": [0, 3, 4],
        "residue_counts": [
            {"residue": 0, "count": 1},
            {"residue": 3, "count": 3},
            {"residue": 4, "count": 3},
        ],
        "witnesses": [
            {"residue": 0, "assignment": [0]},
            {"residue": 3, "assignment": [3]},
            {"residue": 4, "assignment": [1]},
        ],
        "table": [
            {"assignment": [0], "residue": 0},
            {"assignment": [1], "residue": 4},
            {"assignment": [2], "residue": 4},
            {"assignment": [3], "residue": 3},
            {"assignment": [4], "residue": 4},
            {"assignment": [5], "residue": 3},
            {"assignment": [6], "residue": 3},
        ],
    }
    input_uri, result_uri = result.artifact_uris
    stored_artifact = domain_services.core.store.get(result_uri)
    assert stored_artifact.payload == stored_result
    assert stored_artifact.manifest.parents == (input_uri,)
    assert result.relationships[0].relation_id == (
        "modular.polynomial_residue_image.assignments.relation"
    )
    assert result.relationships[0].source_artifact_uris == (input_uri,)
    assert result.relationships[0].target_artifact_uris == (result_uri,)


def test_modular_polynomial_residue_image_handles_multivariate_domains(
    domain_services,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.assignments.materialize",
            input={
                "modulus": 5,
                "variables": [
                    {"name": "x", "residues": [0, 1, 2]},
                    {"name": "y", "residues": [0, 2]},
                ],
                "terms": [
                    {"coefficient": "-1", "exponents": [0, 1]},
                    {"coefficient": "1", "exponents": [2, 0]},
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = _stored_modular_residue_result(domain_services, result)
    assert output["normalized_terms"] == [
        {"coefficient": 4, "exponents": [0, 1]},
        {"coefficient": 1, "exponents": [2, 0]},
    ]
    assert output["total_assignments"] == 6
    assert output["table"] == [
        {"assignment": [0, 0], "residue": 0},
        {"assignment": [0, 2], "residue": 3},
        {"assignment": [1, 0], "residue": 1},
        {"assignment": [1, 2], "residue": 4},
        {"assignment": [2, 0], "residue": 4},
        {"assignment": [2, 2], "residue": 2},
    ]
    assert output["image"] == [0, 1, 2, 3, 4]


def test_modular_polynomial_residue_result_rejects_an_incomplete_table(
    domain_services,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.assignments.materialize",
            input=_cubic_residue_payload(),
        )
    )
    corrupted = deepcopy(_stored_modular_residue_result(domain_services, result))
    corrupted["table"].pop()

    with pytest.raises(ValidationError, match="complete table length"):
        ModularPolynomialResidueImageResult.model_validate(corrupted)


def test_modular_polynomial_residue_result_rejects_oversized_domains_before_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_product(*_: object) -> object:
        raise AssertionError("oversized Cartesian domains must not be materialized")

    monkeypatch.setattr(number_theory_contracts, "product", unexpected_product)
    domains = [list(range(32))] * 6

    with pytest.raises(ValidationError, match="result domains exceed"):
        ModularPolynomialResidueImageResult.model_validate(
            {
                "semantics_version": "modular-polynomial-residue-image.v1",
                "modulus": 32,
                "variable_order": ["a", "b", "c", "d", "e", "f"],
                "domains": domains,
                "normalized_terms": [],
                "enumeration_scope": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
                "total_assignments": 1,
                "image": [0],
                "residue_counts": [{"residue": 0, "count": 1}],
                "witnesses": [{"residue": 0, "assignment": [0, 0, 0, 0, 0, 0]}],
                "table": [{"assignment": [0, 0, 0, 0, 0, 0], "residue": 0}],
            }
        )


def test_modular_polynomial_residue_image_reproduces_divisibility_polynomial(
    domain_services,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.assignments.materialize",
            input={
                "modulus": 7**7,
                "variables": [
                    {"name": "a", "residues": [18]},
                    {"name": "b", "residues": [1]},
                ],
                "terms": [
                    {"coefficient": "7", "exponents": [1, 6]},
                    {"coefficient": "21", "exponents": [2, 5]},
                    {"coefficient": "35", "exponents": [3, 4]},
                    {"coefficient": "35", "exponents": [4, 3]},
                    {"coefficient": "21", "exponents": [5, 2]},
                    {"coefficient": "7", "exponents": [6, 1]},
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = _stored_modular_residue_result(domain_services, result)
    assert output["modulus"] == 823543
    assert output["total_assignments"] == 1
    assert output["image"] == [0]
    assert output["witnesses"] == [{"residue": 0, "assignment": [18, 1]}]
    assert output["table"] == [{"assignment": [18, 1], "residue": 0}]


def _permutation_trinomial_payload() -> dict[str, object]:
    return {
        "characteristic": 11,
        "modulus_coefficients_ascending": [2, 1, 0, 0, 1],
        "terms": [
            {"coefficient": [1, 0, 0, 0], "exponent": 131},
            {"coefficient": [1, 0, 0, 0], "exponent": 1211},
            {"coefficient": [10, 0, 0, 0], "exponent": 1331},
        ],
    }


def test_extension_field_permutation_map_is_complete_and_verified(
    tmp_path: Path,
) -> None:
    with open_domain_services(
        tmp_path / "verified-extension-field",
        build_number_theory_bundle(),
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as domain_services:
        bundle = build_number_theory_bundle()
        installed = OperationInstaller(
            domain_services.core.store,
            domain_services.core.schemas,
            domain_services.core.artifacts,
        ).install(bundle)
        adapters, _installation = install_exact_domain_verification(
            domain_services.core.store,
            domain_services.core.schemas,
            domain_services.core.artifacts,
            domain_services.installation.verification,
            domain_services.core.checkers,
            bundles={"number_theory": (bundle, installed)},
            authorize=True,
        )
        for adapter in adapters:
            domain_services.installation.register_capability(adapter)
        payload = _permutation_trinomial_payload()
        computed = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="finite_field.polynomial_map.fibers.compute",
                input=payload,
            )
        )
        assert computed.execution.status is ExecutionStatus.COMPLETED
        assert computed.output["result"] == {
            "semantics_version": "finite-field-polynomial-map.v1",
            "characteristic": 11,
            "extension_degree": 4,
            "modulus_coefficients_ascending": [2, 1, 0, 0, 1],
            "terms": payload["terms"],
            "enumeration_order": "BASE_P_LEAST_SIGNIFICANT_COEFFICIENT_FIRST",
            "total_inputs": 14641,
            "distinct_outputs": 14641,
            "collision_excess": 0,
            "fiber_histogram": [{"fiber_size": 1, "output_count": 14641}],
            "is_permutation": True,
            "first_collision": None,
            "output_sequence_sha256": (
                "6b30b8bcd45642e6320e79cffb547f4fb8b74291a5c2476e010403b94c2d57a6"
            ),
        }

        verified = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="finite_field.polynomial_map.fibers.verify",
                mode=CapabilityMode.VERIFY,
                input={"input": payload, "candidate": computed.output["result"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"


def test_extension_field_request_rejects_unverifiable_replay_cost() -> None:
    with pytest.raises(ValidationError, match="exhaustive replay exceeds"):
        FiniteFieldPolynomialMapRequest.model_validate(
            {
                "characteristic": 139,
                "modulus_coefficients_ascending": [1, 0, 1],
                "terms": [
                    {"coefficient": [1, 0], "exponent": 999_999_900 + index}
                    for index in range(32)
                ],
            }
        )


def test_extension_field_map_reports_first_collision(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.polynomial_map.fibers.compute",
            input={
                "characteristic": 3,
                "modulus_coefficients_ascending": [1, 0, 1],
                "terms": [{"coefficient": [1, 0], "exponent": 2}],
            },
        )
    )
    result = computed.output["result"]
    assert result["is_permutation"] is False
    assert result["distinct_outputs"] == 5
    assert result["collision_excess"] == 4
    assert result["first_collision"] is not None


def test_extension_field_map_rejects_reducible_modulus_without_artifacts(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.polynomial_map.fibers.compute",
            input={
                "characteristic": 3,
                "modulus_coefficients_ascending": [2, 0, 1],
                "terms": [{"coefficient": [1, 0], "exponent": 1}],
            },
        )
    )
    assert computed.execution.status is ExecutionStatus.ERROR
    assert computed.artifact_uris == ()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "modulus": 7,
            "variables": [{"name": "x", "residues": [0, 1]}],
            "terms": [{"coefficient": "1", "exponents": [1, 0]}],
        },
        {
            "modulus": 7,
            "variables": [{"name": "x", "residues": [0, 1]}],
            "terms": [{"coefficient": "7", "exponents": [1]}],
        },
        {
            "modulus": 17,
            "variables": [
                {"name": "x", "residues": list(range(17))},
                {"name": "y", "residues": list(range(17))},
                {"name": "z", "residues": list(range(17))},
            ],
            "terms": [{"coefficient": "1", "exponents": [1, 1, 1]}],
        },
    ],
)
def test_modular_polynomial_residue_image_rejects_invalid_scope_before_writes(
    domain_services,
    payload: dict[str, object],
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=payload,
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_NUMBER_THEORY_REQUEST"
    assert result.artifact_uris == ()


def test_modular_polynomial_residue_image_assignment_bound_is_exact() -> None:
    accepted = ModularPolynomialResidueImageRequest.model_validate(
        {
            "modulus": 16,
            "variables": [
                {"name": "x", "residues": list(range(16))},
                {"name": "y", "residues": list(range(16))},
                {"name": "z", "residues": list(range(16))},
            ],
            "terms": [{"coefficient": "1", "exponents": [1, 1, 1]}],
        }
    )

    assert len(accepted.variables) == 3
    assert accepted.modulus == 16


def test_modular_polynomial_residue_image_is_discoverable_by_intent(
    domain_services,
) -> None:
    discovered = domain_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "complete sparse polynomial residue image modulo an integer "
                "with witnesses and an exhaustive table"
            ),
            domain="modular",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == (
        "modular.polynomial_residue_image.compute"
    )
    assert discovered.matches[0].has_invocation_examples is True
