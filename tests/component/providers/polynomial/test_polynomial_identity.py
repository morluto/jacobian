"""Rational-function and polynomial identity capability tests."""

from __future__ import annotations

from tests.component.providers.polynomial.polynomial_capabilities_support import (
    identity_input as _identity_input,
)
from tests.component.providers.polynomial.polynomial_capabilities_support import (
    rational_function_identity_input as _rational_function_identity_input,
)

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import Conclusion, InputStatus


def test_rational_function_identity_cross_multiplies_exactly(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.rational_function.identity.verify",
            input=_rational_function_identity_input(),
        )
    )

    assert result.output["identical"] is True
    assert result.output["conclusion"] == Conclusion.TRUE.value
    assert result.output["equality_semantics"] == (
        "QQ_FRACTION_FIELD_CROSS_MULTIPLICATION"
    )
    assert result.verification_record_uri is not None

    semantics_uri = runtime.polynomial.rational_function_identity_semantics_uri
    semantics = runtime.core.store.get(semantics_uri)
    assert semantics.payload["definition"]["pointwise_definedness"] == "outside scope"
    for output_key in ("left_uri", "right_uri", "claim_uri", "certificate_uri"):
        artifact = runtime.core.store.get(result.output[output_key])
        assert artifact.manifest.semantics_uri == semantics_uri


def test_rational_function_identity_reports_exact_difference(
    authorized_polynomial_services,
) -> None:
    result = authorized_polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.rational_function.identity.verify",
            input=_rational_function_identity_input(equal=False),
        )
    )

    assert result.output["identical"] is False
    assert result.output["conclusion"] == Conclusion.FALSE.value
    assert result.verification_record_uri is not None


def test_rational_function_identity_rejects_zero_denominator(
    authorized_polynomial_services,
) -> None:
    request = _rational_function_identity_input()
    request["left"]["denominator"] = {"terms": []}
    result = authorized_polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.rational_function.identity.verify",
            input=request,
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_RATIONAL_FUNCTION_IDENTITY_REQUEST"


def test_rational_function_identity_preserves_checker_rejection_as_unknown(
    revocable_polynomial_services,
) -> None:
    runtime = revocable_polynomial_services
    checker_id = runtime.polynomial.rational_function_identity_checker_id
    assert checker_id is not None
    runtime.core.checkers.revoke(checker_id, reason="exercise fail-closed projection")
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.rational_function.identity.verify",
            input=_rational_function_identity_input(),
        )
    )

    assert result.output["identical"] is None
    assert result.output["conclusion"] == Conclusion.UNKNOWN.value
    assert result.output["verification_record_uri"] is None


def test_polynomial_identity_descriptor_example_is_directly_invocable(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    descriptor = descriptors["polynomial.identity.verify"]
    example = descriptor.invocation_examples[0]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=example.input,
        )
    )

    assert result.verification_record_uri is not None


def test_polynomial_identity_verifies_equal_coefficients(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.identity.verify",
            input=_identity_input(),
        )
    )

    assert result.output["identical"] is True
    assert result.output["conclusion"] == "TRUE"
    assert result.verification_record_uri is not None
    assert result.output["left_uri"] != result.output["right_uri"]

    semantics_uri = runtime.polynomial.identity_semantics_uri
    assert semantics_uri != runtime.polynomial.semantics_uri
    semantics = runtime.core.store.get(semantics_uri)
    assert semantics.payload["name"] == "jacobian.sparse-rational-polynomial-ring"
    for output_key in ("left_uri", "right_uri", "claim_uri", "certificate_uri"):
        artifact = runtime.core.store.get(result.output[output_key])
        assert artifact.manifest.semantics_uri == semantics_uri
    record = runtime.core.store.get(result.output["verification_record_uri"])
    assert (
        record.payload["bindings"]["semantics_digest"]
        == semantics.manifest.object_digest
    )
    assert record.payload["relationship_source_artifact_uris"] == [
        result.output["left_uri"]
    ]
    assert record.payload["relationship_target_artifact_uris"] == [
        result.output["right_uri"]
    ]
    assert record.payload["obligation_uri"] is None
    checker = runtime.core.checkers.get(result.output["checker_id"])
    assert checker.semantics_uris == (semantics_uri,)

    rejected = runtime.application.verification.verify_certificate(
        certificate_uri=result.output["certificate_uri"],
        checker_id=result.output["checker_id"],
        supporting_artifact_uris=(result.output["claim_uri"],),
    )
    assert rejected.input.status is InputStatus.REJECTED
    assert rejected.conclusion is Conclusion.UNKNOWN
    assert rejected.verification_record_uri is None


def test_polynomial_identity_verifies_a_difference(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.identity.verify",
            input=_identity_input(right_coefficient=3),
        )
    )

    assert result.output["identical"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.verification_record_uri is not None
    assert result.output["first_coefficient_mismatch"] == {
        "exponents": [2, 0],
        "left_coefficient": {"num": "2", "den": "1"},
        "right_coefficient": {"num": "3", "den": "1"},
        "left_minus_right": {"num": "-1", "den": "1"},
    }
    record = runtime.core.store.get(result.output["verification_record_uri"])
    assert record.payload["conclusion"] == Conclusion.FALSE.value
    assert record.payload["relation_id"] is None


def test_polynomial_identity_canonicalizes_duplicate_terms(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.identity.verify",
            input={
                "variables": ["x"],
                "left": {
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                        {
                            "coefficient": {"num": "-1", "den": "1"},
                            "exponents": [1],
                        },
                    ]
                },
                "right": {"terms": []},
            },
        )
    )

    assert result.execution.status.value == "COMPLETED"
    assert result.output["identical"] is True
    assert result.output["conclusion"] == Conclusion.TRUE.value
    assert result.verification_record_uri is not None


def test_polynomial_identity_preserves_checker_rejection_as_unknown(
    revocable_polynomial_services,
) -> None:
    runtime = revocable_polynomial_services
    checker_id = runtime.polynomial.identity_checker_id
    assert checker_id is not None
    runtime.core.checkers.revoke(checker_id, reason="exercise fail-closed projection")

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.identity.verify",
            input=_identity_input(),
        )
    )

    assert result.output["identical"] is None
    assert result.output["conclusion"] == Conclusion.UNKNOWN.value
    assert result.output["verification_record_uri"] is None
