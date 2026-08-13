from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.capability_errors import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityRequest,
)
from jacobian.contracts.results import VerificationResult
from jacobian.domains.finite_fields import build_finite_field_bundle
from jacobian.domains.finite_fields.contracts import (
    FiniteMapTableRequest,
    LinearMapRankRequest,
    RestrictScalarsRequest,
)
from jacobian.exact_domain_checkers import ExactComputedVerificationAdapter
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
    ProjectivePoint,
    element,
    finite_field,
    finite_polynomial,
    finite_polynomial_map,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

pytestmark = pytest.mark.requires_provider("flint")


def _omit_presentation_defaults(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _omit_presentation_defaults(item)
            for key, item in value.items()
            if key not in {"generator", "element_encoding_version"}
        }
    if isinstance(value, list):
        return [_omit_presentation_defaults(item) for item in value]
    return value


def _request() -> LinearMapRankRequest:
    presentation = FiniteFieldPresentation(
        characteristic=2,
        modulus_coefficients=(1, 1, 1),
    )
    axis = Axis(name="b", labels=("b1",))
    direction = ProjectivePoint(
        presentation=presentation,
        axis=axis,
        coordinates=(
            FiniteFieldElement(presentation=presentation, coordinates=(1, 0)),
        ),
    )
    linear_map = FiniteLinearMap(
        source_axis=Axis(name="source", labels=("B1",)),
        target_axis=Axis(name="target", labels=("y1", "y2")),
        matrix=PrimeFieldMatrix(prime=2, entries=((1,), (0,)), columns=1),
    )
    return LinearMapRankRequest(direction=direction, linear_map=linear_map)


def _cross_field_rank_payload() -> dict[str, object]:
    request = _request()
    payload = request.model_dump(mode="json")
    linear_map = payload["linear_map"]
    assert isinstance(linear_map, dict)
    matrix = linear_map["matrix"]
    assert isinstance(matrix, dict)
    matrix["prime"] = 3
    return payload


def _restriction_request() -> RestrictScalarsRequest:
    presentation = FiniteFieldPresentation(
        characteristic=2,
        modulus_coefficients=(1, 1, 1),
    )
    row_axis = Axis(name="b", labels=("b1",))
    column_axis = Axis(name="y", labels=("y1",))
    basis_axis = Axis(name="basis", labels=("B1",))
    one = FiniteFieldElement(presentation=presentation, coordinates=(1, 0))
    a = FiniteFieldElement(presentation=presentation, coordinates=(0, 1))
    subspace = FiniteDimensionalSubspace(
        presentation=presentation,
        basis_axis=basis_axis,
        basis=(
            AxisBoundMatrix(
                presentation=presentation,
                row_axis=row_axis,
                column_axis=column_axis,
                entries=((a,),),
            ),
        ),
    )
    direction = ProjectivePoint(
        presentation=presentation,
        axis=row_axis,
        coordinates=(one,),
    )
    return RestrictScalarsRequest(subspace=subspace, direction=direction)


def test_rank_request_rejects_cross_field_values_before_execution(
    finite_field_services,
) -> None:
    result = finite_field_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.compute",
            input=_cross_field_rank_payload(),
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_FINITE_FIELD_REQUEST"
    assert result.diagnostics[0].stage == "finite_field_input_validation"
    assert set(result.output) == {"error"}
    assert result.verification_record_uri is None
    assert result.artifact_uris == ()


def test_operator_authorized_sympy_replay_accepts_rank_and_rejects_forgery(
    finite_field_services,
) -> None:
    request = _request()
    input_payload = request.model_dump(mode="json")

    services = finite_field_services
    computed = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.compute",
            input=input_payload,
        )
    )
    candidate = computed.output["result"]
    verified = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.verify",
            input={"input": input_payload, "candidate": candidate},
        )
    )
    forged = {**candidate, "rank": 0}
    rejected = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.verify",
            input={"input": input_payload, "candidate": forged},
        )
    )

    assert computed.output["result"]["rank"] == 1
    assert computed.verification_record_uri is None
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["verification_record_uri"] is None


@pytest.mark.parametrize(
    ("field", "unrelated"),
    [
        ("operation_id", "matrix.rank.compute"),
        ("checker_id", "checker://sha256/" + "0" * 64),
        ("claim_digest", "sha256:" + "1" * 64),
        ("semantics_digest", "sha256:" + "2" * 64),
        ("candidate_digest", "sha256:" + "3" * 64),
    ],
)
def test_inline_verification_record_rejects_unrelated_projected_identity(
    finite_field_services,
    field: str,
    unrelated: str,
) -> None:
    request = _request()
    input_payload = request.model_dump(mode="json")

    services = finite_field_services
    computed = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.compute",
            input=input_payload,
        )
    )
    verified = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.linear_map.rank.verify",
            input={"input": input_payload, "candidate": computed.output["result"]},
        )
    )
    forged = verified.model_copy(
        update={"output": {**verified.output, field: unrelated}}
    )

    expected = field[:-3] if field.endswith("_id") else field
    with pytest.raises(CapabilityError, match=f"different {expected}"):
        services.core.capabilities._validate_verified_result(forged)


def test_inline_verification_record_rejects_stale_candidate_binding(
    finite_field_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    input_payload = _omit_presentation_defaults(request.model_dump(mode="json"))
    assert isinstance(input_payload, dict)

    services = finite_field_services
    try:
        computed = services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="finite_field.linear_map.rank.compute",
                input=input_payload,
            )
        )
        adapter = services.core.capabilities._adapters[
            "finite_field.linear_map.rank.verify"
        ]
        assert isinstance(adapter, ExactComputedVerificationAdapter)
        verify_inline_exact = adapter.verification.verify_inline_exact
        accepted: list[VerificationResult] = []

        def capture_accepted_result(**kwargs: Any) -> VerificationResult:
            result = verify_inline_exact(**kwargs)
            accepted.append(result)
            return result

        monkeypatch.setattr(
            adapter.verification,
            "verify_inline_exact",
            capture_accepted_result,
        )
        verified = services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="finite_field.linear_map.rank.verify",
                input={
                    "input": input_payload,
                    "candidate": computed.output["result"],
                },
            )
        )
        assert verified.verification_record_uri is not None
        assert len(accepted) == 1
        monkeypatch.setattr(
            adapter.verification,
            "verify_inline_exact",
            lambda **_: accepted[0],
        )
        stale_request = CapabilityRequest(
            capability_id="finite_field.linear_map.rank.verify",
            input={
                "input": input_payload,
                "candidate": {**computed.output["result"], "rank": 0},
            },
        )

        rejected = services.core.capabilities.invoke(stale_request)
    finally:
        monkeypatch.undo()

    assert rejected.execution.status == "ERROR"
    assert rejected.verification_record_uri is None
    assert rejected.execution.detail is not None
    assert "does not bind the verified values" in rejected.execution.detail


def test_operator_authorized_sympy_replay_checks_restriction_of_scalars(
    finite_field_services,
) -> None:
    request = _restriction_request()
    input_payload = request.model_dump(mode="json")

    services = finite_field_services
    computed = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.restrict_scalars.compute",
            input=input_payload,
        )
    )
    candidate = computed.output["result"]
    verified = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.restrict_scalars.verify",
            input={"input": input_payload, "candidate": candidate},
        )
    )
    forged = deepcopy(candidate)
    forged["matrix"]["entries"] = [[1], [0]]
    rejected = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.restrict_scalars.verify",
            input={"input": input_payload, "candidate": forged},
        )
    )

    assert candidate["matrix"]["entries"] == [[0], [1]]
    assert verified.output["status"] == "VERIFIED"
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["verification_record_uri"] is None


def test_operator_authorized_sympy_replay_checks_complete_polynomial_table(
    finite_field_services,
) -> None:
    presentation = finite_field(2, (1, 1, 1))
    zero = element(presentation, (0, 0))
    one = element(presentation, (1, 0))
    request = FiniteMapTableRequest(
        polynomial_map=finite_polynomial_map(
            finite_polynomial(presentation, (zero, zero, zero, one))
        )
    )
    input_payload = request.model_dump(mode="json")

    services = finite_field_services
    computed = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.polynomial_map.table.compute",
            input=input_payload,
        )
    )
    candidate = computed.output["result"]
    verified = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.polynomial_map.table.verify",
            input={"input": input_payload, "candidate": candidate},
        )
    )
    forged = deepcopy(candidate)
    forged["entries"][2][1] = forged["entries"][0][1]
    rejected = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_field.polynomial_map.table.verify",
            input={"input": input_payload, "candidate": forged},
        )
    )

    assert verified.output["status"] == "VERIFIED"
    assert rejected.output["status"] == "REJECTED"


def test_missing_flint_omits_only_flint_operations_and_their_checkers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.providers.flint_runtime import (
        python_flint_finite_field_provider_runtime,
    )

    unavailable = python_flint_finite_field_provider_runtime().model_copy(
        update={
            "availability": CapabilityProviderAvailability.UNAVAILABLE,
            "version": None,
            "digest": None,
            "digest_kind": None,
            "diagnostic": "python-flint is not installed",
        }
    )
    monkeypatch.setattr(
        "jacobian.domains.finite_fields.bundle."
        "python_flint_finite_field_provider_runtime",
        lambda: unavailable,
    )

    with open_exact_domain_services(
        tmp_path,
        build_finite_field_bundle(),
    ) as services:
        ids = {
            descriptor.capability_id
            for descriptor in services.core.capabilities.catalog().capabilities
        }

    assert {
        "finite_field.projective_line.enumerate",
        "finite_field.polynomial_map.fibers.compute",
        "finite_field.polynomial_map.collision.compute",
        "finite_field.polynomial_map.permutation.compute",
        "finite_field.polynomial_map.fibers.verify",
        "finite_field.polynomial_map.collision.verify",
        "finite_field.polynomial_map.permutation.verify",
    } <= ids
    assert {
        "finite_field.restrict_scalars.compute",
        "finite_field.restrict_scalars.verify",
        "finite_field.linear_map.rank.compute",
        "finite_field.linear_map.rank.verify",
        "finite_field.direction_rank_ledger.compute",
        "finite_field.orbit_distribution.compute",
        "finite_field.polynomial_map.table.compute",
        "finite_field.polynomial_map.table.verify",
    }.isdisjoint(ids)
